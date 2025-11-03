import os
import time
import threading
import queue
import logging
from concurrent import futures

import numpy as np
if not hasattr(np, 'int'):
    np.int = int

import grpc
import torch
import torch.nn.functional as F

import xai_service_pb2_grpc
import xai_service_pb2
from xai_service_pb2_grpc import ExplanationsServicer
import pandas as pd
from modules.lib import _load_model, _load_dataset
from ExplainabilityMethodsRepository.ExplanationsHandler import (
    PDPHandler,
    TwoDPDPHandler,
    ALEHandler,
    CounterfactualsHandler,
    PrototypesHandler,
    GLANCEHandler,
    SegmentationAttributionHandler,
    SHAPHandler,
    FeatureImportanceHandler,  # experimentExplanation FI
)
from ExplainabilityMethodsRepository.config import shared_resources
from ExplainabilityMethodsRepository.src.glance.iterative_merges.iterative_merges import apply_action_pandas
from ExplainabilityMethodsRepository.segmentation import df_to_instances, replacement_feature_importance_batched
from sklearn.inspection import permutation_importance

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)


def safe_downsample(inputs, targets, masks, factor=4):
    if factor == 1:
        return inputs, targets, masks
    N, T, C, H, W = inputs.shape
    Hn, Wn = H // factor, W // factor
    inputs_ = inputs.view(N * T, C, H, W)
    inputs_ds = F.interpolate(inputs_, size=(Hn, Wn), mode='bilinear', align_corners=False)
    inputs_ds = inputs_ds.view(N, T, C, Hn, Wn)
    targets_ds = F.interpolate(targets, size=(Hn, Wn), mode='bilinear', align_corners=False)
    masks_ds = F.interpolate(masks.float(), size=(Hn, Wn), mode='nearest').long()
    return inputs_ds, targets_ds, masks_ds


def make_env(state, pct, msg, explanation=None, feat_imp=None):
    env = xai_service_pb2.StreamEnvelope(
        progress=xai_service_pb2.Progress(
            state=state,
            percent=int(max(0, min(100, pct))),
            message=str(msg or "")
        )
    )
    if explanation is not None:
        env.explanation.CopyFrom(explanation)
    if feat_imp is not None:
        env.feature_importance.CopyFrom(feat_imp)
    return env


class ExplainabilityExecutor(ExplanationsServicer):
    def __init__(self):
        self.dispatch_table = {
            ('featureExplanation', 'pdp'): PDPHandler(),
            ('featureExplanation', '2dpdp'): TwoDPDPHandler(),
            ('featureExplanation', 'ale'): ALEHandler(),
            ('featureExplanation', 'counterfactuals'): CounterfactualsHandler(),
            ('hyperparameterExplanation', 'pdp'): PDPHandler(),
            ('hyperparameterExplanation', '2dpdp'): TwoDPDPHandler(),
            ('hyperparameterExplanation', 'ale'): ALEHandler(),
            ('hyperparameterExplanation', 'counterfactuals'): CounterfactualsHandler(),
            ('experimentExplanation', 'pdp'): PDPHandler(),
            ('experimentExplanation', '2dpdp'): TwoDPDPHandler(),
            ('experimentExplanation', 'ale'): ALEHandler(),
            ('experimentExplanation', 'counterfactuals'): CounterfactualsHandler(),
            ('featureExplanation', 'prototypes'): PrototypesHandler(),
            ('featureExplanation', 'global_counterfactuals'): GLANCEHandler(),
            ('featureExplanation', 'segmentation'): SegmentationAttributionHandler(),
            ('featureExplanation', 'shap'): SHAPHandler(),
            ('experimentExplanation', 'feature_importance'): FeatureImportanceHandler(),
        }

    # ----------------- Existing unary (unchanged) -----------------
    def GetExplanation(self, request, context):
        logger.info(f"Unary GetExplanation: {request.explanation_type}/{request.explanation_method}")
        explanation_type = request.explanation_type
        explanation_method = request.explanation_method

        handler = self.dispatch_table.get((explanation_type, explanation_method))
        if handler:
            return handler.handle(request, explanation_type)  # no progress
        raise ValueError(f"Unsupported explanation method '{explanation_method}' for type '{explanation_type}'")

    def GetFeatureImportance(self, request, context):
        # Keep your current unary FI for compatibility
        model_path = request.model
        fi_type = request.type
        model, name = _load_model(model_path[0])

        train_data = _load_dataset(request.data.X_train)
        train_labels = _load_dataset(request.data.Y_train)
        test_data = _load_dataset(request.data.X_test)
        test_labels = _load_dataset(request.data.Y_test)

        if fi_type == 'FeatureImportance':
            if name in ('sklearn', 'tensorflow'):
                result = permutation_importance(model, test_data, test_labels, scoring='accuracy', n_repeats=5, random_state=42)
                feature_importances = list(zip(test_data.columns, result.importances_mean))
                sorted_features = sorted(feature_importances, key=lambda x: x[1], reverse=True)
            elif name == 'pytorch':
                df = pd.concat([train_data, train_labels], axis="columns")
                df = df[df["instance_id"] == df["instance_id"].iloc[0]]
                instances, lons, lats, labels = df_to_instances(df, C=4, patch_size=(512, 512))
                feature_names = ["dem", "mask", "wd_in", "rain"]
                inputs = torch.tensor(instances[0], dtype=torch.float32).unsqueeze(0)
                labels_t = torch.tensor(labels[0], dtype=torch.float32).unsqueeze(0)
                masks = inputs[:, [0], 1, :, :]
                device = "cpu"
                flood_threshold = 0.01
                inputs, labels_t, masks = safe_downsample(inputs, labels_t, masks, factor=4)
                rmse_differences, csi_differences = replacement_feature_importance_batched(
                    model=model, inputs=inputs.to(device), targets=labels_t.to(device), masks=masks.to(device),
                    flooded_min=flood_threshold, flooded_max=5.0, n_trials=2, batch_size=1, device=device,
                )
                mean_rmse = rmse_differences.mean(axis=1)
                sorted_features = sorted(list(zip(feature_names, mean_rmse)), key=lambda x: x[1], reverse=True)
            else:
                raise ValueError(f"Unknown backend for FI: {name}")
            return xai_service_pb2.FeatureImportanceResponse(
                feature_importances=[
                    xai_service_pb2.FeatureImportance(feature_name=f, importance_score=float(s))
                    for f, s in sorted_features
                ]
            )

        elif fi_type == 'SHAP':
            import shap
            explainer = shap.Explainer(model)
            ex = explainer(train_data)
            vals = ex.values
            feature_names = list(ex.feature_names)
            def mean_abs_shap(v): return np.abs(v).mean(axis=0)
            if vals.ndim == 2:
                imp = mean_abs_shap(vals)
                order = np.argsort(imp)[::-1]
                sorted_features = [(feature_names[i], float(imp[i])) for i in order]
            else:
                sorted_features = []
            return xai_service_pb2.FeatureImportanceResponse(
                feature_importances=[
                    xai_service_pb2.FeatureImportance(feature_name=f, importance_score=s)
                    for f, s in sorted_features
                ]
            )
        else:
            raise ValueError(f"Unsupported FI type: {fi_type}")

    def ApplyAffectedActions(self, request, context):
        try:
            affected = shared_resources.get("affected")
            clusters_res = shared_resources.get("clusters_res")
            affected_clusters = shared_resources.get("affected_clusters")
            if affected is None or clusters_res is None or affected_clusters is None:
                raise ValueError("Required shared resources are missing")

            sorted_actions_dict = dict(sorted(clusters_res.items(), key=lambda item: item[1]['cost']))
            actions = [stats["action"] for i, stats in sorted_actions_dict.items()]

            num_features = affected._get_numeric_data().columns.to_list()
            cate_features = affected.columns.difference(num_features)

            applied_affected = pd.DataFrame()
            for i, val in enumerate(list(affected_clusters.Chosen_Action.unique())):
                aff = affected_clusters[affected_clusters['Chosen_Action'] == val]
                if val != '-':
                    applied_df = apply_action_pandas(
                        aff[affected.columns.to_list()],
                        actions[int(val - 1)],
                        num_features,
                        cate_features,
                        '-',
                    )
                    applied_df['Chosen_Action'] = val
                    applied_affected = pd.concat([applied_affected, applied_df])
                else:
                    aff['Chosen_Action'] = '-'
                    cols = affected.columns.to_list()
                    cols.append('Chosen_Action')
                    applied_affected = pd.concat([applied_affected, aff[cols]])

            applied_affected = applied_affected.sort_index()
            shared_resources['applied_affected'] = applied_affected

            applied_affected_response = {}
            for i, col in enumerate(applied_affected.columns):
                applied_affected_response[col] = xai_service_pb2.TableContents(
                    index=i + 1,
                    values=applied_affected[col].astype(str).tolist(),
                )

            return xai_service_pb2.ApplyAffectedActionsResponse(
                applied_affected_actions=applied_affected_response
            )

        except Exception as e:
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return xai_service_pb2.ApplyAffectedActionsResponse()

    # ----------------- NEW streaming: GetExplanationStream -----------------
    def GetExplanationStream(self, request, context):
        q = queue.Queue()
        done = {"flag": False}
        final_result = {"resp": None}
        err = {"exc": None}

        def progress_cb(percent, message, partial_result=None, state=xai_service_pb2.Progress.RUNNING):
            try:
                env = make_env(
                    state=xai_service_pb2.Progress.PARTIAL if partial_result is not None else state,
                    pct=percent,
                    msg=message,
                    explanation=partial_result
                )
                q.put(env)
            except Exception as e:
                logger.exception("Enqueue progress failed: %s", e)

        def worker():
            try:
                explanation_type = request.explanation_type
                explanation_method = request.explanation_method
                handler = self.dispatch_table.get((explanation_type, explanation_method))
                if not handler:
                    raise ValueError(f"Unsupported explanation method '{explanation_method}' for type '{explanation_type}'")
                q.put(make_env(xai_service_pb2.Progress.RUNNING, 5, "Loading data/model"))
                result = handler.handle(request, explanation_type, progress_cb=progress_cb)
                final_result["resp"] = result
            except Exception as e:
                err["exc"] = e
            finally:
                done["flag"] = True

        threading.Thread(target=worker, daemon=True).start()

        yield make_env(xai_service_pb2.Progress.QUEUED, 0, "Queued")
        last_hb = time.time()
        HEARTBEAT = 5

        while True:
            try:
                env = q.get(timeout=0.5)
                yield env
            except queue.Empty:
                pass

            now = time.time()
            if not done["flag"] and now - last_hb >= HEARTBEAT:
                last_hb = now
                yield make_env(xai_service_pb2.Progress.RUNNING, 10, "Still running…")

            if done["flag"]:
                if err["exc"] is not None:
                    yield make_env(xai_service_pb2.Progress.ERROR, 100, f"Error: {err['exc']}")
                    context.set_details(str(err["exc"]))
                    context.set_code(grpc.StatusCode.INTERNAL)
                else:
                    yield make_env(xai_service_pb2.Progress.DONE, 100, "Done", explanation=final_result["resp"])
                break

    # ----------------- NEW streaming: GetFeatureImportanceStream -----------------
    def GetFeatureImportanceStream(self, request, context):
        q = queue.Queue()
        done = {"flag": False}
        final_result = {"resp": None}
        err = {"exc": None}

        def progress_cb(percent, message, state=xai_service_pb2.Progress.RUNNING):
            try:
                q.put(make_env(state=state, pct=percent, msg=message))
            except Exception as e:
                logger.exception("Enqueue FI progress failed: %s", e)

        def worker():
            try:
                model_path = request.model
                fi_type = request.type
                model, name = _load_model(model_path[0])

                train_data = _load_dataset(request.data.X_train)
                train_labels = _load_dataset(request.data.Y_train)
                test_data = _load_dataset(request.data.X_test)
                test_labels = _load_dataset(request.data.Y_test)

                progress_cb(10, "Preparing data/model")

                if fi_type == 'FeatureImportance':
                    if name in ('sklearn', 'tensorflow'):
                        progress_cb(40, "Permutation importance")
                        result = permutation_importance(model, test_data, test_labels, scoring='accuracy', n_repeats=5, random_state=42)
                        feature_importances = list(zip(test_data.columns, result.importances_mean))
                        sorted_features = sorted(feature_importances, key=lambda x: x[1], reverse=True)
                    elif name == 'pytorch':
                        progress_cb(35, "Preparing tensors")
                        df = pd.concat([train_data, train_labels], axis="columns")
                        df = df[df["instance_id"] == df["instance_id"].iloc[0]]
                        instances, lons, lats, labels = df_to_instances(df, C=4, patch_size=(512, 512))
                        feature_names = ["dem", "mask", "wd_in", "rain"]
                        inputs = torch.tensor(instances[0], dtype=torch.float32).unsqueeze(0)
                        labels_t = torch.tensor(labels[0], dtype=torch.float32).unsqueeze(0)
                        masks = inputs[:, [0], 1, :, :]
                        device = "cpu"
                        flood_threshold = 0.01
                        inputs, labels_t, masks = safe_downsample(inputs, labels_t, masks, factor=4)
                        progress_cb(70, "Replacement feature importance (batched)")
                        rmse_differences, csi_differences = replacement_feature_importance_batched(
                            model=model, inputs=inputs.to(device), targets=labels_t.to(device), masks=masks.to(device),
                            flooded_min=flood_threshold, flooded_max=5.0, n_trials=2, batch_size=1, device=device,
                        )
                        mean_rmse = rmse_differences.mean(axis=1)
                        sorted_features = sorted(list(zip(feature_names, mean_rmse)), key=lambda x: x[1], reverse=True)
                    else:
                        raise ValueError(f"Unknown backend for FI: {name}")

                    progress_cb(90, "Formatting")
                    final_result["resp"] = xai_service_pb2.FeatureImportanceResponse(
                        feature_importances=[
                            xai_service_pb2.FeatureImportance(feature_name=f, importance_score=float(s))
                            for f, s in sorted_features
                        ]
                    )

                elif fi_type == 'SHAP':
                    import shap
                    progress_cb(40, "Initializing SHAP")
                    explainer = shap.Explainer(model)
                    progress_cb(70, "Computing SHAP")
                    ex = explainer(train_data)
                    vals = ex.values
                    feature_names = list(ex.feature_names)
                    def mean_abs_shap(v): return np.abs(v).mean(axis=0)
                    shap_list = []
                    if vals.ndim == 2:
                        imp = mean_abs_shap(vals)
                        order = np.argsort(imp)[::-1]
                        shap_list = [(feature_names[i], float(imp[i])) for i in order]
                    progress_cb(90, "Formatting")
                    final_result["resp"] = xai_service_pb2.FeatureImportanceResponse(
                        feature_importances=[
                            xai_service_pb2.FeatureImportance(feature_name=f, importance_score=s)
                            for f, s in shap_list
                        ]
                    )
                else:
                    raise ValueError(f"Unsupported FI type: {fi_type}")
            except Exception as e:
                err["exc"] = e
            finally:
                done["flag"] = True

        threading.Thread(target=worker, daemon=True).start()

        yield make_env(xai_service_pb2.Progress.QUEUED, 0, "Queued")
        last_hb = time.time()
        HEARTBEAT = 5

        while True:
            try:
                env = q.get(timeout=0.5)
                yield env
            except queue.Empty:
                pass

            now = time.time()
            if not done["flag"] and now - last_hb >= HEARTBEAT:
                last_hb = now
                yield make_env(xai_service_pb2.Progress.RUNNING, 10, "Still running…")

            if done["flag"]:
                if err["exc"] is not None:
                    yield make_env(xai_service_pb2.Progress.ERROR, 100, f"Error: {err['exc']}")
                    context.set_details(str(err["exc"]))
                    context.set_code(grpc.StatusCode.INTERNAL)
                else:
                    yield make_env(xai_service_pb2.Progress.DONE, 100, "Done", feat_imp=final_result["resp"])
                break


def serve():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),   # 50 MB
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),# 50 MB

            # keepalives for long streams (avoid LB/proxy idle timeouts)
            ('grpc.keepalive_time_ms', 60_000),
            ('grpc.keepalive_timeout_ms', 20_000),
            ('grpc.keepalive_permit_without_calls', 1),
            ('grpc.http2.max_pings_without_data', 0),
            ('grpc.http2.min_time_between_pings_ms', 10_000),
            ('grpc.http2.min_ping_interval_without_data_ms', 10_000),
        ]
    )
    xai_service_pb2_grpc.add_ExplanationsServicer_to_server(ExplainabilityExecutor(), server)
    port = os.getenv('XAI_SERVER_PORT', '50051')
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    logging.info(f"Server started on port {port}")
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
