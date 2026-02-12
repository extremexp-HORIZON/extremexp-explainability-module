import numpy as np
if not hasattr(np, 'int'):
    np.int = int
import grpc
from concurrent import futures
import time
import json
import os

import torch
from torch.utils.data import default_collate
import torch.nn.functional as F
import xai_service_pb2_grpc
import xai_service_pb2
from xai_service_pb2_grpc import ExplanationsServicer
from modules.lib import *
from modules.experiment_highlights import (
    convert_runs_data_to_csv,
    build_default_pipeline,
    build_default_insights_pipeline,
)
from ExplainabilityMethodsRepository.ExplanationsHandler import *
from ExplainabilityMethodsRepository.config import shared_resources
from ExplainabilityMethodsRepository.src.glance.iterative_merges.iterative_merges import apply_action_pandas
from ExplainabilityMethodsRepository.segmentation import df_to_instances, replacement_feature_importance_batched
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from modules.lib import _load_model,_load_dataset

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

def safe_downsample(inputs, targets, masks, factor=4):
    """
    Downsample inputs (N, T, C, H, W), targets (N, 1, H, W), masks (N, 1, H, W)
    by a given spatial factor using appropriate interpolation.
    """
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

class ExplainabilityExecutor(ExplanationsServicer):

    def GetExplanation(self, request, context):
        logger.info(f"[GetExplanation] Request received - Type: {request.explanation_type}, Method: {request.explanation_method}")
        start_time = time.time()


        #for request in request_iterator:
        explanation_type = request.explanation_type
        explanation_method = request.explanation_method

        dispatch_table = {
            (explanation_type, 'pdp'): PDPHandler(),
            (explanation_type, '2dpdp'): TwoDPDPHandler(),
            (explanation_type, 'ale'): ALEHandler(),
            (explanation_type, 'counterfactuals'): CounterfactualsHandler(),
            ('featureExplanation', 'prototypes'): PrototypesHandler(),
            ('featureExplanation', 'global_counterfactuals'): GLANCEHandler(),
            ('featureExplanation', 'segmentation'): SegmentationAttributionHandler(),
            ('featureExplanation', 'shap'): SHAPHandler(),
            ('experimentExplanation', 'feature_importance'): FeatureImportanceHandler(),
            # Add more handlers as needed
        }
        
        handler = dispatch_table.get((explanation_type, explanation_method))

        if handler:
            result = handler.handle(request, explanation_type)
            elapsed_time = time.time() - start_time
            logger.info(f"[GetExplanation] Request completed successfully - Type: {explanation_type}, Method: {explanation_method}, Duration: {elapsed_time:.2f}s")
            return result
        else:
            raise ValueError(f"Unsupported explanation method '{explanation_method}' for type '{explanation_type}'")
        
    def ApplyAffectedActions(self,request,context):
        logger.info("[ApplyAffectedActions] Request received - Starting action application")
        start_time = time.time()
        try:
            # Handle the empty request
            print("Received ApplyAffectedActionsRequest (empty). Proceeding with action application.")

            # The logic here can be independent of the request parameters since there are no parameters.
            affected = shared_resources.get("affected")
            clusters_res = shared_resources.get("clusters_res")
            affected_clusters = shared_resources.get("affected_clusters")
            # index = affected_clusters['index']
            # affected_clusters = affected_clusters.drop(columns='index')

            # Sort actions by cost and apply them in sequence
            sorted_actions_dict = dict(sorted(clusters_res.items(), key=lambda item: item[1]['cost']))
            actions = [stats["action"] for i, stats in sorted_actions_dict.items()]

            # Define numeric and categorical features
            num_features = affected._get_numeric_data().columns.to_list()
            cate_features = affected.columns.difference(num_features)

            applied_affected = pd.DataFrame()
            print("Applying Actions")
            # Apply actions based on 'Chosen_Action'
            for i, val in enumerate(list(affected_clusters.Chosen_Action.unique())):
                aff = affected_clusters[affected_clusters['Chosen_Action'] == val]
                if val != '-':
                    applied_df = apply_action_pandas(
                        aff[affected.columns.to_list()],
                        actions[int(val-1)],
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
            print("Actions Applied")
            applied_affected = applied_affected.sort_index()
            # applied_affected['index'] = index

            # Store the result in shared resources (or return directly as needed)
            shared_resources['applied_affected'] = applied_affected

            # Prepare the response
            applied_affected_response = {}
            for i,col in enumerate(applied_affected.columns):
                applied_affected_response[col] = xai_service_pb2.TableContents(index=i+1,
                    values=applied_affected[col].astype(str).tolist(),
                )
                
            elapsed_time = time.time() - start_time
            logger.info(f"[ApplyAffectedActions] Request completed successfully - Duration: {elapsed_time:.2f}s")
            return xai_service_pb2.ApplyAffectedActionsResponse(
                applied_affected_actions=applied_affected_response
            )
        
        except Exception as e:
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return xai_service_pb2.ApplyAffectedActionsResponse()


    def GetFeatureImportance(self, request, context):
        logger.info(f"[GetFeatureImportance] Request received - Type: {request.type}")
        start_time = time.time()
        from ExplainabilityMethodsRepository.ExplanationsHandler import BaseExplanationHandler
        
        handler = BaseExplanationHandler()
        model_path = request.model
        type = request.type
        model, name = _load_model(model_path[0])


        train_data = _load_dataset(request.data.X_train)
        train_labels = _load_dataset(request.data.Y_train)          
        test_data = _load_dataset(request.data.X_test)
        test_labels = _load_dataset(request.data.Y_test) 

        if type == 'FeatureImportance':

            if name == 'sklearn':
                logger.info("[GetFeatureImportance] Computing sklearn permutation importance")
                result = permutation_importance(model, test_data, test_labels,scoring='accuracy', n_repeats=5, random_state=42)
                feature_importances = list(zip(test_data.columns, result.importances_mean))
                sorted_features = sorted(feature_importances, key=lambda x: x[1], reverse=True)
            elif name == 'tensorflow':
                logger.info("[GetFeatureImportance] Computing tensorflow permutation importance")
                result = permutation_importance(model, test_data, test_labels,scoring='accuracy', n_repeats=5, random_state=42)
                feature_importances = list(zip(test_data.columns, result.importances_mean))
                sorted_features = sorted(feature_importances, key=lambda x: x[1], reverse=True)
            elif name == 'pytorch':
                logger.info("[GetFeatureImportance] Computing pytorch feature importance with segmentation")
                df = pd.concat([train_data, train_labels], axis="columns")
                df = df[df["instance_id"] == df["instance_id"].iloc[0]]

                # returns lists
                instances, lons, lats, labels, _ = df_to_instances(df, C=4, patch_size=(512, 512))
                feature_names = ["dem", "mask", "wd_in", "rain"]

                # single instance, convert to tensors
                inputs = torch.tensor(instances[0], dtype=torch.float32).unsqueeze(0)  # (1,6,4,512,512)
                labels = torch.tensor(labels[0], dtype=torch.float32).unsqueeze(0)     # (1,1,512,512)
                masks = inputs[:, [0], 1, :, :]                                        # channel-1 mask

                # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                device = "cpu"
                flood_threshold = 0.01

                # Downsample before explainability (fast & safe)
                inputs, labels, masks = safe_downsample(inputs, labels, masks, factor=4)

                logger.info("Starting replacement feature importance computation...")
                start_time = time.time()
                rmse_differences, csi_differences = replacement_feature_importance_batched(
                    model=model,
                    inputs=inputs.to(device),
                    targets=labels.to(device),
                    masks=masks.to(device),
                    flooded_min=flood_threshold,
                    flooded_max=5.0,
                    n_trials=2,        # fewer trials = faster
                    batch_size=1,
                    device=device,
                )
                end_time = time.time()
                logger.info(f"Replacement feature importance completed in {end_time - start_time:.5f} seconds.")

                logger.info(f"RMSE differences shape: {rmse_differences.shape}")
                logger.info(f"CSI differences shape: {csi_differences.shape}")

                mean_rmse = rmse_differences.mean(axis=1)

                feature_importances = list(zip(feature_names, mean_rmse))
                logger.info(f"Feature importances (mean RMSE): {feature_importances}")

                sorted_features = sorted(feature_importances, key=lambda x: x[1], reverse=True)
            elapsed_time = time.time() - start_time
            logger.info(f"[GetFeatureImportance] Request completed successfully - Type: {type}, Method: FeatureImportance, Duration: {elapsed_time:.2f}s")
            return xai_service_pb2.FeatureImportanceResponse(feature_importances=[xai_service_pb2.FeatureImportance(feature_name=feature,importance_score=importance) for feature, importance in sorted_features])
        elif type == 'SHAP':
            logger.info("[GetFeatureImportance] Computing SHAP importance")

            import shap
            explainer, X_for_expl, feat_names = make_explainer_any(model, train_data, test_data)
            ex = explainer(X_for_expl)  # shap.Explanation
            try:
                ex.feature_names = feat_names
            except Exception:
                pass
            vals = np.asarray(ex.values)
            feature_names = list(ex.feature_names)
            if vals.ndim == 3:
                vals = vals[:, :, 1]

            def mean_abs_shap(v):
                # v: (n_samples, n_features)
                return np.abs(v).mean(axis=0)
            
            shap_importance = {"global_importance": {}}

            if vals.ndim == 2:
                # regression or binary (single output)
                imp = mean_abs_shap(vals)  # (n_features,)
                order = np.argsort(imp)[::-1]

                importance = [
                    {
                        "feature": feature_names[i],
                        "mean_abs_shap": float(imp[i]),
                    }
                    for i in order
                ]

                shap_importance["global_importance"] = importance     
            sorted_features = [(r["feature"], float(r["mean_abs_shap"])) for r in shap_importance["global_importance"]]
            elapsed_time = time.time() - start_time
            logger.info(f"[GetFeatureImportance] Request completed successfully - Type: {type}, Method: SHAP, Duration: {elapsed_time:.2f}s")
            return xai_service_pb2.FeatureImportanceResponse(feature_importances=[xai_service_pb2.FeatureImportance(feature_name=feature,importance_score=importance) for feature, importance in sorted_features])


    def RunExperimentHighlights(self, request, context):
        """Run the clustering + insights pipelines on experiment runs data.

        The client sends the raw runs JSON (same structure as the original
        runs.json file). Here we:
        - parse the JSON
        - convert it to the workflows DataFrame
        - run the default clustering pipeline
        - standardize metrics and run the insights pipeline

        For now we only return a simple success flag and message.
        """

        logger.info("[RunExperimentHighlights] Request received")
        start_time = time.time()

        try:
            # 1) Parse JSON payload from the request
            runs_json = request.runs_json
            try:
                runs = json.loads(runs_json)
            except json.JSONDecodeError as e:
                msg = f"Invalid runs_json payload: {e}"
                logger.error(msg)
                context.set_details(msg)
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                return xai_service_pb2.ExperimentRunsResponse(success=False, message=msg)

            # 2) Convert to DataFrame and collect parameter/metric names
            df_converted, param_names, metric_names = convert_runs_data_to_csv(runs)

            # 3) Build and run the clustering pipeline
            pipeline = build_default_pipeline()

            pipeline_params = {
                'df_converted': df_converted,
                'param_names': param_names,
                'metric_names': metric_names,
                # Default thresholds copied from your standalone script
                'low_cv_threshold': 0.05,
                'high_cv_threshold': 1.5,
                'pca_variance_threshold': 0.8,
                'mca_inertia_threshold': 0.8,
                'corr_threshold': 0.75,
                'eta_threshold': 0.33,
                'min_k': 2,
                'max_k': 20,
                'n_std': 1.5,
            }

            pipeline.run(**pipeline_params)

            # 4) Collect clustering results
            df_clustered = pipeline.get_result('step_save_results', key='df_clustered')
            medoids = pipeline.get_result('step_save_results', key='medoid_df')
            cluster_metadata = pipeline.get_result('step_create_cluster_metadata', key='cluster_metadata_df')
            X_processed_df = pipeline.get_result('step_save_results', key='processed_df')
            metric_cols = pipeline.get_result('step_save_results', key='metric_cols')
            param_cols = pipeline.get_result('step_save_results', key='hyperparam_cols')

            if df_clustered is None or medoids is None or X_processed_df is None:
                msg = "Clustering pipeline did not produce expected results (df_clustered/medoids/processed_df)."
                logger.error(msg)
                context.set_details(msg)
                context.set_code(grpc.StatusCode.INTERNAL)
                return xai_service_pb2.ExperimentRunsResponse(success=False, message=msg)

            # 5) Ensure metric columns exist in clustered data
            available_cols = set(df_clustered.columns)
            metric_cols = [col for col in metric_cols if col in available_cols]

            if not metric_cols:
                msg = "No metric columns found in clustered data; cannot proceed with insights pipeline."
                logger.error(msg)
                context.set_details(msg)
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                return xai_service_pb2.ExperimentRunsResponse(success=False, message=msg)

            # 6) Standardize original metrics
            X_original = df_clustered[metric_cols].copy()
            scaler = StandardScaler()
            X_standardized = scaler.fit_transform(X_original)

            cluster_labels = df_clustered['cluster'].values
            n_clusters = int(df_clustered['cluster'].max()) + 1

            small_clusters = set()
            if cluster_metadata is not None:
                small_clusters = set(
                    cluster_metadata[cluster_metadata['is_small'] == True]['cluster_id'].tolist()
                )

            # 7) Build and run the insights pipeline
            pipeline_insights = build_default_insights_pipeline()

            pipeline_insights_params = {
                'df_clustered': df_clustered,
                'medoids': medoids,
                'X_standardized': X_standardized,
                'X_processed_df': X_processed_df,
                'metric_cols': metric_cols,
                'param_cols': param_cols,
                'cluster_labels': cluster_labels,
                'n_clusters': n_clusters,
                'small_clusters': small_clusters,
                'correlation_threshold': 0.75,
                'n_iterations': None,
            }

            pipeline_insights.run(**pipeline_insights_params)
            print("Insights pipeline completed successfully.")

            # Extract the comprehensive cluster insights from the pipeline results
            cluster_insights = pipeline_insights.results.get('step_phase1_comprehensive_cluster_insights')
            logger.info(f"Available results: {cluster_insights}")

            # Compute total elapsed time for the whole operation
            elapsed_time = time.time() - start_time
            msg = f"Experiment highlights completed successfully in {elapsed_time:.2f}s."
            logger.info(f"[RunExperimentHighlights] {msg}")

            # Serialize insights to JSON string for the proto field
            try:
                cluster_insights_json = json.dumps(cluster_insights) if cluster_insights is not None else "null"
            except TypeError:
                # Fallback in case there are non-serializable objects inside
                cluster_insights_json = json.dumps(str(cluster_insights))

            return xai_service_pb2.ExperimentRunsResponse(
                success=True,
                message=msg,
                elapsed_time=elapsed_time,
                cluster_insights_json=cluster_insights_json,
            )

        except Exception as e:
            msg = f"Unexpected error in RunExperimentHighlights: {e}"
            logger.exception(msg)
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return xai_service_pb2.ExperimentRunsResponse(success=False, message=str(e))


def serve():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),   # 50 MB
            ('grpc.max_receive_message_length', 50 * 1024 * 1024), # 50 MB
            ('grpc.keepalive_time_ms', 60000),
            ('grpc.keepalive_timeout_ms', 5000), 
            ('grpc.keepalive_permit_without_calls', True),
            ('grpc.http2.max_pings_without_data', 0),
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
