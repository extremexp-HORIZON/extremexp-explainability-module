import ast
import logging
import numpy as np
import pandas as pd
import torch
import dice_ml

from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.preprocessing import OneHotEncoder
from raiutils.exceptions import UserConfigValidationException
from aix360.algorithms.protodash import ProtodashExplainer
from captum.attr import IntegratedGradients

import xai_service_pb2

from ExplainabilityMethodsRepository.ALE_generic import ale
from ExplainabilityMethodsRepository.config import shared_resources
from ExplainabilityMethodsRepository.pdp import partial_dependence_1D, partial_dependence_2D
from ExplainabilityMethodsRepository.src.glance.iterative_merges.iterative_merges import C_GLANCE, cumulative
from ExplainabilityMethodsRepository.segmentation import (
    treat_input,
    model_wrapper,
    df_to_instances,
    attributions_to_filtered_long_df,
)
from modules.lib import *
from modules.lib import (_load_dataset, _load_model, _load_multidimensional_array)

logger = logging.getLogger(__name__)


class BaseExplanationHandler:
    """Base class for all explanation handlers."""

    def handle(self, request, explanation_type, progress_cb=lambda *a, **k: None):
        raise NotImplementedError("Subclasses should implement this method")

    def _load_or_train_surrogate_model(self, hyperparameters, metrics):
        logger.info("Surrogate model does not exist. Training a new one.")
        surrogate_model = proxy_model(hyperparameters, metrics, 'XGBoostRegressor')
        return surrogate_model

    def _load_or_train_cf_surrogate_model(self, hyper_configs, query):
        surrogate_model, proxy_dataset = instance_proxy(hyper_configs, query)
        return surrogate_model, proxy_dataset


class GLANCEHandler(BaseExplanationHandler):
    def handle(self, request, explanation_type, progress_cb=lambda *a, **k: None):
        if explanation_type != 'featureExplanation':
            raise ValueError(f"Unknown explanation type: {explanation_type}")

        progress_cb(10, "Loading datasets & model")
        gcf_size = request.gcf_size
        cf_generator = request.cf_generator
        cluster_action_choice_algo = request.cluster_action_choice_algo

        model_path = request.model
        train_data = _load_dataset(request.data.X_train)
        train_labels = _load_dataset(request.data.Y_train)
        train_data['label'] = train_labels

        test_data = _load_dataset(request.data.X_test)
        test_labels = _load_dataset(request.data.Y_test)
        y_pred = _load_dataset(request.data.Y_pred)
        test_data['label'] = test_labels

        dataset = pd.concat([train_data, test_data], ignore_index=True)
        model, name = _load_model(model_path[0])

        test_data['target'] = y_pred['predictions']
        affected = test_data[test_data.target == 0]
        affected = affected.drop(columns=['label'])
        shared_resources["affected"] = affected.drop(columns=['target'])

        progress_cb(30, "Fitting global counterfactual model")
        global_method = C_GLANCE(
            model=model,
            initial_clusters=50,
            final_clusters=gcf_size,
            num_local_counterfactuals=10,
        )

        global_method.fit(
            dataset.drop(columns=['label']),
            dataset['label'],
            test_data.drop(columns=['label']),
            test_data.drop(columns=['label', 'target']).columns.tolist(),
            cf_generator=cf_generator,
            cluster_action_choice_algo=cluster_action_choice_algo
        )
        try:
            progress_cb(60, "Computing group counterfactuals")
            clusters, clusters_res, eff, cost = global_method.explain_group(affected.drop(columns=['target']))

            sorted_actions_dict = dict(sorted(clusters_res.items(), key=lambda item: item[1]['cost']))
            actions = [stats["action"] for i, stats in sorted_actions_dict.items()]
            i = 1
            all_clusters = {}
            num_features = test_data._get_numeric_data().columns.to_list()
            cate_features = test_data.columns.difference(num_features)

            for key in clusters:
                clusters[key]['Cluster'] = i
                all_clusters[i] = clusters[key]
                i = i + 1

            shared_resources["clusters_res"] = clusters_res
            progress_cb(75, "Post-processing actions")
            combined_df = pd.concat(all_clusters.values(), ignore_index=True)
            cluster = combined_df['Cluster']
            combined_df = combined_df.drop(columns='Cluster')
            new_aff = affected.drop(columns='target')[:20].copy(deep=True)
            new_aff['unique_id'] = new_aff.groupby(list(new_aff.columns.difference(['index']))).cumcount()
            combined_df['unique_id'] = combined_df.groupby(list(combined_df.columns)).cumcount()
            result = combined_df.merge(new_aff, on=list(combined_df.columns) + ['unique_id'], how='left')
            result = result.drop(columns='unique_id')
            eff, cost, pred_list, chosen_actions, costs = cumulative(
                model,
                result,
                actions,
                global_method.dist_func_dataframe,
                global_method.numerical_features_names,
                global_method.categorical_features_names,
                "-",
            )

            eff_cost_actions = {}
            for i, arr in pred_list.items():
                column_name = f"Action{i}_Prediction"
                result[column_name] = arr
                eff_act = pred_list[i].sum() / max(len(affected), 1)
                denom = max(pred_list[i].sum(), 1)
                cost_act = costs[i - 1][costs[i - 1] != np.inf].sum() / denom
                eff_cost_actions[i] = {'eff': eff_act, 'cost': cost_act}

            result['Cluster'] = cluster
            result['Chosen_Action'] = chosen_actions
            result['Chosen_Action'] = result['Chosen_Action'] + 1
            result = result.replace(np.inf, '-')
            shared_resources["affected_clusters"] = result
            filtered_data = {
                    k: {
                        **{
                            'action': {ak: av for ak, av in v['action'].items() if av != 0 and av != '-'}
                        },
                        **{kk: vv for kk, vv in v.items() if kk != 'action'}
                    }
                    for k, v in sorted_actions_dict.items()
                }
            actions_returned  = [stats["action"] for i,stats in filtered_data.items()]
            actions_ret = pd.DataFrame(actions_returned).fillna('-')
            # Optional preview partial:

            progress_cb(95, "Finalizing")
            return xai_service_pb2.ExplanationsResponse(
                explainability_type=explanation_type,
                explanation_method='global_counterfactuals',
                explainability_model=model_path[0],
                plot_name='Global Counterfactual Explanations',
                plot_descr="Counterfactual Explanations identify the minimal changes needed to alter predictions for a group.",
                plot_type='Table',
                feature_list=dataset.drop(columns=['label']).columns.tolist(),
                hyperparameter_list=[],
                affected_clusters={col: xai_service_pb2.TableContents(index=i + 1, values=result[col].astype(str).tolist())
                                   for i, col in enumerate(result.columns)},
                eff_cost_actions={
                    str(key): xai_service_pb2.EffCost(eff=value['eff'], cost=value['cost'])
                    for key, value in eff_cost_actions.items()
                },
                TotalEffectiveness=float(round(eff / max(len(affected), 1), 2)),
                TotalCost=float(round(cost / max(eff, 1), 2)),
                actions = {col: xai_service_pb2.TableContents(index=i+1,values=actions_ret[col].astype(str).tolist()) for i,col in enumerate(actions_ret.columns)},

            )
        except UserConfigValidationException as e:
            if str(e) == "No counterfactuals found for any of the query points! Kindly check your configuration.":
                return xai_service_pb2.ExplanationsResponse(
                    explainability_type=explanation_type,
                    explanation_method='global_counterfactuals',
                    explainability_model=model_path[0],
                    plot_name='Error',
                    plot_descr="No counterfactuals found with the selected Local Counterfactual Method.",
                    plot_type='Error',
                    feature_list=[],
                    hyperparameter_list=[],
                    affected_clusters={},
                    eff_cost_actions={},
                    TotalEffectiveness=0.0,
                    TotalCost=0.0,
                    actions={},
                )
            raise


class PDPHandler(BaseExplanationHandler):
    def handle(self, request, explanation_type, progress_cb=lambda *a, **k: None):
        if explanation_type == 'featureExplanation':
            progress_cb(10, "Loading data/model")
            model_path = request.model
            train_data = _load_dataset(request.data.X_train)
            train_labels = _load_dataset(request.data.Y_train)
            model, name = _load_model(model_path[0])

            feature = request.feature1 or train_data.columns.tolist()[0]
            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
            numeric_features = train_data.select_dtypes(include=numerics).columns.tolist()
            categorical_features = train_data.columns.drop(numeric_features)

            progress_cb(60, "Computing PDP")
            pdp = partial_dependence(
                model, train_data,
                features=[train_data.columns.tolist().index(feature)],
                feature_names=train_data.columns.tolist(),
                categorical_features=categorical_features
            )

            axis_type = 'categorical' if isinstance(pdp['grid_values'][0][0], str) else 'numerical'
            pdp_grid = [value.tolist() for value in pdp['grid_values']][0]
            pdp_vals = [value.tolist() for value in pdp['average']][0]

            progress_cb(90, "Formatting")
            return xai_service_pb2.ExplanationsResponse(
                explainability_type=explanation_type,
                explanation_method='pdp',
                explainability_model=model_path[0],
                plot_name='Partial Dependence Plot (PDP)',
                plot_descr="PD Plots show how a feature affects predictions, holding others constant.",
                plot_type='LinePlot',
                features=xai_service_pb2.Features(feature1=feature, feature2=''),
                feature_list=train_data.columns.tolist(),
                hyperparameter_list=[],
                xAxis=xai_service_pb2.Axis(axis_name=f'{feature}', axis_values=[str(v) for v in pdp_grid], axis_type=axis_type),
                yAxis=xai_service_pb2.Axis(axis_name='PDP Values', axis_values=[str(v) for v in pdp_vals], axis_type='numerical'),
                zAxis=xai_service_pb2.Axis(axis_name='', axis_values='', axis_type='')
            )

        elif explanation_type == 'hyperparameterExplanation':
            hyper_configs = request.hyper_configs
            progress_cb(10, "Training surrogate model")
            hyper_space = create_hyperspace(hyper_configs)
            hyper_df, sorted_metrics = create_hyper_df(hyper_configs)
            surrogate_model = self._load_or_train_surrogate_model(hyper_df, sorted_metrics)

            param_grid = transform_grid(hyper_space)
            param_space, name = dimensions_aslists(param_grid)
            space = Space(param_space)
            feats = {n: i for i, n in enumerate(name)}
            plot_dims = [(row, space.dimensions[row]) for row in range(space.n_dims)]

            pdp_samples = space.rvs(n_samples=1000, random_state=123456)
            feature = request.feature1 or name[0]

            progress_cb(60, "Computing PDP (surrogate)")
            index, dim = plot_dims[feats[feature]]
            xi1, yi1 = partial_dependence_1D(space, surrogate_model, index, samples=pdp_samples, name=name, n_points=100)
            x = xi1.tolist()
            y = yi1
            axis_type = 'categorical' if (len(x) and isinstance(x[0], str)) else 'numerical'

            progress_cb(90, "Formatting")
            return xai_service_pb2.ExplanationsResponse(
                explainability_type=explanation_type,
                explanation_method='pdp',
                explainability_model="",
                plot_name='Partial Dependence Plot (PDP)',
                plot_descr="Effect of a hyperparameter on the metric (surrogate).",
                plot_type='LinePlot',
                features=xai_service_pb2.Features(feature1=feature, feature2=''),
                feature_list=[],
                hyperparameter_list=name,
                xAxis=xai_service_pb2.Axis(axis_name=f'{feature}', axis_values=[str(v) for v in x], axis_type=axis_type),
                yAxis=xai_service_pb2.Axis(axis_name='PDP Values', axis_values=[str(v) for v in y], axis_type='numerical'),
                zAxis=xai_service_pb2.Axis(axis_name='', axis_values='', axis_type=''),
            )

        elif explanation_type == 'experimentExplanation':
            experiment_configs = request.experiment_configs
            progress_cb(10, "Training surrogate model (experiment)")
            keep_common_variability_points(experiment_configs)
            hyper_space = create_hyperspace(experiment_configs)
            hyper_df, sorted_metrics = create_hyper_df(experiment_configs)
            surrogate_model = self._load_or_train_surrogate_model(hyper_df, sorted_metrics)
            param_grid = transform_grid(hyper_space)
            param_space, name = dimensions_aslists(param_grid)
            space = Space(param_space)
            feats = {n: i for i, n in enumerate(name)}
            plot_dims = [(row, space.dimensions[row]) for row in range(space.n_dims)]
            pdp_samples = space.rvs(n_samples=1000, random_state=123456)

            feature = request.feature1 or name[0]
            progress_cb(60, "Computing PDP (experiment)")
            index, dim = plot_dims[feats[feature]]
            xi1, yi1 = partial_dependence_1D(space, surrogate_model, index, samples=pdp_samples, name=name, n_points=100)
            x = xi1.tolist()
            y = yi1
            axis_type = 'categorical' if (len(x) and isinstance(x[0], str)) else 'numerical'

            progress_cb(90, "Formatting")
            return xai_service_pb2.ExplanationsResponse(
                explainability_type=explanation_type,
                explanation_method='pdp',
                explainability_model="",
                plot_name='Partial Dependence Plot (PDP)',
                plot_descr="Effect of variability point on the metric (experiment surrogate).",
                plot_type='LinePlot',
                features=xai_service_pb2.Features(feature1=feature, feature2=''),
                feature_list=[],
                hyperparameter_list=name,
                xAxis=xai_service_pb2.Axis(axis_name=f'{feature}', axis_values=[str(v) for v in x], axis_type=axis_type),
                yAxis=xai_service_pb2.Axis(axis_name='PDP Values', axis_values=[str(v) for v in y], axis_type='numerical'),
                zAxis=xai_service_pb2.Axis(axis_name='', axis_values='', axis_type=''),
            )
        else:
            raise ValueError(f"Unknown explanation type: {explanation_type}")


class TwoDPDPHandler(BaseExplanationHandler):
    def handle(self, request, explanation_type, progress_cb=lambda *a, **k: None):
        if explanation_type == 'featureExplanation':
            progress_cb(10, "Loading data/model")
            model_path = request.model
            train_data = _load_dataset(request.data.X_train)
            train_labels = _load_dataset(request.data.Y_train)
            model, name = _load_model(model_path[0])

            feature1 = request.feature1 or train_data.columns.tolist()[0]
            feature2 = request.feature2 or train_data.columns.tolist()[1]

            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
            numeric_features = train_data.select_dtypes(include=numerics).columns.tolist()
            categorical_features = train_data.columns.drop(numeric_features)

            progress_cb(60, "Computing 2D PDP")
            pdp = partial_dependence(
                model, train_data,
                features=[(train_data.columns.tolist().index(feature1), train_data.columns.tolist().index(feature2))],
                feature_names=train_data.columns.tolist(),
                categorical_features=categorical_features
            )

            axis_type_0 = 'categorical' if isinstance(pdp['grid_values'][0][0], str) else 'numerical'
            axis_type_1 = 'categorical' if isinstance(pdp['grid_values'][1][0], str) else 'numerical'
            pdp_grid_1 = [value.tolist() for value in pdp['grid_values']][0]
            pdp_grid_2 = [value.tolist() for value in pdp['grid_values']][1]
            pdp_vals = [value.tolist() for value in pdp['average']][0]

            progress_cb(90, "Formatting 2D PDP")
            return xai_service_pb2.ExplanationsResponse(
                explainability_type=explanation_type,
                explanation_method='2dpdp',
                explainability_model=model_path[0],
                plot_name='2D-Partial Dependence Plot (2D-PDP)',
                plot_descr="Marginal effect of two features on predictions.",
                plot_type='ContourPlot',
                features=xai_service_pb2.Features(feature1=feature1, feature2=feature2),
                feature_list=train_data.columns.tolist(),
                hyperparameter_list=[],
                xAxis=xai_service_pb2.Axis(axis_name=f'{feature1}', axis_values=[str(v) for v in pdp_grid_1], axis_type=axis_type_0),
                yAxis=xai_service_pb2.Axis(axis_name=f'{feature2}', axis_values=[str(v) for v in pdp_grid_2], axis_type=axis_type_1),
                zAxis=xai_service_pb2.Axis(axis_name='', axis_values=[str(v) for v in pdp_vals], axis_type='numerical'),
            )

        elif explanation_type == 'hyperparameterExplanation':
            hyper_configs = request.hyper_configs
            progress_cb(10, "Training surrogate model")
            hyper_space = create_hyperspace(hyper_configs)
            hyper_df, sorted_metrics = create_hyper_df(hyper_configs)
            surrogate_model = self._load_or_train_surrogate_model(hyper_df, sorted_metrics)
            param_grid = transform_grid(hyper_space)
            param_space, name = dimensions_aslists(param_grid)
            space = Space(param_space)

            feature1 = request.feature1 or name[0]
            feature2 = request.feature2 or name[1]
            index1 = name.index(feature1)
            index2 = name.index(feature2)

            pdp_samples = space.rvs(n_samples=1000, random_state=123456)
            progress_cb(60, "Computing 2D PDP (surrogate)")
            xi, yi, zi = partial_dependence_2D(space, surrogate_model, index1, index2, pdp_samples, name, 100)

            x = [arr.tolist() for arr in xi]
            y = [arr.tolist() for arr in yi]
            z = [arr.tolist() for arr in zi]

            progress_cb(90, "Formatting")
            return xai_service_pb2.ExplanationsResponse(
                explainability_type=explanation_type,
                explanation_method='2dpdp',
                explainability_model='',
                plot_name='2D-Partial Dependence Plot (2D-PDP)',
                plot_descr="Metric changes when two hyperparameters vary (surrogate).",
                plot_type='ContourPlot',
                features=xai_service_pb2.Features(feature1=feature1, feature2=feature2),
                feature_list=[],
                hyperparameter_list=name,
                xAxis=xai_service_pb2.Axis(axis_name=f'{feature2}', axis_values=[str(v) for v in x],
                                           axis_type='categorical' if (len(x) and isinstance(x[0], str)) else 'numerical'),
                yAxis=xai_service_pb2.Axis(axis_name=f'{feature1}', axis_values=[str(v) for v in y],
                                           axis_type='categorical' if (len(y) and isinstance(y[0], str)) else 'numerical'),
                zAxis=xai_service_pb2.Axis(axis_name='', axis_values=[str(v) for v in z], axis_type='numerical'),
            )

        elif explanation_type == 'experimentExplanation':
            experiment_configs = request.experiment_configs
            progress_cb(10, "Training surrogate model (experiment)")
            keep_common_variability_points(experiment_configs)
            hyper_space = create_hyperspace(experiment_configs)
            hyper_df, sorted_metrics = create_hyper_df(experiment_configs)
            surrogate_model = self._load_or_train_surrogate_model(hyper_df, sorted_metrics)
            param_grid = transform_grid(hyper_space)
            param_space, name = dimensions_aslists(param_grid)
            space = Space(param_space)

            feature1 = request.feature1 or name[0]
            feature2 = request.feature2 or name[1]
            index1 = name.index(feature1)
            index2 = name.index(feature2)

            pdp_samples = space.rvs(n_samples=1000, random_state=123456)
            progress_cb(60, "Computing 2D PDP (experiment)")
            xi, yi, zi = partial_dependence_2D(space, surrogate_model, index1, index2, pdp_samples, name, 100)

            x = [arr.tolist() for arr in xi]
            y = [arr.tolist() for arr in yi]
            z = [arr.tolist() for arr in zi]

            progress_cb(90, "Formatting")
            return xai_service_pb2.ExplanationsResponse(
                explainability_type=explanation_type,
                explanation_method='2dpdp',
                explainability_model='',
                plot_name='2D-Partial Dependence Plot (2D-PDP)',
                plot_descr="Metric changes when two variability points vary (experiment surrogate).",
                plot_type='ContourPlot',
                features=xai_service_pb2.Features(feature1=feature1, feature2=feature2),
                feature_list=[],
                hyperparameter_list=name,
                xAxis=xai_service_pb2.Axis(axis_name=f'{feature2}', axis_values=[str(v) for v in x],
                                           axis_type='categorical' if (len(x) and isinstance(x[0], str)) else 'numerical'),
                yAxis=xai_service_pb2.Axis(axis_name=f'{feature1}', axis_values=[str(v) for v in y],
                                           axis_type='categorical' if (len(y) and isinstance(y[0], str)) else 'numerical'),
                zAxis=xai_service_pb2.Axis(axis_name='', axis_values=[str(v) for v in z], axis_type='numerical'),
            )
        else:
            raise ValueError(f"Unknown explanation type: {explanation_type}")


class ALEHandler(BaseExplanationHandler):
    def handle(self, request, explanation_type, progress_cb=lambda *a, **k: None):
        if explanation_type == 'featureExplanation':
            progress_cb(10, "Loading data/model")
            model_path = request.model
            train_data = _load_dataset(request.data.X_train)
            train_labels = _load_dataset(request.data.Y_train)
            model, name = _load_model(model_path[0])

            feature = request.feature1 or train_data.columns.tolist()[0]

            progress_cb(60, "Computing ALE")
            if str(train_data[feature].dtype) in ['int', 'float', 'int64', 'float64', 'float32']:
                ale_eff = ale(X=train_data, model=model, feature=[feature], plot=False, grid_size=50, include_CI=True, C=0.95)
            else:
                ale_eff = ale(X=train_data, model=model, feature=[feature], plot=False, grid_size=50,
                              predictors=train_data.columns.tolist(), include_CI=True, C=0.95)

            progress_cb(90, "Formatting")
            return xai_service_pb2.ExplanationsResponse(
                explainability_type=explanation_type,
                explanation_method='ale',
                explainability_model=model_path[0],
                plot_name='Accumulated Local Effects Plot (ALE)',
                plot_descr="Effect of a single feature on the predicted outcome.",
                plot_type='LinePLot',
                features=xai_service_pb2.Features(feature1=feature, feature2=''),
                feature_list=train_data.columns.tolist(),
                hyperparameter_list=[],
                xAxis=xai_service_pb2.Axis(axis_name=f'{feature}', axis_values=[str(v) for v in ale_eff.index.tolist()],
                                           axis_type='categorical' if isinstance(ale_eff.index.tolist()[0], str) else 'numerical'),
                yAxis=xai_service_pb2.Axis(axis_name='ALE Values', axis_values=[str(v) for v in ale_eff.eff.tolist()],
                                           axis_type='numerical'),
                zAxis=xai_service_pb2.Axis(axis_name='', axis_values='', axis_type='')
            )

        elif explanation_type == 'hyperparameterExplanation':
            hyper_configs = request.hyper_configs
            progress_cb(10, "Training surrogate model")
            hyper_space = create_hyperspace(hyper_configs)
            hyper_df, sorted_metrics = create_hyper_df(hyper_configs)
            surrogate_model = self._load_or_train_surrogate_model(hyper_df, sorted_metrics)
            param_grid = transform_grid(hyper_space)
            param_space, name = dimensions_aslists(param_grid)
            space = Space(param_space)

            feature1 = request.feature1 or name[0]
            pdp_samples = space.rvs(n_samples=1000, random_state=123456)
            data = pd.DataFrame(pdp_samples, columns=[n for n in name])

            progress_cb(60, "Computing ALE (surrogate)")
            if str(data[feature1].dtype) in ['int', 'float', 'int64', 'float64', 'float32']:
                ale_eff = ale(X=data, model=surrogate_model, feature=[feature1], plot=False, grid_size=50, include_CI=True, C=0.95)
            else:
                ale_eff = ale(X=data, model=surrogate_model, feature=[feature1], plot=False, grid_size=50,
                              predictors=data.columns.tolist(), include_CI=True, C=0.95)

            progress_cb(90, "Formatting")
            return xai_service_pb2.ExplanationsResponse(
                explainability_type=explanation_type,
                explanation_method='ale',
                explainability_model='',
                plot_name='Accumulated Local Effects Plot (ALE)',
                plot_descr='Effect of a hyperparameter on the accuracy (surrogate).',
                plot_type='LinePLot',
                features=xai_service_pb2.Features(feature1=feature1, feature2=''),
                feature_list=[],
                hyperparameter_list=name,
                xAxis=xai_service_pb2.Axis(axis_name=f'{feature1}', axis_values=[str(v) for v in ale_eff.index.tolist()],
                                           axis_type='categorical' if isinstance(ale_eff.index.tolist()[0], str) else 'numerical'),
                yAxis=xai_service_pb2.Axis(axis_name='ALE Values', axis_values=[str(v) for v in ale_eff.eff.tolist()],
                                           axis_type='numerical'),
                zAxis=xai_service_pb2.Axis(axis_name='', axis_values='', axis_type='')
            )

        elif explanation_type == 'experimentExplanation':
            experiment_configs = request.experiment_configs
            progress_cb(10, "Training surrogate model (experiment)")
            keep_common_variability_points(experiment_configs)
            hyper_space = create_hyperspace(experiment_configs)
            hyper_df, sorted_metrics = create_hyper_df(experiment_configs)
            surrogate_model = self._load_or_train_surrogate_model(hyper_df, sorted_metrics)
            param_grid = transform_grid(hyper_space)
            param_space, name = dimensions_aslists(param_grid)
            space = Space(param_space)

            feature1 = request.feature1 or name[0]
            pdp_samples = space.rvs(n_samples=1000, random_state=123456)
            data = pd.DataFrame(pdp_samples, columns=[n for n in name])

            progress_cb(60, "Computing ALE (experiment)")
            if str(data[feature1].dtype) in ['int', 'float', 'int64', 'float64', 'float32']:
                ale_eff = ale(X=data, model=surrogate_model, feature=[feature1], plot=False, grid_size=50, include_CI=True, C=0.95)
            else:
                ale_eff = ale(X=data, model=surrogate_model, feature=[feature1], plot=False, grid_size=50,
                              predictors=data.columns.tolist(), include_CI=True, C=0.95)

            progress_cb(90, "Formatting")
            return xai_service_pb2.ExplanationsResponse(
                explainability_type=explanation_type,
                explanation_method='ale',
                explainability_model='',
                plot_name='Accumulated Local Effects Plot (ALE)',
                plot_descr='Effect of a variability point on the metric (experiment surrogate).',
                plot_type='LinePLot',
                features=xai_service_pb2.Features(feature1=feature1, feature2=''),
                feature_list=[],
                hyperparameter_list=name,
                xAxis=xai_service_pb2.Axis(axis_name=f'{feature1}', axis_values=[str(v) for v in ale_eff.index.tolist()],
                                           axis_type='categorical' if isinstance(ale_eff.index.tolist()[0], str) else 'numerical'),
                yAxis=xai_service_pb2.Axis(axis_name='ALE Values', axis_values=[str(v) for v in ale_eff.eff.tolist()],
                                           axis_type='numerical'),
                zAxis=xai_service_pb2.Axis(axis_name='', axis_values='', axis_type='')
            )
        else:
            raise ValueError(f"Unknown explanation type: {explanation_type}")


class CounterfactualsHandler(BaseExplanationHandler):
    def handle(self, request, explanation_type, progress_cb=lambda *a, **k: None):
        if explanation_type == 'featureExplanation':
            progress_cb(10, "Preparing query & data")
            query = ast.literal_eval(request.query)
            query = pd.DataFrame([query])
            query = query.drop(columns=['label'])
            target = 'label'

            model_path = request.model
            train_data = _load_dataset(request.data.X_train)
            train_labels = _load_dataset(request.data.Y_train)
            model, name = _load_model(model_path[0])

            dataframe = pd.concat([train_data.reset_index(drop=True), train_labels.reset_index(drop=True)], axis=1)
            d = dice_ml.Data(
                dataframe=dataframe,
                continuous_features=dataframe.drop(columns=target).select_dtypes(include='number').columns.tolist(),
                outcome_name=target
            )
            m = dice_ml.Model(model=model, backend="sklearn")
            exp = dice_ml.Dice(d, m, method="random")

            progress_cb(60, "Generating counterfactuals")
            try:
                e1 = exp.generate_counterfactuals(query.drop(columns=['prediction']), total_CFs=5, desired_class="opposite", sample_size=5000)
                e1.visualize_as_dataframe(show_only_changes=True)
                cfs = e1.cf_examples_list[0].final_cfs_df
                query.rename(columns={"prediction": target}, inplace=True)

                cfs['Type'] = 'Counterfactual'
                query['Type'] = 'Factual'
                factual = query.iloc[0].drop(['Type']) if 'Type' in query.columns else query.iloc[0]

                diffs = []
                for _, row in cfs.iterrows():
                    diff_row = {}
                    for col in factual.index:
                        if col in ['Type', target]:
                            continue
                        cf_val = row[col]
                        f_val = factual[col]
                        if pd.isna(cf_val) or pd.isna(f_val):
                            diff = '-'
                        elif cf_val != f_val:
                            try:
                                delta = cf_val - f_val
                                diff = f'+{delta}' if delta > 0 else f'{delta}'
                            except Exception:
                                diff = cf_val
                        else:
                            diff = '-'
                        diff_row[col] = diff
                    diff_row['Type'] = 'Counterfactual'
                    diffs.append(diff_row)

                diffs_df = pd.DataFrame(diffs)
                factual_diff = {col: factual[col] for col in factual.index if col not in ['Type', target]}
                factual_diff['Type'] = 'Factual'
                diffs_df = pd.concat([pd.DataFrame([factual_diff]), diffs_df], ignore_index=True)
                cf_only = diffs_df[diffs_df['Type'] == 'Counterfactual']
                cols_to_drop = [col for col in cf_only.columns if col not in ['Type'] and (cf_only[col] == '-').all()]
                diffs_df.drop(columns=cols_to_drop, inplace=True)
                cfs = pd.concat([query, cfs])
                diffs_df['label'] = cfs['label'].values

                progress_cb(90, "Formatting")
                return xai_service_pb2.ExplanationsResponse(
                    explainability_type=explanation_type,
                    explanation_method='counterfactuals',
                    explainability_model=model_path[0],
                    plot_name='Counterfactual Explanations',
                    plot_descr="Minimal changes needed to alter a prediction.",
                    plot_type='Table',
                    feature_list=dataframe.columns.tolist(),
                    hyperparameter_list=[],
                    table_contents={col: xai_service_pb2.TableContents(index=i + 1, values=diffs_df[col].astype(str).tolist()) for i, col in enumerate(diffs_df.columns)}
                )
            except UserConfigValidationException as e:
                if str(e) == "No counterfactuals found for any of the query points! Kindly check your configuration.":
                    return xai_service_pb2.ExplanationsResponse(
                        explainability_type=explanation_type,
                        explanation_method='couterfactuals',
                        explainability_model=model_path[0],
                        plot_name='Error',
                        plot_descr=f"An error occurred while generating the explanation: {str(e)}",
                        plot_type='Error',
                        feature_list=dataframe.columns.tolist(),
                        hyperparameter_list=[],
                )
                raise

        elif explanation_type == 'hyperparameterExplanation':
            model_path = request.model
            hyper_configs = request.hyper_configs
            query = ast.literal_eval(request.query)
            if isinstance(query, dict):
                query = pd.DataFrame([query])
                prediction = query['prediction']
                label = query['label']
                query = query.drop(columns=['label', 'prediction'])
            else:
                query = np.array(query)
                label = pd.Series(1)
                prediction = pd.Series(2)

            progress_cb(20, "Creating proxy dataset/model")
            try:
                surrogate_model, proxy_dataset = self._load_or_train_cf_surrogate_model(hyper_configs, query)
            except (UserConfigValidationException, ValueError) as e:
                return xai_service_pb2.ExplanationsResponse(
                    explainability_type=explanation_type,
                    explanation_method='couterfactuals',
                    explainability_model=model_path[0],
                    plot_name='Error',
                    plot_descr=f"An error occurred while generating the explanation: {str(e)}",
                    plot_type='Error',
                    feature_list=[],
                    hyperparameter_list=[],
                )
            hp_query = create_cfquery_df(hyper_configs, model_path[0])

            d = dice_ml.Data(
                dataframe=proxy_dataset,
                continuous_features=proxy_dataset.drop(columns='BinaryLabel').select_dtypes(include='number').columns.tolist(),
                outcome_name='BinaryLabel'
            )
            m = dice_ml.Model(model=surrogate_model, backend="sklearn")
            exp = dice_ml.Dice(d, m, method="random")

            progress_cb(60, "Generating HP counterfactuals")
            try:
                e1 = exp.generate_counterfactuals(hp_query, total_CFs=5, desired_class=int(label.values[0]), sample_size=5000)
            except UserConfigValidationException as e:
                return xai_service_pb2.ExplanationsResponse(
                    explainability_type=explanation_type,
                    explanation_method='couterfactuals',
                    explainability_model=model_path[0],
                    plot_name='Error',
                    plot_descr=f"An error occurred while generating the explanation: {str(e)}",
                    plot_type='Error',
                    feature_list=[],
                    hyperparameter_list=hp_query.columns.tolist(),
                )

            dtypes_dict = proxy_dataset.drop(columns='BinaryLabel').dtypes.to_dict()
            cfs = e1.cf_examples_list[0].final_cfs_df
            for col, dtype in dtypes_dict.items():
                cfs[col] = cfs[col].astype(dtype)
                scaled_query, scaled_cfs = min_max_scale(proxy_dataset=proxy_dataset, factual=hp_query.copy(deep=True), counterfactuals=cfs.copy(deep=True), label='BinaryLabel')
            cfs['Cost'] = cf_difference(scaled_query, scaled_cfs)
            cfs = cfs.sort_values(by='Cost')
            cfs['Type'] = 'Counterfactual'
            hp_query['Cost'] = '-'
            hp_query['Type'] = 'Factual'

            hp_query['BinaryLabel'] = prediction
            factual = hp_query.iloc[0].drop(['Type', 'Cost', 'BinaryLabel'])
            diffs = []
            for _, row in cfs.iterrows():
                diff_row = {}
                for col in factual.index:
                    cf_val = row[col]; f_val = factual[col]
                    if pd.isna(cf_val) or pd.isna(f_val):
                        diff = '-'
                    elif cf_val != f_val:
                        try:
                            delta = cf_val - f_val
                            diff = f'+{delta}' if delta > 0 else f'{delta}'
                        except Exception:
                            diff = cf_val
                    else:
                        diff = '-'
                    diff_row[col] = diff
                diff_row['Cost'] = row['Cost']
                diff_row['Type'] = 'Counterfactual'
                diff_row['BinaryLabel'] = row['BinaryLabel']
                diffs.append(diff_row)

            diffs_df = pd.DataFrame(diffs)
            factual_diff = {col: factual[col] for col in factual.index}
            factual_diff['Cost'] = '-'
            factual_diff['Type'] = 'Factual'
            factual_diff['BinaryLabel'] = prediction.values[0]
            diffs_df = pd.concat([pd.DataFrame([factual_diff]), diffs_df], ignore_index=True)

            cf_only = diffs_df[diffs_df['Type'] == 'Counterfactual']
            cols_to_drop = [col for col in factual.index if (cf_only[col] == '-').all()]
            diffs_df.drop(columns=cols_to_drop, inplace=True)

            progress_cb(90, "Formatting")
            return xai_service_pb2.ExplanationsResponse(
                explainability_type=explanation_type,
                explanation_method='counterfactuals',
                explainability_model=model_path[0],
                plot_name='Counterfactual Explanations',
                plot_descr="Minimal changes on hyperparameters to correct a misclassification.",
                plot_type='Table',
                feature_list=[],
                hyperparameter_list=hp_query.drop(columns=['Cost', 'Type', 'BinaryLabel']).columns.tolist(),
                table_contents={col: xai_service_pb2.TableContents(index=i + 1, values=diffs_df[col].astype(str).tolist()) for i, col in enumerate(diffs_df.columns)}
            )

        elif explanation_type == 'experimentExplanation':
            raise NotImplementedError("Counterfactual explanations for experimentExplanation are not implemented yet.")


class PrototypesHandler(BaseExplanationHandler):
    def handle(self, request, explanation_type, progress_cb=lambda *a, **k: None):
        progress_cb(10, "Loading data/model")
        query = ast.literal_eval(request.query)
        query = pd.DataFrame([query])
        label = query['label']
        prediction = query['prediction']
        query.drop(columns=['label', 'prediction'], inplace=True)
        categorical_cols = query.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = query.select_dtypes(exclude=['object', 'category']).columns.tolist()

        model_path = request.model
        train_data = _load_dataset(request.data.X_train)
        train_labels = _load_dataset(request.data.Y_train)
        model, name = _load_model(model_path[0])

        progress_cb(60, "Computing prototypes")
        explainer = ProtodashExplainer()
        reference_set_train = train_data.copy(deep=True)
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder.fit(reference_set_train[categorical_cols])

        query_encoded_cat = encoder.transform(query[categorical_cols])
        ref_encoded_cat = encoder.transform(reference_set_train[categorical_cols])
        query_numeric = query[numerical_cols].to_numpy()
        ref_numeric = reference_set_train[numerical_cols].to_numpy()

        query_encoded = np.hstack((query_numeric, query_encoded_cat))
        ref_encoded = np.hstack((ref_numeric, ref_encoded_cat))

        (W, S, _) = explainer.explain(query_encoded.reshape(1, -1), ref_encoded, m=5)
        prototypes = reference_set_train.reset_index(drop=True).iloc[S, :].copy()
        prototypes['prediction'] = model.predict(prototypes)
        prototypes = prototypes.reset_index(drop=True).T
        prototypes.rename(columns={0: 'Prototype1', 1: 'Prototype2', 2: 'Prototype3', 3: 'Prototype4', 4: 'Prototype5'}, inplace=True)
        prototypes = prototypes.reset_index()

        prototypes.set_index('index', inplace=True)
        boolean_df = pd.DataFrame(index=prototypes.index)
        query['prediction'] = prediction
        for col in prototypes.columns:
            boolean_df[col] = prototypes[col] == query.loc[0][prototypes.index].values

        prototypes.reset_index(inplace=True)
        new_row = pd.DataFrame([{
            'index': 'Weights',
            'Prototype1': np.around(W / np.sum(W), 2)[0],
            'Prototype2': np.around(W / np.sum(W), 2)[1],
            'Prototype3': np.around(W / np.sum(W), 2)[2],
            'Prototype4': np.around(W / np.sum(W), 2)[3],
            'Prototype5': np.around(W / np.sum(W), 2)[4]
        }])
        prototypes = pd.concat([prototypes, new_row], ignore_index=True)
        new_bool_row = pd.DataFrame([{
            'index': 'Weights',
            'Prototype1': False, 'Prototype2': False, 'Prototype3': False, 'Prototype4': False, 'Prototype5': False
        }])
        boolean_df = pd.concat([boolean_df, new_bool_row], ignore_index=True)

        table_contents = {col: xai_service_pb2.TableContents(index=i + 1, values=prototypes[col].astype(str).tolist(), colour=boolean_df[col].astype(str).tolist())
                          for i, col in enumerate(prototypes.columns)}

        progress_cb(90, "Formatting")
        return xai_service_pb2.ExplanationsResponse(
            explainability_type=explanation_type,
            explanation_method='prototypes',
            explainability_model=model_path[0],
            plot_name='Prototypes',
            plot_descr="Prototypical examples with weights for representativeness.",
            plot_type='Table',
            feature_list=train_data.columns.tolist(),
            hyperparameter_list=[],
            table_contents=table_contents
        )


class SegmentationAttributionHandler(BaseExplanationHandler):
    """Handler that computes attributions"""
    def handle(self, request, explanation_type, progress_cb=lambda *a, **k: None):
        if explanation_type != 'featureExplanation':
            raise ValueError(f"Unsupported explanation_type {explanation_type}")

        progress_cb(10, "Loading dataset & model")
        model_path = request.model
        train_data = _load_dataset(request.data.X_train)
        train_labels = _load_dataset(request.data.Y_train)
        test_data = _load_dataset(request.data.X_test)
        test_labels = _load_dataset(request.data.Y_test)

        test_df = pd.concat([test_data, test_labels], axis="columns")
        instances, x_coords, y_coords, labels = df_to_instances(test_df)
        instance, x_coords, y_coords, label = instances[0], x_coords[0], y_coords[0], labels[0]
        mask = (instance[[0], 1, :, :] == 1).astype(np.float32)
        query, x_coords, y_coords, gt, mask = instance, x_coords, y_coords, label, mask

        model, _ = _load_model(model_path[0])
        if not isinstance(model, torch.nn.Module):
            raise ValueError(f"Model {model_path[0]} is not a supported torch model")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device).eval()

        progress_cb(50, "Preparing inputs")
        test_input = treat_input(query, device=device)
        test_mask = treat_input(mask, device=device)
        baseline = torch.zeros_like(test_input)

        ig = IntegratedGradients(lambda inp: model_wrapper(inp, model=model, mask=test_mask))

        progress_cb(70, "Computing attributions")
        attributions, delta = ig.attribute(
            test_input,
            baseline,
            target=None,
            n_steps=5,
            return_convergence_delta=True,
            internal_batch_size=1,
        )

        attributions_np = attributions.squeeze().detach().cpu().numpy()
        MAX_POINTS = 2_000
        df_attrs, _ = attributions_to_filtered_long_df(
            attributions=attributions_np,
            x_coords=np.asarray(x_coords),
            y_coords=np.asarray(y_coords),
            mask=test_mask.detach().cpu().numpy().squeeze(),
            channel_names=['DEM', 'Mask', 'WD_IN', 'RAIN'],
            max_rows=MAX_POINTS,
        )
        df_feats, _ = attributions_to_filtered_long_df(
            attributions=test_input.squeeze().detach().cpu().numpy(),
            x_coords=np.asarray(x_coords),
            y_coords=np.asarray(y_coords),
            mask=test_mask.detach().cpu().numpy().squeeze(),
            channel_names=['DEM', 'Mask', 'WD_IN', 'RAIN'],
            max_rows=MAX_POINTS,
        )

        table_contents_feats = {col: xai_service_pb2.TableContents(index=i + 1, values=df_feats[col].astype(str).tolist())
                                for i, col in enumerate(df_feats.columns)}
        table_contents_attrs = {col: xai_service_pb2.TableContents(index=i + 1, values=df_attrs[col].astype(str).tolist())
                                for i, col in enumerate(df_attrs.columns)}

        progress_cb(90, "Formatting")
        return xai_service_pb2.ExplanationsResponse(
            explainability_type=explanation_type,
            explanation_method='segmentation',
            explainability_model=model_path[0],
            plot_name='Attributions',
            plot_descr="Attributes the model's output to each input feature (pixel).",
            plot_type='Table',
            hyperparameter_list=[],
            features_table=table_contents_feats,
            attributions_table=table_contents_attrs,
            features_table_columns=df_feats.columns.tolist(),
            attributions_table_columns=df_attrs.columns.tolist(),
        )


class SHAPHandler(BaseExplanationHandler):
    def handle(self, request, explanation_type, progress_cb=lambda *a, **k: None):
        import shap
        if explanation_type != 'featureExplanation':
            raise ValueError("Unsupported explanation_type")

        progress_cb(10, "Loading data/model")
        model_path = request.model
        train_data = _load_dataset(request.data.X_train)
        train_labels = _load_dataset(request.data.Y_train)
        test_data = _load_dataset(request.data.X_test)
        test_labels = _load_dataset(request.data.Y_test)
        model, name = _load_model(model_path[0])

        idx = request.instance_index

        progress_cb(35, "Initializing SHAP")
        explainer = shap.Explainer(model)

        progress_cb(55, "Computing SHAP values")
        ex = explainer(test_data)

        progress_cb(75, "Formatting explanation")
        shap_explanations = shap_waterfall_payload(ex, idx=idx, class_idx=None, top_k=10, include_rest=False)

        final = xai_service_pb2.ExplanationsResponse(
            explainability_type="featureExplanation",
            explanation_method="shap",
            explainability_model=model_path[0],
            plot_name="SHAP",
            plot_descr="SHAP assigns a contribution to each feature.",
            plot_type="Bar Plot",
            xAxis=xai_service_pb2.Axis(
                axis_name="E[f(X)] and f(x)",
                axis_values=[str(shap_explanations['expected_value']), str(shap_explanations['prediction_value'])],
                axis_type='numerical'
            ),
            shap_contributions=[
                xai_service_pb2.ShapContributions(
                    feature_name=str(r["feature"]),
                    feature_value=float(r.get("feature_value") or 0.0),
                    shap_value=float(r["shap"]),
                )
                for r in shap_explanations["contributions"]
            ],
        )

        # optional partial with top-3
        top3 = xai_service_pb2.ExplanationsResponse()
        top3.CopyFrom(final)
        del top3.shap_contributions[:]
        top3.shap_contributions.extend(final.shap_contributions[:3])
        progress_cb(85, "Top-3 ready", partial_result=top3)

        progress_cb(95, "Finalizing")
        return final


class FeatureImportanceHandler(BaseExplanationHandler):
    """experimentExplanation → permutation FI over surrogate"""
    def handle(self, request, explanation_type, progress_cb=lambda *a, **k: None):
        if explanation_type != 'experimentExplanation':
            raise ValueError(f"Unknown explanation type: {explanation_type}")

        experiment_configs = request.experiment_configs
        progress_cb(10, "Training surrogate (experiment)")
        keep_common_variability_points(experiment_configs)

        hyper_space = create_hyperspace(experiment_configs)
        hyper_df, sorted_metrics = create_hyper_df(experiment_configs)
        surrogate_model = self._load_or_train_surrogate_model(hyper_df, sorted_metrics)

        progress_cb(60, "Computing permutation importance")
        result = permutation_importance(
            surrogate_model,
            hyper_df,
            np.array(sorted_metrics),
            scoring='neg_root_mean_squared_error',
            n_repeats=5,
            random_state=42
        )

        feature_importances = list(zip(hyper_df.columns, result.importances_mean))
        sorted_features = sorted(feature_importances, key=lambda x: x[1], reverse=True)
        feat_imp_df = pd.DataFrame(sorted_features, columns=['Feature', 'Importance'])

        table_contents_featimp = {
            col: xai_service_pb2.TableContents(index=i + 1, values=feat_imp_df[col].astype(str).tolist())
            for i, col in enumerate(feat_imp_df.columns)
        }

        progress_cb(90, "Formatting")
        return xai_service_pb2.ExplanationsResponse(
            explainability_type=explanation_type,
            explanation_method='feature_importance',
            explainability_model="",
            plot_name='Experiment Variability Point Importance',
            plot_descr="Impact of different options on the selected metric.",
            plot_type='Table',
            hyperparameter_list=feat_imp_df['Feature'].tolist(),
            table_contents=table_contents_featimp,
        )
