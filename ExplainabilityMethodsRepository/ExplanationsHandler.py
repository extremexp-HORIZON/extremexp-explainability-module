import xai_service_pb2_grpc
import xai_service_pb2
import joblib
import dill as pickle
from modules.lib import *
from ExplainabilityMethodsRepository.pdp import partial_dependence_1D,partial_dependence_2D
from ExplainabilityMethodsRepository.ALE_generic import ale 
from sklearn.inspection import partial_dependence
import ast
import dice_ml
from aix360.algorithms.protodash import ProtodashExplainer
from ExplainabilityMethodsRepository.src.glance.iterative_merges.iterative_merges import C_GLANCE,cumulative
from raiutils.exceptions import UserConfigValidationException
import grpc

class BaseExplanationHandler:
    """Base class for all explanation handlers."""
    
    def handle(self, request, models, data, model_name, explanation_type):
        raise NotImplementedError("Subclasses should implement this method")

    def _load_model(self, model_path, model_name):
        """Helper to load model (same as before)."""
        import dill as pickle
        try:
            with open(model_path, 'rb') as f:
                if model_name == 'Ideko_model':
                    return pickle.load(f)
                else:
                    return joblib.load(f)
        except FileNotFoundError:
            print(f"Model '{model_path}' does not exist.")
            return None

    def _load_or_train_surrogate_model(self,workflows):
        surrogate_model, hyperparameters_list = proxy_model(workflows, 'XGBoostRegressor')
        return surrogate_model, hyperparameters_list
        
    def _load_or_train_cf_surrogate_model(self, models, model_name, original_model, train, train_labels,query):
        if model_name =='Ideko_model':
            try:
                with open(models[model_name]['cfs_surrogate_model'], 'rb') as f:
                    surrogate_model = joblib.load(f)
                    proxy_dataset = pd.read_csv(models[model_name]['cfs_surrogate_dataset'],index_col=0)
            except FileNotFoundError:
                print("Surrogate model does not exist. Training new surrogate model")
        else:
            surrogate_model , proxy_dataset = instance_proxy(train,train_labels,original_model, query.loc[0],original_model.param_grid)
            # joblib.dump(surrogate_model, models[model_name]['cfs_surrogate_model'])  
            # proxy_dataset.to_csv(models[model_name]['cfs_surrogate_dataset'])
        return surrogate_model, proxy_dataset


class GLANCEHandler(BaseExplanationHandler):
    def handle(self, request, models, data, model_name, explanation_type):
        if explanation_type == 'featureExplanation':
            gcf_size = request.gcf_size  # Global counterfactual size
            cf_generator = request.cf_generator  # Counterfactual generator method
            cluster_action_choice_algo = request.cluster_action_choice_algo

            model_id = request.model_id
            trained_models = self._load_model(models[model_name]['all_models'], model_name)
            model = trained_models[model_id]
            X_train = pd.read_csv(data[model_name]['train'],index_col=0) 
            train_labels = pd.read_csv(data[model_name]['train_labels'],index_col=0)
            X_test = pd.read_csv(data[model_name]['test'],index_col=0) 
            test_labels = pd.read_csv(data[model_name]['test_labels'],index_col=0) 
            X_train['target'] = train_labels
            X_test['target'] = test_labels
            data = pd.concat([X_train,X_test])

            X_test.drop(columns='target',inplace=True)
            preds = model.predict(X_test)
            X_test['target'] = preds
            affected = X_test[X_test.target == 0]

            global_method = C_GLANCE(
                model=model,
                initial_clusters=50,
                final_clusters=gcf_size,
                num_local_counterfactuals=10,
            )

            global_method.fit(
                data.drop(columns=['target']),
                data['target'],
                X_test,
                X_test.drop(columns='target').columns.tolist(),
                cf_generator=cf_generator,
                cluster_action_choice_algo=cluster_action_choice_algo
            )
            try:
                clusters, clusters_res, eff, cost = global_method.explain_group(affected.drop(columns='target')[:20])

                sorted_actions_dict = dict(sorted(clusters_res.items(), key=lambda item: item[1]['cost']))
                actions = [stats["action"] for i,stats in sorted_actions_dict.items()]
                i=1
                all_clusters = {}
                num_features = X_test._get_numeric_data().columns.to_list()
                cate_features = X_test.columns.difference(num_features)

                for key in clusters:
                    clusters[key]['Cluster'] = i
                    all_clusters[i] = clusters[key]
                    i=i+1

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
                    eff_act = pred_list[i].sum()/len(affected[:20])
                    cost_act = costs[i-1][costs[i-1] != np.inf].sum()/pred_list[i].sum()
                    eff_cost_actions[i] = {'eff':eff_act , 'cost':cost_act}

                result['Cluster'] = cluster
                
                result['Chosen_Action'] = chosen_actions
                result['Chosen_Action'] = result['Chosen_Action'] + 1
                result = result.replace(np.inf , '-')


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
        
                return xai_service_pb2.ExplanationsResponse(
                    explainability_type = explanation_type,
                    explanation_method = 'global_counterfactuals',
                    explainability_model = model_name,
                    plot_name = 'Global Counterfactual Explanations',
                    plot_descr = "Counterfactual Explanations identify the minimal changes needed to alter a machine learning model's prediction for a given instance.",
                    plot_type = 'Table',
                    feature_list = data.drop(columns=['target']).columns.tolist(),
                    hyperparameter_list = [],
                    affected_clusters = {col: xai_service_pb2.TableContents(index=i+1,values=result[col].astype(str).tolist()) for i,col in enumerate(result.columns)},
                    eff_cost_actions = {
                        str(key): xai_service_pb2.EffCost(
                            eff=value['eff'],  
                            cost=value['cost']  
                        ) for key, value in eff_cost_actions.items()
                    },
                    TotalEffectiveness = float(round(eff/20,3)),
                    TotalCost = float(round(cost/eff,2)),
                    actions = {col: xai_service_pb2.TableContents(index=i+1,values=actions_ret[col].astype(str).tolist()) for i,col in enumerate(actions_ret.columns)},

                ) 
            except UserConfigValidationException as e:
                # Handle known Dice error for missing counterfactuals
                if str(e) == "No counterfactuals found for any of the query points! Kindly check your configuration.":
                    return xai_service_pb2.ExplanationsResponse(
                    explainability_type=explanation_type,
                    explanation_method='global_counterfactuals',
                    explainability_model=model_name,
                    plot_name='Error',
                    plot_descr=f"An error occurred while generating the explanation: {str(e)}",
                    plot_type='Error',
                    feature_list=[],
                    hyperparameter_list=[],
                    affected_clusters={},
                    eff_cost_actions={},
                    TotalEffectiveness=0.0,
                    TotalCost=0.0,
                    actions={},
                )

class PDPHandler(BaseExplanationHandler):

    def handle(self, request, models, data, model_name, explanation_type):

        if explanation_type == 'featureExplanation':
            model_id = request.model_id

            original_model = self._load_model(models[model_name]['original_model'], model_name)
            trained_models = self._load_model(models[model_name]['all_models'], model_name)
            model = trained_models[model_id]
            dataframe = pd.DataFrame()
            dataframe = pd.read_csv(data[model_name]['train'],index_col=0) 
            if not request.feature1:
                print('Feature is missing, initializing with first feature from features list')
                features = dataframe.columns.tolist()[0]
            else:
                features = request.feature1
            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
            numeric_features = dataframe.select_dtypes(include=numerics).columns.tolist()
            categorical_features = dataframe.columns.drop(numeric_features)

            pdp = partial_dependence(model, dataframe, features = [dataframe.columns.tolist().index(features)],
                                    feature_names=dataframe.columns.tolist(),categorical_features=categorical_features)
            
            if type(pdp['grid_values'][0][0]) == str:
                axis_type='categorical' 
            else: axis_type = 'numerical'

            pdp_grid = [value.tolist() for value in pdp['grid_values']][0]
            pdp_vals = [value.tolist() for value in pdp['average']][0]
            return xai_service_pb2.ExplanationsResponse(
                explainability_type = explanation_type,
                explanation_method = 'pdp',
                explainability_model = model_name,
                plot_name = 'Partial Dependence Plot (PDP)',
                plot_descr = "PD (Partial Dependence) Plots show how a feature affects a model's predictions, holding other features constant, to illustrate feature impact.",
                plot_type = 'LinePlot',
                features = xai_service_pb2.Features(
                            feature1=features, 
                            feature2=''),
                feature_list = dataframe.columns.tolist(),
                hyperparameter_list = [],
                xAxis = xai_service_pb2.Axis(
                            axis_name=f'{features}', 
                            axis_values=[str(value) for value in pdp_grid], 
                            axis_type=axis_type  
                ),
                yAxis = xai_service_pb2.Axis(
                            axis_name='PDP Values', 
                            axis_values=[str(value) for value in pdp_vals], 
                            axis_type='numerical'
                ),
                zAxis = xai_service_pb2.Axis(
                            axis_name='', 
                            axis_values='', 
                            axis_type=''                    
                )
            )
        else:
            workflows = request.workflows
            workflows = ast.literal_eval(workflows)
            print('Training Surrogate Model')
            surrogate_model, hyperparameters_list = self._load_or_train_surrogate_model(workflows)
            
            param_grid = transform_to_param_grid(hyperparameters_list)
            param_grid = transform_grid(param_grid)
            param_space, name = dimensions_aslists(param_grid)
            space = Space(param_space)

            feats = {}
            for index,n in enumerate(name):
                feats[n] = index

            plot_dims = []
            for row in range(space.n_dims):
                # if space.dimensions[row].is_constant:
                #     continue
                plot_dims.append((row, space.dimensions[row]))
                
            pdp_samples = space.rvs(n_samples=1000,random_state=123456)
            if not request.feature1:
                print('Feature is missing, initializing with first hyperparameter from hyperparameters list')
                feature = name[0]
            else: 
                feature = request.feature1

            xi = []
            yi=[]
            index, dim = plot_dims[feats[feature]]
            xi1, yi1 = partial_dependence_1D(space, surrogate_model,
                                                index,
                                                samples=pdp_samples,
                                                name=name,
                                                n_points=100)

            xi.append(xi1)
            yi.append(yi1)
                
            x = [arr.tolist() for arr in xi]
            y = [arr for arr in yi]
            axis_type = 'categorical' if isinstance(x[0][0], str) else 'numerical'
            return xai_service_pb2.ExplanationsResponse(
                explainability_type=explanation_type,
                explanation_method='pdp',
                explainability_model=model_name,
                plot_name='Partial Dependence Plot (PDP)',
                plot_descr="PD (Partial Dependence) Plots show how different hyperparameter values affect a model's accuracy, holding other hyperparameters constant.",
                plot_type='LinePlot',
                features=xai_service_pb2.Features(
                    feature1=feature, 
                    feature2=''
                ),
                feature_list = [],
                hyperparameter_list = name,
                xAxis=xai_service_pb2.Axis(
                    axis_name=f'{feature}',
                    axis_values=[str(value) for value in x[0]],
                    axis_type=axis_type
                ),
                yAxis=xai_service_pb2.Axis(
                    axis_name='PDP Values',
                    axis_values=[str(value) for value in y[0]],
                    axis_type='numerical'
                ),
                zAxis=xai_service_pb2.Axis(
                    axis_name='',
                    axis_values='',
                    axis_type=''
                ),
            )

class TwoDPDPHandler(BaseExplanationHandler):

    def handle(self, request, models, data, model_name, explanation_type):
        if explanation_type == 'featureExplanation':
            model_id = request.model_id
            original_model = self._load_model(models[model_name]['original_model'], model_name)
            trained_models = self._load_model(models[model_name]['all_models'], model_name)
            model = trained_models[model_id]
            dataframe = pd.read_csv(data[model_name]['train'],index_col=0)                        
            if not request.feature1:
                print('Feature is missing, initializing with first feature from features list')
                feature1 = dataframe.columns.tolist()[0]
                feature2 = dataframe.columns.tolist()[1]
            else: 
                feature1 = request.feature1
                feature2 = request.feature2
            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

            numeric_features = dataframe.select_dtypes(include=numerics).columns.tolist()
            categorical_features = dataframe.columns.drop(numeric_features)

            pdp = partial_dependence(model, dataframe, features = [(dataframe.columns.tolist().index(feature1),dataframe.columns.tolist().index(feature2))],
                                    feature_names=dataframe.columns.tolist(),categorical_features=categorical_features)
            

            if type(pdp['grid_values'][0][0]) == str:
                axis_type_0='categorical' 
            else: axis_type_0 = 'numerical'

            if type(pdp['grid_values'][1][0]) == str:
                axis_type_1='categorical' 
            else: axis_type_1 = 'numerical'


            pdp_grid_1 = [value.tolist() for value in pdp['grid_values']][0]
            pdp_grid_2 = [value.tolist() for value in pdp['grid_values']][1]
            pdp_vals = [value.tolist() for value in pdp['average']][0]
            return xai_service_pb2.ExplanationsResponse(
                explainability_type = explanation_type,
                explanation_method = '2dpdp',
                explainability_model = model_name,
                plot_name = '2D-Partial Dependence Plot (2D-PDP)',
                plot_descr = "2D-PD plots visualize how the model's accuracy changes when two hyperparameters vary.",
                plot_type = 'ContourPlot',
                features = xai_service_pb2.Features(
                            feature1=feature1, 
                            feature2=feature2),
                feature_list = dataframe.columns.tolist(),
                hyperparameter_list = [],
                xAxis = xai_service_pb2.Axis(
                            axis_name=f'{feature1}', 
                            axis_values=[str(value) for value in pdp_grid_1], 
                            axis_type=axis_type_0  
                ),
                yAxis = xai_service_pb2.Axis(
                            axis_name=f'{feature2}', 
                            axis_values=[str(value) for value in pdp_grid_2], 
                            axis_type=axis_type_1
                ),
                zAxis = xai_service_pb2.Axis(
                            axis_name='', 
                            axis_values=[str(value) for value in pdp_vals], 
                            axis_type='numerical'                    
                ),
            )
        else:
            workflows = request.workflows
            workflows = ast.literal_eval(workflows)
            print('Training Surrogate Model')
            surrogate_model, hyperparameters_list = self._load_or_train_surrogate_model(workflows)
            
            param_grid = transform_to_param_grid(hyperparameters_list)
            param_space, name = dimensions_aslists(param_grid)
            space = Space(param_space)
            if not request.feature1:
                print('Feature is missing, initializing with first hyperparameter from hyperparameters list')
                feature1 = name[0]
                feature2 = name[1]
            else: 
                feature1 = request.feature1
                feature2 = request.feature2
            
            index1 = name.index(feature1)
            index2 = name.index(feature2)


            plot_dims = []
            for row in range(space.n_dims):
                if space.dimensions[row].is_constant:
                    continue
                plot_dims.append((row, space.dimensions[row]))
            
            pdp_samples = space.rvs(n_samples=1000,random_state=123456)

            _ ,dim_1 = plot_dims[index1]
            _ ,dim_2 = plot_dims[index2]
            xi, yi, zi = partial_dependence_2D(space, surrogate_model,
                                                    index1, index2,
                                                    pdp_samples,name, 100)
            
            
            x = [arr.tolist() for arr in xi]
            y = [arr.tolist() for arr in yi]
            z = [arr.tolist() for arr in zi]

            return xai_service_pb2.ExplanationsResponse(
                        explainability_type = explanation_type,
                        explanation_method = '2dpdp',
                        explainability_model = model_name,
                        plot_name = '2D-Partial Dependence Plot (2D-PDP)',
                        plot_descr = "2D-PD plots visualize how the model's accuracy changes when two hyperparameters vary.",
                        plot_type = 'ContourPlot',
                        features = xai_service_pb2.Features(
                                    feature1=feature1, 
                                    feature2=feature2),
                        feature_list = [],
                        hyperparameter_list = name,
                        xAxis = xai_service_pb2.Axis(
                                    axis_name=f'{feature2}', 
                                    axis_values=[str(value) for value in x], 
                                    axis_type='categorical' if isinstance(x[0], str) else 'numerical'
                        ),
                        yAxis = xai_service_pb2.Axis(
                                    axis_name=f'{feature1}', 
                                    axis_values=[str(value) for value in y], 
                                    axis_type='categorical' if isinstance(y[0], str) else 'numerical'
                        ),
                        zAxis = xai_service_pb2.Axis(
                                    axis_name='', 
                                    axis_values=[str(value) for value in z], 
                                    axis_type='numerical' 
                        ),
                        
            )
    
class ALEHandler(BaseExplanationHandler):

    def handle(self, request, models, data, model_name, explanation_type):
        if explanation_type == 'featureExplanation':
            original_model = self._load_model(models[model_name]['original_model'], model_name)
            trained_models = self._load_model(models[model_name]['all_models'], model_name)
            model_id = request.model_id
            model = trained_models[model_id]

            dataframe = pd.read_csv(data[model_name]['train'],index_col=0) 
            if not request.feature1:
                print('Feature is missing, initializing with first features from features list')
                features = dataframe.columns.tolist()[0]
            else: 
                features = request.feature1

            if dataframe[features].dtype in ['int','float']:
                ale_eff = ale(X=dataframe, model=model, feature=[features],plot=False, grid_size=50, include_CI=True, C=0.95)
            else:
                ale_eff = ale(X=dataframe, model=model, feature=[features],plot=False, grid_size=50, predictors=dataframe.columns.tolist(), include_CI=True, C=0.95)

            return xai_service_pb2.ExplanationsResponse(
                explainability_type = explanation_type,
                explanation_method = 'ale',
                explainability_model = model_name,
                plot_name = 'Accumulated Local Effects Plot (ALE)',
                plot_descr = "ALE plots illustrate the effect of a single feature on the predicted outcome of a machine learning model.",
                plot_type = 'LinePLot',
                features = xai_service_pb2.Features(
                            feature1=features, 
                            feature2=''),
                feature_list = dataframe.columns.tolist(),
                hyperparameter_list = [],
                xAxis = xai_service_pb2.Axis(
                            axis_name=f'{features}', 
                            axis_values=[str(value) for value in ale_eff.index.tolist()], 
                            axis_type='categorical' if isinstance(ale_eff.index.tolist()[0], str) else 'numerical'
                ),
                yAxis = xai_service_pb2.Axis(
                            axis_name='ALE Values', 
                            axis_values=[str(value) for value in ale_eff.eff.tolist()], 
                            axis_type='categorical' if isinstance(ale_eff.eff.tolist()[0], str) else 'numerical'
                ),
                zAxis = xai_service_pb2.Axis(
                            axis_name='', 
                            axis_values='', 
                            axis_type=''                    
                ),
                
            )
        else:
            workflows = request.workflows
            workflows = ast.literal_eval(workflows)
            print('Training Surrogate Model')
            surrogate_model, hyperparameters_list = self._load_or_train_surrogate_model(workflows)
            
            param_grid = transform_to_param_grid(hyperparameters_list)
            param_space, name = dimensions_aslists(param_grid)
            space = Space(param_space)

            plot_dims = []
            for row in range(space.n_dims):
                if space.dimensions[row].is_constant:
                    continue
                plot_dims.append((row, space.dimensions[row]))

            if not request.feature1:
                print('Feature is missing, initializing with first hyperparameter from hyperparameter list')
                feature1 = name[0]
            else: 
                feature1 = request.feature1

            pdp_samples = space.rvs(n_samples=1000,random_state=123456)
            data = pd.DataFrame(pdp_samples,columns=[n for n in name])

            if data[feature1].dtype in ['int','float']:
                # data = data.drop(columns=feat)
                # data[feat] = d1[feat]  
                ale_eff = ale(X=data, model=surrogate_model, feature=[feature1],plot=False, grid_size=50, include_CI=True, C=0.95)
            else:
                ale_eff = ale(X=data, model=surrogate_model, feature=[feature1],plot=False, grid_size=50,predictors=data.columns.tolist(), include_CI=True, C=0.95)

            return xai_service_pb2.ExplanationsResponse(
                explainability_type = explanation_type,
                explanation_method = 'ale',
                explainability_model = model_name,
                plot_name = 'Accumulated Local Effects Plot (ALE)',
                plot_descr = "ALE Plots illustrate the effect of a single hyperparameter on the accuracy of a machine learning model.",
                plot_type = 'LinePLot',
                features = xai_service_pb2.Features(
                            feature1=feature1, 
                            feature2=''),
                feature_list = [],            
                hyperparameter_list = name,
                xAxis = xai_service_pb2.Axis(
                            axis_name=f'{feature1}', 
                            axis_values=[str(value) for value in ale_eff.index.tolist()], 
                            axis_type='categorical' if isinstance(ale_eff.index.tolist()[0], str) else 'numerical'
                ),
                yAxis = xai_service_pb2.Axis(
                            axis_name='ALE Values', 
                            axis_values=[str(value) for value in ale_eff.eff.tolist()], 
                            axis_type='categorical' if isinstance(ale_eff.eff.tolist()[0], str) else 'numerical'
                ),
                zAxis = xai_service_pb2.Axis(
                            axis_name='', 
                            axis_values='', 
                            axis_type=''                    
                ),
                
            )
        
class CounterfactualsHandler(BaseExplanationHandler):

    def handle(self, request, models, data, model_name, explanation_type):
        
        if explanation_type == 'featureExplanation':
            model_id = request.model_id
            query = request.query
            query = ast.literal_eval(query)
            query = pd.DataFrame([query])

            query = query.drop(columns=['id','label'])
            original_model = self._load_model(models[model_name]['original_model'], model_name)
            trained_models = self._load_model(models[model_name]['all_models'], model_name)
            model = trained_models[model_id]

            target = request.target


            train = pd.read_csv(data[model_name]['train'],index_col=0)  
            train_labels = pd.read_csv(data[model_name]['train_labels'],index_col=0)  
            
            dataframe = pd.concat([train.reset_index(drop=True), train_labels.reset_index(drop=True)], axis = 1)

            d = dice_ml.Data(dataframe=dataframe, 
                continuous_features=dataframe.drop(columns=target).select_dtypes(include='number').columns.tolist()
                , outcome_name=target)
    
            # Using sklearn backend
            m = dice_ml.Model(model=original_model, backend="sklearn")
            # Using method=random for generating CFs
            exp = dice_ml.Dice(d, m, method="random")
            e1 = exp.generate_counterfactuals(query.drop(columns=['prediction']), total_CFs=5, desired_class="opposite",sample_size=5000)
            e1.visualize_as_dataframe(show_only_changes=True)
            cfs = e1.cf_examples_list[0].final_cfs_df
            query.rename(columns={"prediction": target},inplace=True)
            # for col in query.columns:
            #     cfs[col] = cfs[col].apply(lambda x: '-' if x == query.iloc[0][col] else x)
            cfs['Type'] = 'Counterfactual'
            query['Type'] = 'Factual'
            
            #cfs = cfs.to_parquet(None)
            cfs = pd.concat([query,cfs])

            return xai_service_pb2.ExplanationsResponse(
                explainability_type = explanation_type,
                explanation_method = 'counterfactuals',
                explainability_model = model_name,
                plot_name = 'Counterfactual Explanations',
                plot_descr = "Counterfactual Explanations identify the minimal changes needed to alter a machine learning model's prediction for a given instance.",
                plot_type = 'Table',
                feature_list = dataframe.columns.tolist(),
                hyperparameter_list = [],
                table_contents = {col: xai_service_pb2.TableContents(index=i+1,values=cfs[col].astype(str).tolist()) for i,col in enumerate(cfs.columns)}
            )
        
        else:
            original_model = self._load_model(models[model_name]['original_model'], model_name)
            model_id = request.model_id
            
            
            if model_name == 'I2Cat_Phising_model':
                query = request.query
                
                query = ast.literal_eval(query)
                query = pd.DataFrame([query])
                train = pd.read_csv(data[model_name]['train'],index_col=0) 
                train_labels = pd.read_csv(data[model_name]['train_labels'],index_col=0) 
                trained_models = self._load_model(models[model_name]['all_models'], model_name)
                model = trained_models[model_id]
                print('Creating Proxy Dataset and Model')
                surrogate_model , proxy_dataset = self._load_or_train_cf_surrogate_model(models, model_name, original_model, train, train_labels,query)
                param_grid = transform_grid(original_model.param_grid)
                param_space, name = dimensions_aslists(param_grid)
                space = Space(param_space)

                plot_dims = []
                for row in range(space.n_dims):
                    if space.dimensions[row].is_constant:
                        continue
                    plot_dims.append((row, space.dimensions[row]))
                iscat = [isinstance(dim[1], Categorical) for dim in plot_dims]
                categorical = [name[i] for i,value in enumerate(iscat) if value == True]
                proxy_dataset[categorical] = proxy_dataset[categorical].astype(str)
                params = model.get_params()
                query = pd.DataFrame(data = {'Model__learning_rate':params['Model__learning_rate'], 'Model__max_depth':params['Model__max_depth'],	'Model__min_child_weight':params['Model__min_child_weight'],'Model__n_estimators':params['Model__n_estimators'],	'preprocessor__num__scaler':params['preprocessor__num__scaler']},index=[0])
                #query = pd.DataFrame.from_dict(original_model.best_params_,orient='index').T
                query[categorical] = query[categorical].astype(str)
                d = dice_ml.Data(dataframe=proxy_dataset, continuous_features=proxy_dataset.drop(columns='BinaryLabel').select_dtypes(include='number').columns.tolist(), outcome_name='BinaryLabel')
                m = dice_ml.Model(model=surrogate_model, backend="sklearn")
                exp = dice_ml.Dice(d, m, method="random")
                e1 = exp.generate_counterfactuals(query, total_CFs=5, desired_class="opposite",sample_size=5000)
                dtypes_dict = proxy_dataset.drop(columns='BinaryLabel').dtypes.to_dict()
                cfs = e1.cf_examples_list[0].final_cfs_df
                for col, dtype in dtypes_dict.items():
                    cfs[col] = cfs[col].astype(dtype)
                    scaled_query, scaled_cfs = min_max_scale(proxy_dataset=proxy_dataset,factual=query.copy(deep=True),counterfactuals=cfs.copy(deep=True),label='BinaryLabel')
            else:
                try:
                    with open(models[model_name]['cfs_surrogate_model'], 'rb') as f:
                        surrogate_model = joblib.load(f)
                        proxy_dataset = pd.read_csv(models[model_name]['cfs_surrogate_dataset'],index_col=0)
                except FileNotFoundError:
                    print("Surrogate model does not exist. Training new surrogate model")
                param_grid = transform_grid(original_model.param_grid)
                param_space, name = dimensions_aslists(param_grid)
                space = Space(param_space)

                plot_dims = []
                for row in range(space.n_dims):
                    if space.dimensions[row].is_constant:
                        continue
                    plot_dims.append((row, space.dimensions[row]))
                iscat = [isinstance(dim[1], Categorical) for dim in plot_dims]
                categorical = [name[i] for i,value in enumerate(iscat) if value == True]
                proxy_dataset[categorical] = proxy_dataset[categorical].astype(str)
                params = original_model.best_estimator_.get_params()
                query = pd.DataFrame(data = {'batch_size':64,'epochs':50,'model__activation_function': 'relu','model__units': [[512,512,512]]},index=[0])
                query[categorical] = query[categorical].astype(str)
                d = dice_ml.Data(dataframe=proxy_dataset, 
                    continuous_features=proxy_dataset.drop(columns='Label').select_dtypes(include='number').columns.tolist()
                    , outcome_name='Label')
                m = dice_ml.Model(model=surrogate_model, backend="sklearn")
                exp = dice_ml.Dice(d, m, method="random")
                e1 = exp.generate_counterfactuals(query, total_CFs=5, desired_class=2,sample_size=5000)
                dtypes_dict = proxy_dataset.drop(columns='Label').dtypes.to_dict()
                cfs = e1.cf_examples_list[0].final_cfs_df
                scaled_query, scaled_cfs = min_max_scale(proxy_dataset=proxy_dataset,factual=query.copy(deep=True),counterfactuals=cfs.copy(deep=True),label='Label')
            cfs['Cost'] = cf_difference(scaled_query, scaled_cfs)
            cfs = cfs.sort_values(by='Cost')
            cfs['Type'] = 'Counterfactual'
            #query['BinaryLabel'] = 1
            query['Cost'] = '-'
            query['Type'] = 'Factual'
            if model_name == 'Ideko_model':
                query['Label'] = 1
                query.rename(columns={'model__activation_function': 'Activ_Func', 'model__units': 'nodes'}, inplace=True)
                cfs.rename(columns={'model__activation_function': 'Activ_Func', 'model__units': 'nodes'}, inplace=True)
            else:
                query['BinaryLabel'] = 1
            cfs = pd.concat([query,cfs])

            return xai_service_pb2.ExplanationsResponse(
                explainability_type = explanation_type,
                explanation_method = 'counterfactuals',
                explainability_model = model_name,
                plot_name = 'Counterfactual Explanations',
                plot_descr = "Counterfactual Explanations identify the minimal changes on hyperparameter values in order to correctly classify a given missclassified instance.",
                plot_type = 'Table',
                feature_list = [],
                hyperparameter_list = name,
                table_contents = {col: xai_service_pb2.TableContents(index=i+1,values=cfs[col].astype(str).tolist()) for i,col in enumerate(cfs.columns)}
            )
        

class PrototypesHandler(BaseExplanationHandler):

    def handle(self, request, models, data, model_name, explanation_type):
        model_id = request.model_id
        query = request.query
        query = ast.literal_eval(query)
        query = pd.DataFrame([query])
        trained_models = self._load_model(models[model_name]['all_models'], model_name)
        model = trained_models[model_id]
        train = pd.read_csv(data[model_name]['train'],index_col=0) 
        train_labels = pd.read_csv(data[model_name]['train_labels'],index_col=0) 
        train['label'] = train_labels
        
        explainer = ProtodashExplainer()
        reference_set_train = train[train.label==0].drop(columns='label')

        (W, S, _)= explainer.explain(np.array(query.drop(columns='predictions')).reshape(1,-1),np.array(reference_set_train),m=5)
        prototypes = reference_set_train.reset_index(drop=True).iloc[S, :].copy()
        prototypes['predictions'] =  model.predict(prototypes)
        prototypes = prototypes.reset_index(drop=True).T
        prototypes.rename(columns={0:'Prototype1',1:'Prototype2',2:'Prototype3',3:'Prototype4',4:'Prototype5'},inplace=True)
        prototypes = prototypes.reset_index()

        prototypes.set_index('index', inplace=True)

        # Create a new empty dataframe for boolean results
        boolean_df = pd.DataFrame(index=prototypes.index)

        # Iterate over each column and compare with the series
        for col in prototypes.columns:
            boolean_df[col] = prototypes[col] == query.loc[0][prototypes.index].values

        prototypes.reset_index(inplace=True)
        prototypes= prototypes.append([{'index': 'Weights', 'Prototype1':np.around(W/np.sum(W), 2)[0],'Prototype2':np.around(W/np.sum(W), 2)[1],'Prototype3':np.around(W/np.sum(W), 2)[2],'Prototype4':np.around(W/np.sum(W), 2)[3],'Prototype5':np.around(W/np.sum(W), 2)[4]}])
        boolean_df=boolean_df.append([{'index': 'Weights', 'Prototype1':False,'Prototype2':False,'Prototype3':False,'Prototype4':False,'Prototype5':False}])

        print(prototypes)
        # Create table_contents dictionary for prototypes
        table_contents =  {col: xai_service_pb2.TableContents(index=i+1,values=prototypes[col].astype(str).tolist(),colour =boolean_df[col].astype(str).tolist()) for i,col in enumerate(prototypes.columns)}


        return xai_service_pb2.ExplanationsResponse(
            explainability_type = explanation_type,
            explanation_method = 'prototypes',
            explainability_model = model_name,
            plot_name = 'Prototypes',
            plot_descr = "Prototypes are prototypical examples that capture the underlying distribution of a dataset. It also weights each prototype to quantify how well it represents the data.",
            plot_type = 'Table',
            feature_list = train.columns.tolist(),
            hyperparameter_list = [],
            table_contents = table_contents
        )