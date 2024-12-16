import grpc
from concurrent import futures
import xai_service_pb2_grpc
import xai_service_pb2
from xai_service_pb2_grpc import ExplanationsServicer
import json
from concurrent import futures
import json
from sklearn.inspection import partial_dependence
from modules.lib import *
from ExplainabilityMethodsRepository.ALE_generic import ale
import joblib
from PyALE import ale
from ExplainabilityMethodsRepository.ExplanationsHandler import *
from ExplainabilityMethodsRepository.config import shared_resources
from ExplainabilityMethodsRepository.src.glance.iterative_merges.iterative_merges import apply_action_pandas

class ExplainabilityExecutor(ExplanationsServicer):

    def GetExplanation(self, request, context):
        print('Reading data')
        models = json.load(open("metadata/models.json"))
        data = json.load(open("metadata/datasets.json"))
        dataframe = pd.DataFrame()
        label = pd.DataFrame()

        #for request in request_iterator:
        explanation_type = request.explanation_type
        explanation_method = request.explanation_method
        model_name = request.model

        dispatch_table = {
            (explanation_type, 'pdp'): PDPHandler(),
            (explanation_type, '2dpdp'): TwoDPDPHandler(),
            (explanation_type, 'ale'): ALEHandler(),
            (explanation_type, 'counterfactuals'): CounterfactualsHandler(),
            ('featureExplanation', 'prototypes'): PrototypesHandler(),
            ('featureExplanation', 'global_counterfactuals'): GLANCEHandler()
            # Add more handlers as needed
        }
        
        handler = dispatch_table.get((explanation_type, explanation_method))

        if handler:
            return handler.handle(request, models, data, model_name, explanation_type)
        else:
            raise ValueError(f"Unsupported explanation method '{explanation_method}' for type '{explanation_type}'")
        
    def ApplyAffectedActions(self,request,context):
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

            return xai_service_pb2.ApplyAffectedActionsResponse(
                applied_affected_actions=applied_affected_response
            )
        
        except Exception as e:
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return xai_service_pb2.ApplyAffectedActionsResponse()

    def Initialization(self, request, context):
        models = json.load(open("metadata/models.json"))
        data = json.load(open("metadata/datasets.json"))
        model_name = request.model_name

        # Load Data
        try:
            with open(models[model_name]['original_model'], 'rb') as f:
                original_model = joblib.load(f)
        except FileNotFoundError:
            print("Model does not exist. Load existing model.")

        train = pd.read_csv(data[model_name]['train'],index_col=0) 
        train_labels = pd.read_csv(data[model_name]['train_labels'],index_col=0) 
        test = pd.read_csv(data[model_name]['test'],index_col=0) 
        test_labels = pd.read_csv(data[model_name]['test_labels'],index_col=0) 
        test['label'] = test_labels
        dataframe = pd.concat([train.reset_index(drop=True), train_labels.reset_index(drop=True)], axis = 1)

        predictions = original_model.predict(test)
        test['Predicted'] = predictions
        test['Label'] = (test['label'] != test['Predicted']).astype(int)

        missclassified_instances = test[test['Label']==1]

        param_grid = original_model.param_grid
        param_grid = transform_grid_plt(param_grid)
        
        # Load surrogate models for PDP - ALE if exists
        try:
            with open(models[model_name]['pdp_ale_surrogate_model'], 'rb') as f:
                pdp_ale_surrogate_model = joblib.load(f)
        except FileNotFoundError:
            print("Surrogate model does not exist. Training new surrogate model") 
            pdp_ale_surrogate_model = proxy_model(param_grid,original_model,'accuracy','XGBoostRegressor')
            joblib.dump(pdp_ale_surrogate_model, models[model_name]['pdp_ale_surrogate_model'])  

        # ---------------------- Run Explainability Methods for Pipeline -----------------------------------------------

        #PDP
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


        xi = []
        yi=[]
        index, dim = plot_dims[feats[list(param_grid.keys())[0]]]
        xi1, yi1 = partial_dependence_1D(space, pdp_ale_surrogate_model,
                                            index,
                                            samples=pdp_samples,
                                            n_points=100)

        xi.append(xi1)
        yi.append(yi1)
            
        x = [arr.tolist() for arr in xi]
        y = [arr for arr in yi]
        axis_type = 'categorical' if isinstance(x[0][0], str) else 'numerical'

        # # 2D PDP
        index1 = name.index(list(param_grid.keys())[0])
        index2 = name.index(list(param_grid.keys())[1])
        
        pdp_samples = space.rvs(n_samples=1000,random_state=123456)

        _ ,dim_1 = plot_dims[index1]
        _ ,dim_2 = plot_dims[index2]
        xi, yi, zi = partial_dependence_2D(space, pdp_ale_surrogate_model,
                                                index1, index2,
                                                pdp_samples, 100)
        
        
        x2d = [arr.tolist() for arr in xi]
        y2d = [arr.tolist() for arr in yi]
        z = [arr.tolist() for arr in zi]
        # # ALE
        pdp_samples = space.rvs(n_samples=1000,random_state=123456)
        data = pd.DataFrame(pdp_samples,columns=[n for n in name])

        if data[list(param_grid.keys())[0]].dtype in ['int','float']:
            # data = data.drop(columns=feat)
            # data[feat] = d1[feat]  
            ale_eff = ale(X=data, model=pdp_ale_surrogate_model, feature=[list(param_grid.keys())[0]],plot=False, grid_size=50, include_CI=True, C=0.95)
        else:
            ale_eff = ale(X=data, model=pdp_ale_surrogate_model, feature=[list(param_grid.keys())[0]],plot=False, grid_size=50,predictors=data.columns.tolist(), include_CI=True, C=0.95)

        # ---------------------- Run Explainability Methods for Model -----------------------------------------------

        # PD Plots
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        features = train.columns.tolist()[0]
        numeric_features = train.select_dtypes(include=numerics).columns.tolist()
        categorical_features = train.columns.drop(numeric_features)

        pdp = partial_dependence(original_model, train, features = [train.columns.tolist().index(features)],
                                feature_names=train.columns.tolist(),categorical_features=categorical_features)

        pdp_grid = [value.tolist() for value in pdp['grid_values']][0]
        pdp_vals = [value.tolist() for value in pdp['average']][0]

        #ALE Plots
        if train[features].dtype in ['int','float']:
            ale_eff_feat = ale(X=train, model=original_model, feature=[features],plot=False, grid_size=50, include_CI=True, C=0.95)
        else:
            ale_eff_feat = ale(X=train, model=original_model, feature=[features],plot=False, grid_size=50, predictors=train.columns.tolist(), include_CI=True, C=0.95)


        return xai_service_pb2.InitializationResponse(


            feature_explanation = xai_service_pb2.Feature_Explanation(
                                    feature_names=train.columns.tolist(),
                                    plots={'pdp': xai_service_pb2.ExplanationsResponse(
                                                    explainability_type = 'featureExplanation',
                                                    explanation_method = 'pdp',
                                                    explainability_model = model_name,
                                                    plot_name = 'Partial Dependence Plot (PDP)',
                                                    plot_descr = "PD (Partial Dependence) Plots show how a feature affects a model's predictions, holding other features constant, to illustrate feature impact.",
                                                    plot_type = 'LinePLot',
                                                    features = xai_service_pb2.Features(
                                                                feature1=features, 
                                                                feature2=''),
                                                    xAxis = xai_service_pb2.Axis(
                                                                axis_name=f'{features}', 
                                                                axis_values=[str(value) for value in pdp_grid], 
                                                                axis_type='categorical' if isinstance(pdp['grid_values'][0][0], str) else 'numerical'
                                                    ),
                                                    yAxis = xai_service_pb2.Axis(
                                                                axis_name='PDP Values', 
                                                                axis_values=[str(value) for value in pdp_vals], 
                                                                axis_type='numerical'
                                                    ),
                                                ),
                                            'ale': xai_service_pb2.ExplanationsResponse(
                                                    explainability_type = 'featureExplanation',
                                                    explanation_method = 'ale',
                                                    explainability_model = model_name,
                                                    plot_name = 'Accumulated Local Effects Plot (ALE)',
                                                    plot_descr = "ALE plots illustrate the effect of a single feature on the predicted outcome of a machine learning model.",
                                                    plot_type = 'LinePLot',
                                                    features = xai_service_pb2.Features(
                                                                feature1=features, 
                                                                feature2=''),
                                                    xAxis = xai_service_pb2.Axis(
                                                                axis_name=f'{features}', 
                                                                axis_values=[str(value) for value in ale_eff_feat.index.tolist()], 
                                                                axis_type='categorical' if isinstance(ale_eff_feat.index.tolist()[0], str) else 'numerical'
                                                    ),
                                                    yAxis = xai_service_pb2.Axis(
                                                                axis_name='ALE Values', 
                                                                axis_values=[str(value) for value in ale_eff_feat.eff.tolist()], 
                                                                axis_type='numerical'
                                                    ),
                                                ),      
                                            },
                            ),

            hyperparameter_explanation = xai_service_pb2.Hyperparameter_Explanation(
                                    hyperparameter_names=list(original_model.param_grid.keys()),
                                    plots={'pdp': xai_service_pb2.ExplanationsResponse(
                                                    explainability_type = 'hyperparameterExplanation',
                                                    explanation_method = 'pdp',
                                                    explainability_model = model_name,
                                                    plot_name = 'Partial Dependence Plot (PDP)',
                                                    plot_descr = "PD (Partial Dependence) Plots show how different hyperparameter values affect a model's accuracy, holding other hyperparameters constant, to illustrate hyperparameters impact.",
                                                    plot_type = 'LinePLot',
                                                    features = xai_service_pb2.Features(
                                                                feature1=list(param_grid.keys())[0], 
                                                                feature2=''),
                                                    xAxis = xai_service_pb2.Axis(
                                                                axis_name=f'{list(param_grid.keys())[0]}', 
                                                                axis_values=[str(value) for value in x[0]], 
                                                                axis_type='categorical' if isinstance(x[0][0], str) else 'numerical'
                                                    ),
                                                    yAxis = xai_service_pb2.Axis(
                                                                axis_name='PDP Values', 
                                                                axis_values=[str(value) for value in y[0]], 
                                                                axis_type='numerical'
                                                    ),
                                                    zAxis = xai_service_pb2.Axis(
                                                                axis_name='', 
                                                                axis_values='', 
                                                                axis_type=''                    
                                                )
                                                ),
                                            '2dpdp': xai_service_pb2.ExplanationsResponse(
                                                    explainability_type = 'hyperparameterExplanation',
                                                    explanation_method = '2dpdp',
                                                    explainability_model = model_name,
                                                    plot_name = '2D-Partial Dependece Plot (2D-PDP)',
                                                    plot_descr = "2D-PD plots visualize how the model's accuracy changes when two hyperparameters vary.",
                                                    plot_type = 'ContourPlot',
                                                    features = xai_service_pb2.Features(
                                                                feature1=list(param_grid.keys())[0], 
                                                                feature2=list(param_grid.keys())[1]),
                                                    xAxis = xai_service_pb2.Axis(
                                                                axis_name=f'{list(param_grid.keys())[1]}', 
                                                                axis_values=[str(value) for value in x2d], 
                                                                axis_type='categorical' if isinstance(x2d[0], str) else 'numerical'
                                                    ),
                                                    yAxis = xai_service_pb2.Axis(
                                                                axis_name=f'{list(param_grid.keys())[0]}', 
                                                                axis_values=[str(value) for value in y2d], 
                                                                axis_type='categorical' if isinstance(y2d[0], str) else 'numerical'
                                                    ),
                                                    zAxis = xai_service_pb2.Axis(
                                                                axis_name='', 
                                                                axis_values=[str(value) for value in z], 
                                                                axis_type='numerical'                    
                                                    )
                                                ),
                                            'ale': xai_service_pb2.ExplanationsResponse(
                                                    explainability_type = 'hyperparameterExplanation',
                                                    explanation_method = 'ale',
                                                    explainability_model = model_name,
                                                    plot_name = 'Accumulated Local Effects Plot (ALE)',
                                                    plot_descr = "ALE Plots illustrate the effect of a single hyperparameter on the accuracy of a machine learning model.",
                                                    features = xai_service_pb2.Features(
                                                                feature1=list(param_grid.keys())[0], 
                                                                feature2=''),
                                                    xAxis = xai_service_pb2.Axis(
                                                                axis_name=f'{list(param_grid.keys())[0]}', 
                                                                axis_values=[str(value) for value in ale_eff.index.tolist()], 
                                                                axis_type='categorical' if isinstance(ale_eff.index.tolist()[0], str) else 'numerical'
                                                    ),
                                                    yAxis = xai_service_pb2.Axis(
                                                                axis_name='ALE Values', 
                                                                axis_values=[str(value) for value in ale_eff.eff.tolist()], 
                                                                axis_type='numerical'
                                                    ),
                                                ),    
                                            },  
                            ),
                )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    xai_service_pb2_grpc.add_ExplanationsServicer_to_server(ExplainabilityExecutor(), server)
    #xai_service_pb2_grpc.add_InfluencesServicer_to_server(MyInfluencesService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()