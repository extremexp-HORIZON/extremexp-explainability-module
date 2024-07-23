import grpc
from concurrent import futures
import xai_service_pb2_grpc
import xai_service_pb2
from xai_service_pb2_grpc import ExplanationsServicer
import json
import numpy as np
from modules.lib_IF import preprocess_data
from concurrent import futures
from modules.lib_IF import *
# import torch.nn.functional as F
import json
from sklearn.inspection import partial_dependence
from modules.lib import *
from modules.ale import *
from modules.explainers import *
import dice_ml
from modules.ALE_generic import ale
import joblib
import io
from PyALE import ale
import ast 
from aix360.algorithms.protodash import ProtodashExplainer
import dill as pickle


class MyExplanationsService(ExplanationsServicer):

    def GetExplanation(self, request, context):
        print('Reading data')
        models = json.load(open("metadata/models.json"))
        data = json.load(open("metadata/datasets.json"))
        dataframe = pd.DataFrame()
        label = pd.DataFrame()

        #for request in request_iterator:
        explanation_type = request.explanation_type
        explanation_method = request.explanation_method

        if explanation_type == 'hyperparameterExplanation':

            if explanation_method == 'pdp':
                feature = request.feature1
                model_id = request.model

                if model_id == 'Ideko_model':
                    try:
                        with open(models[model_id]['original_model'], 'rb') as f:
                            original_model = pickle.load(f)
                    except FileNotFoundError:
                        print("Model does not exist. Load existing model.")
                else:
                    try:
                        with open(models[model_id]['original_model'], 'rb') as f:
                            original_model = joblib.load(f)
                    except FileNotFoundError:
                        print("Model does not exist. Load existing model.")

                param_grid = original_model.param_grid
                param_grid = transform_grid_plt(param_grid)
                try:
                    with open(models[model_id]['pdp_ale_surrogate_model'], 'rb') as f:
                        surrogate_model = joblib.load(f)
                except FileNotFoundError:
                    print("Surrogate model does not exist. Training new surrogate model") 
                    surrogate_model = proxy_model(param_grid,original_model,'accuracy','XGBoostRegressor')
                    joblib.dump(surrogate_model, models[model_id]['pdp_ale_surrogate_model'])                   

                x,y = ComputePDP(param_grid=param_grid, model=surrogate_model, feature=feature)
                if type(x[0][0]) == str:
                    axis_type='categorical' 
                else: axis_type = 'numerical'

                return xai_service_pb2.ExplanationsResponse(
                    explainability_type = explanation_type,
                    explanation_method = explanation_method,
                    explainability_model = model_id,
                    plot_name = 'Partial Dependence Plot (PDP)',
                    plot_descr = "PD (Partial Dependence) Plots show how different hyperparameter values affect a model's accuracy, holding other hyperparameters constant, to illustrate hyperparameters impact.",
                    plot_type = 'LinePlot',
                    features = xai_service_pb2.Features(
                                feature1=feature, 
                                feature2=''),
                    xAxis = xai_service_pb2.Axis(
                                axis_name=f'{feature}', 
                                axis_values=[str(value) for value in x[0]], 
                                axis_type=axis_type  
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
                )

            elif explanation_method == '2dpdp':

                feature1 = request.feature1
                feature2 = request.feature2
                model_id = request.model

                if model_id == 'Ideko_model':
                    try:
                        with open(models[model_id]['original_model'], 'rb') as f:
                            original_model = pickle.load(f)
                    except FileNotFoundError:
                        print("Model does not exist. Load existing model.")
                else:
                    try:
                        with open(models[model_id]['original_model'], 'rb') as f:
                            original_model = joblib.load(f)
                    except FileNotFoundError:
                        print("Model does not exist. Load existing model.")

                param_grid = original_model.param_grid
                param_grid = transform_grid_plt(param_grid)
                try:
                    with open(models[model_id]['pdp_ale_surrogate_model'], 'rb') as f:
                        surrogate_model = joblib.load(f)
                except FileNotFoundError:
                    print("Surrogate model does not exist. Training new surrogate model") 
                    surrogate_model = proxy_model(param_grid,original_model,'accuracy','XGBoostRegressor')
                    joblib.dump(surrogate_model, models[model_id]['pdp_ale_surrogate_model'])   

                x,y,z = ComputePDP2D(param_grid=param_grid, model=surrogate_model,feature1=feature1,feature2=feature2)

                return xai_service_pb2.ExplanationsResponse(
                    explainability_type = explanation_type,
                    explanation_method = explanation_method,
                    explainability_model = model_id,
                    plot_name = '2D-Partial Dependence Plot (2D-PDP)',
                    plot_descr = "2D-PD plots visualize how the model's accuracy changes when two hyperparameters vary.",
                    plot_type = 'ContourPlot',
                    features = xai_service_pb2.Features(
                                feature1=feature1, 
                                feature2=feature2),
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
                    )
                )

            elif explanation_method == 'ale':


                feature = request.feature1
                model_id = request.model
                if model_id == 'Ideko_model':
                    try:
                        with open(models[model_id]['original_model'], 'rb') as f:
                            original_model = pickle.load(f)
                    except FileNotFoundError:
                        print("Model does not exist. Load existing model.")
                else:
                    try:
                        with open(models[model_id]['original_model'], 'rb') as f:
                            original_model = joblib.load(f)
                    except FileNotFoundError:
                        print("Model does not exist. Load existing model.")

                param_grid = original_model.param_grid
                param_grid = transform_grid_plt(param_grid)
                try:
                    with open(models[model_id]['pdp_ale_surrogate_model'], 'rb') as f:
                        surrogate_model = joblib.load(f)
                except FileNotFoundError:
                    print("Surrogate model does not exist. Training new surrogate model") 
                    surrogate_model = proxy_model(param_grid,original_model,'accuracy','XGBoostRegressor')
                    joblib.dump(surrogate_model, models[model_id]['pdp_ale_surrogate_model'])  

                ale_eff = ComputeALE(param_grid=param_grid, model=surrogate_model, feature=feature)
                return xai_service_pb2.ExplanationsResponse(
                    explainability_type = explanation_type,
                    explanation_method = explanation_method,
                    explainability_model = model_id,
                    plot_name = 'Accumulated Local Effects Plot (ALE)',
                    plot_descr = "ALE Plots illustrate the effect of a single hyperparameter on the accuracy of a machine learning model.",
                    plot_type = 'LinePLot',
                    features = xai_service_pb2.Features(
                                feature1=feature, 
                                feature2=''),
                    xAxis = xai_service_pb2.Axis(
                                axis_name=f'{feature}', 
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
                    )
                )

            elif explanation_method == 'influenceFunctions':  
                # chunk_df = pd.read_parquet(io.BytesIO(request.train_data))  # Deserialize DataFrame chunk
                # dataframe = pd.concat([dataframe, chunk_df], ignore_index=True)


                # chunk_df_label = pd.read_parquet(io.BytesIO(request.train_labels))  # Deserialize DataFrame chunk
                # label = pd.concat([label, chunk_df_label], ignore_index=True)
                model_id = request.model    
                try:
                    with open(models[model_id]['original_model'], 'rb') as f:
                        original_model = joblib.load(f)
                except FileNotFoundError:
                    print("Model does not exist. Load existing model.")  
                num_influential = request.num_influential    

                train = pd.read_csv(data[model_id]['train'],index_col=0) 
                train_labels = pd.read_csv(data[model_id]['train_labels'],index_col=0) 
                test = pd.read_csv(data[model_id]['test'],index_col=0) 
                test_labels = pd.read_csv(data[model_id]['test_labels'],index_col=0) 
                cat_columns = train.select_dtypes(exclude=[np.number]).columns.tolist()
                numeric_columns = train.select_dtypes(exclude=['object']).columns.tolist()


                new_train = preprocess_data(data=train,label_encoded_features=[],one_hot_encoded_features=cat_columns,numerical_features=numeric_columns)
                new_test = preprocess_data(data=test,label_encoded_features=[],one_hot_encoded_features=cat_columns,numerical_features=numeric_columns)
                new_train.reset_index(drop=True,inplace=True)
                new_test.reset_index(drop=True,inplace=True)

                train_labels.reset_index(drop=True,inplace=True)
                test_labels.reset_index(drop=True,inplace=True)

                y = train_labels.loc[:1500].squeeze().to_numpy()
                yt = test_labels.iloc[[29,58,14955,14980]].squeeze().to_numpy()

                x = new_train.loc[:1500].values
                xt = new_test.iloc[[29,58,14955,14980]].values
                
                    #Compute influences 
                influences = compute_IF(model=original_model.best_estimator_.named_steps['Model'].module,loss=F.binary_cross_entropy,training_data = x, 
                                        test_data = xt, train_labels= y, test_labels= yt,
                                        influence_type='up', inversion_method='direct', hessian_regularization=0.5)
                
                show_influences(influences,5,new_train,train_labels)
                positive = show_pos_inf_instance(influences,num_influential,new_train,train_labels)
                negative = show_neg_inf_instance(influences,num_influential,new_train,train_labels)



                influences = influences.flatten().tolist()
                positive = positive.to_parquet(None)
                negative = negative.to_parquet(None)
                    # Create a response message
                response = xai_service_pb2.ExplanationsResponse(influences=influences,positive=positive,negative = negative)

                return response
            elif explanation_method == 'counterfactuals':  
                print("receiving")

                model_name = request.model
                model_id = request.model_id

                if model_name == 'Ideko_model':
                    try:
                        with open(models[model_name]['original_model'], 'rb') as f:
                            original_model = pickle.load(f)
                    except FileNotFoundError:
                        print("Model does not exist. Load existing model.")
                else:
                    query = request.query
                    
                    query = ast.literal_eval(query)
                    query = pd.DataFrame([query])
                    try:
                        with open(models[model_name]['original_model'], 'rb') as f:
                            original_model = joblib.load(f)
                    except FileNotFoundError:
                        print("Model does not exist. Load existing model.")


                    try:
                        with open(models[model_name]['all_models'], 'rb') as f:
                            trained_models = joblib.load(f)
                    except FileNotFoundError:
                        print("Model does not exist. Load existing model.")

                    model = trained_models[model_id]

                # try:
                #     with open(models[model_name]['cfs_surrogate_model'], 'rb') as f:
                #         surrogate_model = joblib.load(f)
                #         proxy_dataset = pd.read_csv(models[model_name]['cfs_surrogate_dataset'],index_col=0)
                # except FileNotFoundError:
                #     print("Surrogate model does not exist. Training new surrogate model") 
                if model_name == 'I2Cat_Phising_model':
                    train = pd.read_csv(data[model_name]['train'],index_col=0) 
                    train_labels = pd.read_csv(data[model_name]['train_labels'],index_col=0) 
                    print('Creating Proxy Dataset and Model')
                    surrogate_model , proxy_dataset = instance_proxy(train,train_labels,original_model, query.loc[0],original_model.param_grid)
                    #joblib.dump(surrogate_model, models[model_name]['cfs_surrogate_model'])  
                    #proxy_dataset.to_csv(models[model_name]['cfs_surrogate_dataset'])
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
                    query = pd.DataFrame(data = {'batch_size':params['batch_size'],'epochs':params['epochs'],'model__activation_function': params['model__activation_function'],'model__units': [params['model__units']]})
                    query[categorical] = query[categorical].astype(str)

                d = dice_ml.Data(dataframe=proxy_dataset, 
                    continuous_features=proxy_dataset.drop(columns='BinaryLabel').select_dtypes(include='number').columns.tolist()
                    , outcome_name='BinaryLabel')
                
                # Using sklearn backend
                m = dice_ml.Model(model=surrogate_model, backend="sklearn")
                # Using method=random for generating CFs
                exp = dice_ml.Dice(d, m, method="random")
                if model_name == 'Ideko_model':
                    e1 = exp.generate_counterfactuals(query, total_CFs=5, desired_class=2,sample_size=5000)
                else:
                    e1 = exp.generate_counterfactuals(query, total_CFs=5, desired_class="opposite",sample_size=5000)
                #e1.visualize_as_dataframe(show_only_changes=True)
                cfs = e1.cf_examples_list[0].final_cfs_df
                dtypes_dict = proxy_dataset.drop(columns='BinaryLabel').dtypes.to_dict()
                for col, dtype in dtypes_dict.items():
                    cfs[col] = cfs[col].astype(dtype)
                scaled_query, scaled_cfs = min_max_scale(proxy_dataset=proxy_dataset,factual=query.copy(deep=True),counterfactuals=cfs.copy(deep=True))
                cfs['Cost'] = cf_difference(scaled_query, scaled_cfs)
                cfs = cfs.sort_values(by='Cost')
                cfs['Type'] = 'Counterfactual'
                query['BinaryLabel'] = 1
                query['Cost'] = '-'
                query['Type'] = 'Factual'
                # for col in query.columns:
                #     cfs[col] = cfs[col].apply(lambda x: '-' if x == query.iloc[0][col] else x)
                cfs = pd.concat([query,cfs])

                
                return xai_service_pb2.ExplanationsResponse(
                    explainability_type = explanation_type,
                    explanation_method = explanation_method,
                    explainability_model = model_name,
                    plot_name = 'Counterfactual Explanations',
                    plot_descr = "Counterfactual Explanations identify the minimal changes on hyperparameter values in order to correctly classify a given missclassified instance.",
                    plot_type = 'Table',
                    table_contents = {col: xai_service_pb2.TableContents(index=i+1,values=cfs[col].astype(str).tolist()) for i,col in enumerate(cfs.columns)}
                )
        elif explanation_type == 'featureExplanation':

            if explanation_method == 'pdp' :
                print('Receiving')

                model_name = request.model
                model_id = request.model_id

                try:
                    with open(models[model_name]['original_model'], 'rb') as f:
                        original_model = joblib.load(f)
                except FileNotFoundError:
                    print("Model does not exist. Load existing model.")

                try:
                    with open(models[model_name]['all_models'], 'rb') as f:
                        trained_models = joblib.load(f)
                except FileNotFoundError:
                    print("Model does not exist. Load existing model.")

                model = trained_models[model_id]

                dataframe = pd.read_csv(data[model_name]['train'],index_col=0) 
                features = request.feature1
                numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
                print(features)
                print([dataframe.columns.tolist().index(features)])
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
                    explanation_method = explanation_method,
                    explainability_model = model_name,
                    plot_name = 'Partial Dependence Plot (PDP)',
                    plot_descr = "PD (Partial Dependence) Plots show how a feature affects a model's predictions, holding other features constant, to illustrate feature impact.",
                    plot_type = 'LinePlot',
                    features = xai_service_pb2.Features(
                                feature1=features, 
                                feature2=''),
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
    
            elif explanation_method == '2D_PDPlots':
                model_name = request.model
                model_id = request.model_id

                try:
                    with open(models[model_name]['original_model'], 'rb') as f:
                        original_model = joblib.load(f)
                except FileNotFoundError:
                    print("Model does not exist. Load existing model.")

                try:
                    with open(models[model_name]['all_models'], 'rb') as f:
                        trained_models = joblib.load(f)
                except FileNotFoundError:
                    print("Model does not exist. Load existing model.")

                model = trained_models[model_id]

                dataframe = pd.read_csv(data[model_name]['train'],index_col=0)                        
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
                    explanation_method = explanation_method,
                    explainability_model = model_name,
                    plot_name = '2D-Partial Dependence Plot (2D-PDP)',
                    plot_descr = "2D-PD plots visualize how the model's accuracy changes when two hyperparameters vary.",
                    plot_type = 'ContourPlot',
                    features = xai_service_pb2.Features(
                                feature1=feature1, 
                                feature2=feature2),
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
                    )
                )
            
            elif explanation_method == 'counterfactuals':
                model_name = request.model
                model_id = request.model_id
                query = request.query
                query = ast.literal_eval(query)
                query = pd.DataFrame([query])

                query = query.drop(columns=['id','label'])
                try:
                    with open(models[model_name]['original_model'], 'rb') as f:
                        original_model = joblib.load(f)
                except FileNotFoundError:
                    print("Model does not exist. Load existing model.")

                try:
                    with open(models[model_name]['all_models'], 'rb') as f:
                        trained_models = joblib.load(f)
                except FileNotFoundError:
                    print("Model does not exist. Load existing model.")

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
                display(cfs)

                return xai_service_pb2.ExplanationsResponse(
                    explainability_type = explanation_type,
                    explanation_method = explanation_method,
                    explainability_model = model_name,
                    plot_name = 'Counterfactual Explanations',
                    plot_descr = "Counterfactual Explanations identify the minimal changes needed to alter a machine learning model's prediction for a given instance.",
                    plot_type = 'Table',
                    table_contents = {col: xai_service_pb2.TableContents(index=i+1,values=cfs[col].astype(str).tolist()) for i,col in enumerate(cfs.columns)}
                )
            
            elif explanation_method == 'ale':
                model_name = request.model
                model_id = request.model_id
                try:
                    with open(models[model_name]['original_model'], 'rb') as f:
                        original_model = joblib.load(f)
                except FileNotFoundError:
                    print("Model does not exist. Load existing model.")

                try:
                    with open(models[model_name]['all_models'], 'rb') as f:
                        trained_models = joblib.load(f)
                except FileNotFoundError:
                    print("Model does not exist. Load existing model.")

                model = trained_models[model_id]

                dataframe = pd.read_csv(data[model_name]['train'],index_col=0) 
                features = request.feature1

                if dataframe[features].dtype in ['int','float']:
                    ale_eff = ale(X=dataframe, model=model, feature=[features],plot=False, grid_size=50, include_CI=True, C=0.95)
                else:
                    ale_eff = ale(X=dataframe, model=model, feature=[features],plot=False, grid_size=50, predictors=dataframe.columns.tolist(), include_CI=True, C=0.95)

                return xai_service_pb2.ExplanationsResponse(
                    explainability_type = explanation_type,
                    explanation_method = explanation_method,
                    explainability_model = model_name,
                    plot_name = 'Accumulated Local Effects Plot (ALE)',
                    plot_descr = "ALE plots illustrate the effect of a single feature on the predicted outcome of a machine learning model.",
                    plot_type = 'LinePLot',
                    features = xai_service_pb2.Features(
                                feature1=features, 
                                feature2=''),
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
                    )
                )
            elif explanation_method == 'prototypes':
                model_name = request.model
                model_id = request.model_id
                query = request.query
                query = ast.literal_eval(query)
                query = pd.DataFrame([query])
                try:
                    with open(models[model_name]['original_model'], 'rb') as f:
                        original_model = joblib.load(f)
                except FileNotFoundError:
                    print("Model does not exist. Load existing model.")

                try:
                    with open(models[model_name]['all_models'], 'rb') as f:
                        trained_models = joblib.load(f)
                except FileNotFoundError:
                    print("Model does not exist. Load existing model.")

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
                    explanation_method = explanation_method,
                    explainability_model = model_name,
                    plot_name = 'Prototypes',
                    plot_descr = "Prototypes are prototypical examples that capture the underlying distribution of a dataset. It also weights each prototype to quantify how well it represents the data.",
                    plot_type = 'Table',
                    table_contents = table_contents
                )
    def Initialization(self, request, context):
        models = json.load(open("metadata/models.json"))
        data = json.load(open("metadata/datasets.json"))
        model_id = request.model_name
        # Load trained model if exists
        try:
            with open(models[model_id]['original_model'], 'rb') as f:
                original_model = joblib.load(f)
        except FileNotFoundError:
            print("Model does not exist. Load existing model.")

        # Load Data
        train = pd.read_csv(data[model_id]['train'],index_col=0) 
        train_labels = pd.read_csv(data[model_id]['train_labels'],index_col=0) 
        test = pd.read_csv(data[model_id]['test'],index_col=0) 
        test_labels = pd.read_csv(data[model_id]['test_labels'],index_col=0) 
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
            with open(models[model_id]['pdp_ale_surrogate_model'], 'rb') as f:
                pdp_ale_surrogate_model = joblib.load(f)
        except FileNotFoundError:
            print("Surrogate model does not exist. Training new surrogate model") 
            pdp_ale_surrogate_model = proxy_model(param_grid,original_model,'accuracy','XGBoostRegressor')
            joblib.dump(pdp_ale_surrogate_model, models[model_id]['pdp_ale_surrogate_model'])  

        # Load surrogate model for CF if exists
        try:
            with open(models[model_id]['cfs_surrogate_model'], 'rb') as f:
                cfs_surrogate_model = joblib.load(f)
                proxy_dataset = pd.read_csv(models[model_id]['cfs_surrogate_dataset'],index_col=0)
        except FileNotFoundError:
            print("Surrogate model does not exist. Training new surrogate model") 
            train = pd.read_csv(data[model_id]['train'],index_col=0) 
            train_labels = pd.read_csv(data[model_id]['train_labels'],index_col=0) 
            cfs_surrogate_model , proxy_dataset = instance_proxy(train,train_labels,original_model, query,original_model.param_grid)
            joblib.dump(cfs_surrogate_model, models[model_id]['cfs_surrogate_model'])  
            proxy_dataset.to_csv(models[model_id]['cfs_surrogate_dataset'])

        # ---------------------- Run Explainability Methods for Pipeline -----------------------------------------------

        #PDP
        x,y = ComputePDP(param_grid=param_grid, model=pdp_ale_surrogate_model, feature=list(param_grid.keys())[0])
        # 2D PDP
        x2d,y2d,z = ComputePDP2D(param_grid=param_grid, model=pdp_ale_surrogate_model,feature1=list(param_grid.keys())[0],feature2=list(param_grid.keys())[1])
        # ALE
        ale_eff_hp = ComputeALE(param_grid=param_grid, model=pdp_ale_surrogate_model, feature=list(param_grid.keys())[0])

        #Counterfactuals
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
        query = pd.DataFrame.from_dict(original_model.best_params_,orient='index').T
        query[categorical] = query[categorical].astype(str)

        # d = dice_ml.Data(dataframe=proxy_dataset, 
        #     continuous_features=proxy_dataset.drop(columns='BinaryLabel').select_dtypes(include='number').columns.tolist()
        #     , outcome_name='BinaryLabel')
        
        # # Using sklearn backend
        # m = dice_ml.Model(model=cfs_surrogate_model, backend="sklearn")
        # # Using method=random for generating CFs
        # exp = dice_ml.Dice(d, m, method="random")
        # e1 = exp.generate_counterfactuals(query, total_CFs=5, desired_class="opposite",sample_size=5000)

        # cfs = e1.cf_examples_list[0].final_cfs_df
        # dtypes_dict = proxy_dataset.drop(columns='BinaryLabel').dtypes.to_dict()
        # for col, dtype in dtypes_dict.items():
        #     cfs[col] = cfs[col].astype(dtype)

        # scaled_query, scaled_cfs = min_max_scale(proxy_dataset=proxy_dataset,factual=query.copy(deep=True),counterfactuals=cfs.copy(deep=True))
        # cfs['Cost'] = cf_difference(scaled_query, scaled_cfs)
        # cfs = cfs.sort_values(by='Cost')
        # query['BinaryLabel'] = 1
        # query['Cost'] = '-'
        # cfs['Type'] = 'Counterfactual'
        # query['Type'] = 'Factual'
        # # for col in query.columns:
        # #     cfs[col] = cfs[col].apply(lambda x: '-' if x == query.iloc[0][col] else x)
        # cfs = pd.concat([query,cfs])

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

        # CounterFactuals

        # d = dice_ml.Data(dataframe=dataframe, 
        # continuous_features=train.select_dtypes(include='number').columns.tolist()
        # , outcome_name='label')

        # # Using sklearn backend
        # m = dice_ml.Model(model=original_model, backend="sklearn")
        # # Using method=random for generating CFs
        # exp = dice_ml.Dice(d, m, method="random")
        # e1 = exp.generate_counterfactuals(missclassified_instances.reset_index(drop=True).loc[0].to_frame().T.drop(columns=['Predicted','label','Label']), total_CFs=5, desired_class="opposite",sample_size=5000)
        # e1.visualize_as_dataframe(show_only_changes=True)
        # cfs_feat = e1.cf_examples_list[0].final_cfs_df
        # # cfs_feat = pd.concat([missclassified_instances.reset_index(drop=True).loc[0].to_frame().T.drop(columns=['label','Label']),cfs_feat])
        # query_feat = missclassified_instances.reset_index(drop=True).loc[0].to_frame().T
        # query_feat = query_feat.drop(columns=['label','Label'])
        # query_feat.rename(columns={'Predicted':'label'},inplace=True)
        # # for col in query_feat.columns:
        # #     cfs_feat[col] = cfs_feat[col].apply(lambda x: '-' if x == query_feat.iloc[0][col] else x)
        # cfs_feat['Type'] = 'Counterfactual'
        # query_feat['Type'] = 'Factual'
        
        # #cfs = cfs.to_parquet(None)
        # cfs_feat = pd.concat([query_feat,cfs_feat])

        return xai_service_pb2.InitializationResponse(


            feature_explanation = xai_service_pb2.Feature_Explanation(
                                    feature_names=train.columns.tolist(),
                                    plots={'pdp': xai_service_pb2.ExplanationsResponse(
                                                    explainability_type = 'featureExplanation',
                                                    explanation_method = 'pdp',
                                                    explainability_model = model_id,
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
                                                    explainability_model = model_id,
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
                                # tables = {'counterfactuals': xai_service_pb2.ExplanationsResponse(
                                #             explainability_type = 'featureExplanation',
                                #             explanation_method = 'counterfactuals',
                                #             explainability_model = model_id,
                                #             plot_name = 'Counterfactual Explanations',
                                #             plot_descr = "Counterfactual Explanations identify the minimal changes needed to alter a machine learning model's prediction for a given instance.",
                                #             plot_type = 'Table',
                                #             table_contents = {col: xai_service_pb2.TableContents(index=i+1,values=cfs_feat[col].astype(str).tolist()) for i,col in enumerate(cfs_feat.columns)}
                                #         )}     
                            ),

            hyperparameter_explanation = xai_service_pb2.Hyperparameter_Explanation(
                                    hyperparameter_names=list(original_model.param_grid.keys()),
                                    plots={'pdp': xai_service_pb2.ExplanationsResponse(
                                                    explainability_type = 'hyperparameterExplanation',
                                                    explanation_method = 'pdp',
                                                    explainability_model = model_id,
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
                                                    explainability_model = model_id,
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
                                                    explainability_model = model_id,
                                                    plot_name = 'Accumulated Local Effects Plot (ALE)',
                                                    plot_descr = "ALE Plots illustrate the effect of a single hyperparameter on the accuracy of a machine learning model.",
                                                    features = xai_service_pb2.Features(
                                                                feature1=list(param_grid.keys())[0], 
                                                                feature2=''),
                                                    xAxis = xai_service_pb2.Axis(
                                                                axis_name=f'{list(param_grid.keys())[0]}', 
                                                                axis_values=[str(value) for value in ale_eff_hp.index.tolist()], 
                                                                axis_type='categorical' if isinstance(ale_eff_hp.index.tolist()[0], str) else 'numerical'
                                                    ),
                                                    yAxis = xai_service_pb2.Axis(
                                                                axis_name='ALE Values', 
                                                                axis_values=[str(value) for value in ale_eff_hp.eff.tolist()], 
                                                                axis_type='numerical'
                                                    ),
                                                ),    
                                            },
                                # tables = {'counterfactuals': xai_service_pb2.ExplanationsResponse(
                                #             explainability_type = 'hyperparameterExplanation',
                                #             explanation_method = 'counterfactuals',
                                #             explainability_model = model_id,
                                #             plot_name = 'Counterfactual Explanations',
                                #             plot_descr = "Counterfactual Explanations identify the minimal changes on hyperparameter values in order to correctly classify a given missclassified instance.",
                                #             plot_type = 'Table',
                                #             table_contents = {col: xai_service_pb2.TableContents(index=i+1,values=cfs[col].astype(str).tolist()) for i,col in enumerate(cfs.columns)}
                                #         )}     
                            ),
                )



    def ModelAnalysisTask(self, request, context):
        models = json.load(open("metadata/models.json"))
        data = json.load(open("metadata/datasets.json"))
        model_name = request.model_name
        model_id = request.model_id
        # Load trained model if exists
        try:
            with open(models[model_name]['original_model'], 'rb') as f:
                original_model = joblib.load(f)
        except FileNotFoundError:
            print("Model does not exist. Load existing model.")

        try:
            with open(models[model_name]['all_models'], 'rb') as f:
                trained_models = joblib.load(f)
        except FileNotFoundError:
            print("Model does not exist. Load existing model.")

        model = trained_models[model_id]

        # Load Data
        train = pd.read_csv(data[model_name]['train'],index_col=0) 
        train_labels = pd.read_csv(data[model_name]['train_labels'],index_col=0) 
        test = pd.read_csv(data[model_name]['test'],index_col=0) 
        test_labels = pd.read_csv(data[model_name]['test_labels'],index_col=0) 

        test['label'] = test_labels
        dataframe = pd.concat([train.reset_index(drop=True), train_labels.reset_index(drop=True)], axis = 1)

        predictions = original_model.predict(test)
        test['Predicted'] = predictions
        test['Label'] = (test['label'] != test['Predicted']).astype(int)


        param_grid = original_model.param_grid
        param_grid = transform_grid_plt(param_grid)
         

        # ---------------------- Run Explainability Methods for Model -----------------------------------------------

        # PD Plots
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        features = train.columns.tolist()[0]
        numeric_features = train.select_dtypes(include=numerics).columns.tolist()
        categorical_features = train.columns.drop(numeric_features)

        pdp = partial_dependence(model, train, features = [train.columns.tolist().index(features)],
                                feature_names=train.columns.tolist(),categorical_features=categorical_features)

        pdp_grid = [value.tolist() for value in pdp['grid_values']][0]
        pdp_vals = [value.tolist() for value in pdp['average']][0]

        #ALE Plots
        if train[features].dtype in ['int','float']:
            ale_eff_feat = ale(X=train, model=model, feature=[features],plot=False, grid_size=50, include_CI=True, C=0.95)
        else:
            ale_eff_feat = ale(X=train, model=model, feature=[features],plot=False, grid_size=50, predictors=train.columns.tolist(), include_CI=True, C=0.95)

        return xai_service_pb2.ModelAnalysisTaskResponse(

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
                )
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    xai_service_pb2_grpc.add_ExplanationsServicer_to_server(MyExplanationsService(), server)
    #xai_service_pb2_grpc.add_InfluencesServicer_to_server(MyInfluencesService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
