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
import torch.nn.functional as F
import json
from sklearn.inspection import partial_dependence
from modules.lib import *
from modules.ale import *
from modules.explainers import *
import dice_ml
from modules.ALE_generic import ale
import joblib
import io


class MyExplanationsService(ExplanationsServicer):

    def GetExplanation(self, request_iterator, context):
        print('Reading data')
        models = json.load(open("metadata/models.json"))
        data = json.load(open("metadata/datasets.json"))
        dataframe = pd.DataFrame()
        label = pd.DataFrame()

        for request in request_iterator:
            explanation_type = request.explanation_type
            explanation_method = request.explanation_method

            if explanation_type == 'Pipeline':

                if explanation_method == 'PDPlots':
                    feature = request.feature1
                    model_id = request.model
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

                    return xai_service_pb2.ExplanationsResponse(
                        pdp_hp_values=json.dumps(x),
                        pdp_values=json.dumps(y)
                    )

                elif explanation_method == '2D_PDPlots':

                    feature1 = request.feature1
                    feature2 = request.feature2
                    model_id = request.model
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
                        pdp2d_xi=json.dumps(x),
                        pdp2d_yi=json.dumps(y),
                        pdp2d_zi=json.dumps(z)
                    )

                elif explanation_method == 'ALEPlots':


                    feature = request.feature1
                    model_id = request.model
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

                    d = ComputeALE(param_grid=param_grid, model=surrogate_model, feature=feature)

                    return xai_service_pb2.ExplanationsResponse(
                        ale_data=d  # Replace with actual data
                    )

                elif explanation_method == 'InfluenceFunctions':  
                    print("receiving")
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

                    print('Data received')
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
                elif explanation_method == 'CounterfactualExplanations':  
                    print("receiving")

                    model_id = request.model
                    print(model_id)
                    try:
                        with open(models[model_id]['original_model'], 'rb') as f:
                            original_model = joblib.load(f)
                    except FileNotFoundError:
                        print("Model does not exist. Load existing model.")

                    try:
                        with open(models[model_id]['cfs_surrogate_model'], 'rb') as f:
                            proxy_model = joblib.load(f)
                            proxy_dataset = pd.read_csv(models[model_id]['cfs_surrogate_dataset'],index_col=0)
                    except FileNotFoundError:
                        print("Surrogate model does not exist. Training new surrogate model") 
                        train = pd.read_csv(data[model_id]['train'],index_col=0) 
                        train_labels = pd.read_csv(data[model_id]['train_labels'],index_col=0) 
                        proxy_model , proxy_dataset = instance_proxy(train,train_labels,original_model, query,original_model.param_grid)
                        joblib.dump(surrogate_model, models[model_id]['cfs_surrogate_model'])  
                        proxy_dataset.to_csv(models[model_id]['cfs_surrogate_dataset'])

                    if model_id == 'I2Cat_Phising_model':
                        query = pd.DataFrame.from_dict(original_model.best_params_,orient='index').T
                        query['preprocessor__num__scaler'] = query['preprocessor__num__scaler'].astype(str)
                        query['Model__learning_rate'] = query['Model__learning_rate'].astype(str)
                        proxy_dataset['Model__learning_rate'] = proxy_dataset['Model__learning_rate'].astype(str)
                    elif model_id == 'UNSW_NB15_model':
                        query = pd.DataFrame.from_dict(original_model.best_params_,orient='index').T
                        query['preprocessor__num__scaler'] = query['preprocessor__num__scaler'].astype(str)
                        query['Model__optimizer'] = query['Model__optimizer'].astype(str)
                        query['Model__lr'] = query['Model__lr'].astype(str)

                        query['Model__batch_size'] = query['Model__batch_size'].astype(np.int64)
                        proxy_dataset['Model__lr'] = proxy_dataset['Model__lr'].astype(str)

                    d = dice_ml.Data(dataframe=proxy_dataset, 
                        continuous_features=proxy_dataset.drop(columns='BinaryLabel').select_dtypes(include='number').columns.tolist()
                        , outcome_name='BinaryLabel')
                    
                    # Using sklearn backend
                    m = dice_ml.Model(model=proxy_model, backend="sklearn")
                    # Using method=random for generating CFs
                    exp = dice_ml.Dice(d, m, method="random")
                    e1 = exp.generate_counterfactuals(query, total_CFs=5, desired_class="opposite",sample_size=5000)
                    #e1.visualize_as_dataframe(show_only_changes=True)
                    cfs = e1.cf_examples_list[0].final_cfs_df
                    dtypes_dict = proxy_dataset.drop(columns='BinaryLabel').dtypes.to_dict()
                    for col, dtype in dtypes_dict.items():
                        cfs[col] = cfs[col].astype(dtype)
                    scaled_query, scaled_cfs = min_max_scale(proxy_dataset=proxy_dataset,factual=query.copy(deep=True),counterfactuals=cfs.copy(deep=True))
                    cfs['Cost'] = cf_difference(scaled_query, scaled_cfs)
                    cfs = cfs.sort_values(by='Cost')
                    cfs = cfs.to_parquet(None)

                    return xai_service_pb2.ExplanationsResponse(
                        cfs=cfs
                    )
            elif explanation_type == 'Model':
                # if explanation_method == 'PDPlots' or explanation_method == '2D_PDPlots' or explanation_method == 'CounterfactualExplanations' or explanation_method == 'ALEPlots':
                #     chunk_df = pd.read_parquet(io.BytesIO(request.train_data))
                #     dataframe = pd.concat([dataframe, chunk_df], ignore_index=True)


                if explanation_method == 'PDPlots' :
                    print('Receiving')

                    model_id = request.model
                    try:
                        with open(models[model_id]['original_model'], 'rb') as f:
                            original_model = joblib.load(f)
                    except FileNotFoundError:
                        print("Model does not exist. Load existing model.")

                    dataframe = pd.read_csv(data[model_id]['train'],index_col=0) 
                    features = request.feature1
                    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
                    print(features)
                    print([dataframe.columns.tolist().index(features)])
                    numeric_features = dataframe.select_dtypes(include=numerics).columns.tolist()
                    categorical_features = dataframe.columns.drop(numeric_features)

                    pdp = partial_dependence(original_model, dataframe, features = [dataframe.columns.tolist().index(features)],
                                            feature_names=dataframe.columns.tolist(),categorical_features=categorical_features)
                    
                    return xai_service_pb2.ExplanationsResponse(
                                pdp_vals=json.dumps([value.tolist() for value in pdp['grid_values']]),
                                pdp_effect=json.dumps(pdp['average'].tolist())
                            )
        
                elif explanation_method == '2D_PDPlots':
                    model_id = request.model
                    try:
                        with open(models[model_id]['original_model'], 'rb') as f:
                            original_model = joblib.load(f)
                    except FileNotFoundError:
                        print("Model does not exist. Load existing model.")

                    dataframe = pd.read_csv(data[model_id]['train'],index_col=0)                        
                    feature1 = request.feature1
                    feature2 = request.feature2
                    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

                    numeric_features = dataframe.select_dtypes(include=numerics).columns.tolist()
                    categorical_features = dataframe.columns.drop(numeric_features)

                    pdp = partial_dependence(original_model, dataframe, features = [(dataframe.columns.tolist().index(feature1),dataframe.columns.tolist().index(feature2))],
                                            feature_names=dataframe.columns.tolist(),categorical_features=categorical_features)
                    
                    return xai_service_pb2.ExplanationsResponse(
                                pdp_vals=json.dumps([value.tolist() for value in pdp['grid_values']]),
                                pdp_effect=json.dumps(pdp['average'].tolist())
                    )
                
                elif explanation_method == 'CounterfactualExplanations':
                    model_id = request.model
                    try:
                        with open(models[model_id]['original_model'], 'rb') as f:
                            original_model = joblib.load(f)
                    except FileNotFoundError:
                        print("Model does not exist. Load existing model.")
                    query = pd.read_parquet(io.BytesIO(request.query))
                    target = request.target
                    train = pd.read_csv(data[model_id]['train'],index_col=0)  
                    train_labels = pd.read_csv(data[model_id]['train_labels'],index_col=0)  
                    dataframe = pd.concat([train.reset_index(drop=True), train_labels.reset_index(drop=True)], axis = 1)
                    d = dice_ml.Data(dataframe=dataframe, 
                        continuous_features=dataframe.drop(columns=target).select_dtypes(include='number').columns.tolist()
                        , outcome_name=target)
            
                    # Using sklearn backend
                    m = dice_ml.Model(model=original_model, backend="sklearn")
                    # Using method=random for generating CFs
                    exp = dice_ml.Dice(d, m, method="random")
                    e1 = exp.generate_counterfactuals(query, total_CFs=5, desired_class="opposite",sample_size=5000)
                    e1.visualize_as_dataframe(show_only_changes=True)
                    cfs = e1.cf_examples_list[0].final_cfs_df
                    display(cfs)
                    cfs = cfs.to_parquet(None)

                    return xai_service_pb2.ExplanationsResponse(
                        cfs=cfs
                    )
                
                elif explanation_method == 'ALEPlots':
                    model_id = request.model
                    try:
                        with open(models[model_id]['original_model'], 'rb') as f:
                            original_model = joblib.load(f)
                    except FileNotFoundError:
                        print("Model does not exist. Load existing model.")

                    dataframe = pd.read_csv(data[model_id]['train'],index_col=0) 
                    features = request.feature1
                    if dataframe[features].dtype in ['int','float']:
                        ale_eff = ale(X=dataframe, model=original_model, feature=[features],plot=False, grid_size=50, include_CI=True, C=0.95)
                    else:
                        ale_eff = ale(X=dataframe, model=original_model, feature=[features],plot=False, grid_size=50, predictors=dataframe.columns.tolist(), include_CI=True, C=0.95)
                    dataframes_list = []
                    dataframes_list.append(ale_eff)
                    d = json.dumps([df.to_json(orient='split') for df in dataframes_list])

                    return xai_service_pb2.ExplanationsResponse(
                        ale_data=d
                    )


            # elif explanation_method == 'InfluenceFunctions' and explanation_type == 'Pipeline':
            #     model = torch.load(io.BytesIO(request.model), map_location=torch.device('cpu'))             
            #     test_data =  pd.read_parquet(io.BytesIO(request.test_data))        
            #     test_labels =  pd.read_parquet(io.BytesIO(request.test_labels))
            #     num_influential = request.num_influential
            #     print('Data received')
            #     print(len(dataframe))
            #     y = label.squeeze().to_numpy()
            #     yt = test_labels.squeeze().to_numpy()

            #     x = dataframe.values
            #     xt = test_data.values
                
            #         #Compute influences 
            #     influences = compute_IF(model=model,loss=F.binary_cross_entropy,training_data = x, 
            #                             test_data = xt, train_labels= y, test_labels= yt,
            #                             influence_type='up', inversion_method='direct', hessian_regularization=0.5)
                
            #     show_influences(influences,5,dataframe,label)
            #     positive = show_pos_inf_instance(influences,num_influential,dataframe,label)
            #     negative = show_neg_inf_instance(influences,num_influential,dataframe,label)



            #     influences = influences.flatten().tolist()
            #     positive = positive.to_parquet(None)
            #     negative = negative.to_parquet(None)
            #         # Create a response message
            #     response = xai_service_pb2.ExplanationsResponse(influences=influences,positive=positive,negative = negative)

            #     return response
        
        # elif explanation_method == 'CounterfactualExplanations' and explanation_method == 'Pipeline':
        #         query = pd.read_parquet(io.BytesIO(request.query))
        #         model_id = request.model
        #         try:
        #             with open(models[model_id]['original_model'], 'rb') as f:
        #                 original_model = joblib.load(f)
        #         except FileNotFoundError:
        #             print("Model does not exist. Load existing model.")

        #         try:
        #             with open(models[model_id]['cfs_surrogate_model'], 'rb') as f:
        #                 proxy_model = joblib.load(f)
        #                 proxy_dataset = pd.read_csv(models[model_id]['cfs_surrogate_dataset'])
        #         except FileNotFoundError:
        #             print("Surrogate model does not exist. Training new surrogate model") 
        #             proxy_model , proxy_dataset = instance_proxy(dataframe,label,original_model, query,original_model.param_grid)
        #             joblib.dump(surrogate_model, models[model_id]['cfs_surrogate_model'])  
        #             proxy_dataset.to_csv(models[model_id]['cfs_surrogate_dataset'])


        #         proxy_dataset.drop(columns='Unnamed: 0',inplace=True)
        #         query = pd.DataFrame.from_dict(original_model.best_params_,orient='index').T
        #         query['preprocessor__num__scaler'] = query['preprocessor__num__scaler'].astype(str)
        #         query['Model__learning_rate'] = query['Model__learning_rate'].astype(str)
        #         proxy_dataset['Model__learning_rate'] = proxy_dataset['Model__learning_rate'].astype(str)

        #         d = dice_ml.Data(dataframe=proxy_dataset, 
        #             continuous_features=proxy_dataset.drop(columns='BinaryLabel').select_dtypes(include='number').columns.tolist()
        #             , outcome_name='BinaryLabel')
                
        #         # Using sklearn backend
        #         m = dice_ml.Model(model=model, backend="sklearn")
        #         # Using method=random for generating CFs
        #         exp = dice_ml.Dice(d, m, method="random")
        #         e1 = exp.generate_counterfactuals(query, total_CFs=5, desired_class="opposite",sample_size=5000)
        #         e1.visualize_as_dataframe(show_only_changes=True)
        #         cfs = e1.cf_examples_list[0].final_cfs_df
        #         display(cfs)
        #         cfs = cfs.to_parquet(None)

        #         return xai_service_pb2.ExplanationsResponse(
        #             cfs=cfs
        #         )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    xai_service_pb2_grpc.add_ExplanationsServicer_to_server(MyExplanationsService(), server)
    #xai_service_pb2_grpc.add_InfluencesServicer_to_server(MyInfluencesService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
