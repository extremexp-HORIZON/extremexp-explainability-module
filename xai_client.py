import json
import grpc
import xai_service_pb2
import xai_service_pb2_grpc
import io
import torch
from modules.lib import transform_grid_plt

class Client():
    def __init__(self):
        self.channel = grpc.insecure_channel('localhost:50051')
        self.stub = xai_service_pb2_grpc.ExplanationsStub(self.channel)

    def dummy_stream(self,explanation_type,explanation_method,param_grid=None,model=None,feature1=None,feature2=None,proxy_dataset=None,query=None,num_influential=None,target=None):
        for i in range(0, 1):
            if explanation_type == 'Pipeline':
                if explanation_method == '2D_PDPlots':
                    yield xai_service_pb2.ExplanationsRequest(
                        explanation_type=explanation_type,
                        explanation_method = explanation_method,
                        model=model,
                        feature1 = feature1,
                        feature2 = feature2
                    )
                    
                elif explanation_method == 'CounterfactualExplanations':
                    yield xai_service_pb2.ExplanationsRequest(
                        explanation_type=explanation_type,
                        explanation_method = explanation_method,
                        model=model
                    )
                elif explanation_method == 'InfluenceFunctions':
                    try:
                        yield xai_service_pb2.ExplanationsRequest(
                            explanation_type = explanation_type,
                            explanation_method = explanation_method,
                            model=model,
                            num_influential = num_influential,
                        )
                    except:
                        print("error sending data")
                    print("Stopped Sending")
                else:
                    yield xai_service_pb2.ExplanationsRequest(
                        explanation_type=explanation_type,
                        explanation_method = explanation_method,
                        model=model,
                        feature1=feature1
                    ) 
                
            elif explanation_type == 'Model':
                if explanation_method == 'PDPlots' or explanation_method=='ALEPlots':
                    try:
                        yield xai_service_pb2.ExplanationsRequest(
                            explanation_type = explanation_type,
                            explanation_method = explanation_method,
                            model=model,
                            feature1 = feature1
                        )
                    except:
                        print("error sending data")
                    print("Stopped Sending")
                elif explanation_method == '2D_PDPlots':
                    try:
                        yield xai_service_pb2.ExplanationsRequest(
                            explanation_type = explanation_type,
                            explanation_method = explanation_method,
                            model=model,
                            feature1 = feature1,
                            feature2 = feature2
                        )
                    except:
                        print("error sending data")
                    print("Stopped Sending")
                elif explanation_method == 'CounterfactualExplanations':
                    try:
                        yield xai_service_pb2.ExplanationsRequest(
                            explanation_type = explanation_type,
                            explanation_method = explanation_method,
                            model=model,
                            target = target,
                            query=query.to_parquet(None),
                            )
                    except:
                        print("error sending data")
                    print("Stopped Sending")

    # def generate_dataframe_chunks(self,explanation_type,explanation_method,train_data,model, test_data=None, 
    #                               train_labels=None,test_labels=None,num_influential=None,features=None,feature1=None,feature2=None,target=None,query=None,chunk_size=1000):
    #     print('sending data')
    #     if explanation_type == 'Pipeline':
    #         if explanation_method == 'InfluenceFunctions':
    #             for i in range(0, len(train_data), chunk_size):
    #                 chunk = train_data[i:i + chunk_size]
    #                 chunk_train = train_labels[i:i + chunk_size]
    #                 chunk_data = chunk.to_parquet(None)
    #                 chunk_train = chunk_train.to_parquet(None)  
    #                 try:
    #                     yield xai_service_pb2.ExplanationsRequest(
    #                         explanation_type = explanation_type,
    #                         explanation_method = explanation_method,
    #                         train_data=chunk_data,
    #                         model=model.getvalue(),
    #                         test_data=test_data.to_parquet(None),
    #                         train_labels = chunk_train,
    #                         test_labels = test_labels.to_frame().to_parquet(None),
    #                         num_influential = num_influential,
    #                     )
    #                 except:
    #                     print("error sending data")
    #             print("Stopped Sending")

            # elif explanation_method == 'Counterfactual_Explanations':
            #     for i in range(0, len(train_data), chunk_size):
            #         chunk = train_data[i:i + chunk_size]
            #         chunk_train = train_labels[i:i + chunk_size]
            #         chunk_data = chunk.to_parquet(None)
            #         chunk_train = chunk_train.to_parquet(None)  
            #         try:
            #             yield xai_service_pb2.ExplanationsRequest(
            #                 explanation_type=explanation_type,
            #                 explanation_method = explanation_method,
            #                 train_data=chunk_data,
            #                 train_labels = chunk_train,
            #                 query=query.to_parquet(None),
            #                 model=model
            #             )
            #         except:
            #             print("error sending data")
        
        # elif explanation_type == 'Model':
        #     if explanation_method == 'PDPlots' or explanation_method=='ALEPlots':
        #         for i in range(0, len(train_data), chunk_size):
        #             chunk = train_data[i:i + chunk_size]
        #             chunk_data = chunk.to_parquet(None)
        #             try:
        #                 yield xai_service_pb2.ExplanationsRequest(
        #                     explanation_type = explanation_type,
        #                     explanation_method = explanation_method,
        #                     train_data=chunk_data,
        #                     model=model.getvalue(),
        #                     features = features
        #                 )
        #             except:
        #                 print("error sending data")
        #         print("Stopped Sending")
        #     elif explanation_method == '2D_PDPlots':
        #         for i in range(0, len(train_data), chunk_size):
        #             chunk = train_data[i:i + chunk_size]
        #             chunk_data = chunk.to_parquet(None)
        #             try:
        #                 yield xai_service_pb2.ExplanationsRequest(
        #                     explanation_type = explanation_type,
        #                     explanation_method = explanation_method,
        #                     train_data=chunk_data,
        #                     model=model.getvalue(),
        #                     feature1 = feature1,
        #                     feature2 = feature2
        #                 )
        #             except:
        #                 print("error sending data")
        #         print("Stopped Sending")

        #     elif explanation_method == 'CounterfactualExplanations':
        #         print('sending cf')
        #         for i in range(0, len(train_data), chunk_size):
        #             chunk = train_data[i:i + chunk_size]
        #             chunk_data = chunk.to_parquet(None)
        #             try:
        #                 yield xai_service_pb2.ExplanationsRequest(
        #                     explanation_type = explanation_type,
        #                     explanation_method = explanation_method,
        #                     train_data=chunk_data,
        #                     model=model.getvalue(),
        #                     target = target,
        #                     query=query.to_parquet(None),
        #                     )
        #             except:
        #                 print("error sending data")
        #         print("Stopped Sending")



    def get_explanations(self,explanation_type,explanation_method,param_grid=None,model=None,feature1=None,feature2=None,train_data=None,test_data=None,train_labels=None,test_labels=None,num_influential=None,proxy_dataset=None,query=None,features=None,target=None):

        # Create a stub for the Explanations service
        # model_bytes = io.BytesIO()
        # torch.save(model, model_bytes)
        # Prepare an ExplanationsRequest for ComputePDP
        if explanation_type == 'Pipeline':
            if explanation_method == '2D_PDPlots':

                explanations_response = self.stub.GetExplanation(self.dummy_stream(explanation_type=explanation_type,explanation_method=explanation_method,
                                                                                model=model,feature1=feature1,feature2=feature2))

            elif explanation_method == 'PDPlots' or explanation_method == 'ALEPlots': 
                explanations_response = self.stub.GetExplanation(self.dummy_stream(explanation_type=explanation_type,explanation_method=explanation_method,
                                                                                model=model,feature1=feature1))
            
            elif explanation_method == 'CounterfactualExplanations':
                explanations_response = self.stub.GetExplanation(self.dummy_stream(explanation_type=explanation_type,explanation_method=explanation_method,model=model))
                
            elif explanation_method == 'InfluenceFunctions':
                try:    
                    print("Start")
                    response = self.stub.GetExplanation(self.dummy_stream(explanation_type=explanation_type,explanation_method=explanation_method,model=model,num_influential=num_influential))
                except grpc.RpcError as e:
                    print(f"Error calling StreamDataFrame: {e}")
                    return None, None, None            


            # Make a gRPC call to the Explanations service
            if explanation_method == 'PDPlots':
                return explanations_response.pdp_hp_values, explanations_response.pdp_values
            elif explanation_method == 'ALEPlots' :
                return explanations_response.ale_data
            elif explanation_method == '2D_PDPlots':
                return explanations_response.pdp2d_xi, explanations_response.pdp2d_yi, explanations_response.pdp2d_zi
            elif explanation_method == 'CounterfactualExplanations':
                return explanations_response.cfs
            if explanation_method== 'InfluenceFunctions':
                return response.influences, response.positive, response.negative
            
        elif explanation_type == 'Model':
            if explanation_method == 'PDPlots' or explanation_method=='ALEPlots':
                explanations_response = self.stub.GetExplanation(self.dummy_stream(explanation_type=explanation_type,explanation_method=explanation_method,
                                                                                       model=model, feature1=feature1))
            elif explanation_method == '2D_PDPlots':
                explanations_response = self.stub.GetExplanation(self.dummy_stream(explanation_type=explanation_type,explanation_method=explanation_method,
                                                                                       model=model, feature1=feature1,feature2=feature2))
            elif explanation_method == 'CounterfactualExplanations':
                explanations_response = self.stub.GetExplanation(self.dummy_stream(explanation_type=explanation_type,explanation_method=explanation_method,
                                                                                       model=model,query=query, target=target))
            # elif explanation_method == 'ALEPlots':
            #     explanations_response = self.stub.GetExplanation(self.generate_dataframe_chunks(explanation_type=explanation_type,explanation_method=explanation_method,train_data=train_data,
            #                                                                            model=model_bytes, features=features))


            
            if explanation_method == 'PDPlots' or explanation_method == '2D_PDPlots':
                return explanations_response.pdp_vals, explanations_response.pdp_effect
            elif explanation_method == 'CounterfactualExplanations':
                return explanations_response.cfs
            elif explanation_method == 'ALEPlots':
                return explanations_response.ale_data


