import json
import grpc
import xai_service_pb2
import xai_service_pb2_grpc
import io
# import torch
from modules.lib import transform_grid_plt

class Client():
    def __init__(self):
        self.channel = grpc.insecure_channel('localhost:50051')
        self.stub = xai_service_pb2_grpc.ExplanationsStub(self.channel)

    def get_explanations(self,explanation_type,explanation_method,param_grid=None,model=None,model_id=None,feature1=None,feature2=None,train_data=None,test_data=None,train_labels=None,test_labels=None,num_influential=None,proxy_dataset=None,query=None,features=None,target=None):

        if explanation_type == 'hyperparameterExplanation':
            if explanation_method == '2dpdp':

                explanations_response = self.stub.GetExplanation(xai_service_pb2.ExplanationsRequest(explanation_type=explanation_type,explanation_method=explanation_method,
                                                                                model=model,model_id=None,feature1=feature1,feature2=feature2))

            elif explanation_method == 'pdp' or explanation_method == 'ale': 
                explanations_response = self.stub.GetExplanation(xai_service_pb2.ExplanationsRequest(explanation_type=explanation_type,explanation_method=explanation_method,
                                                                                model=model,model_id=None,feature1=feature1))
            
            elif explanation_method == 'counterfactuals':
                explanations_response = self.stub.GetExplanation(xai_service_pb2.ExplanationsRequest(explanation_type=explanation_type,explanation_method=explanation_method,model=model,model_id=None,))
                
            elif explanation_method == 'influenceFunctions':
                try:    
                    print("Start")
                    response = self.stub.GetExplanation(xai_service_pb2.ExplanationsRequest(explanation_type=explanation_type,explanation_method=explanation_method,model=model,model_id=None,num_influential=num_influential))
                except grpc.RpcError as e:
                    print(f"Error calling StreamDataFrame: {e}")
                    return None, None, None            


            # Make a gRPC call to the Explanations service
            if explanation_method == 'pdp':
                return explanations_response.explainability_type,explanations_response.explanation_method,explanations_response.explainability_model,explanations_response.plot_name,explanations_response.plot_descr,explanations_response.plot_type,explanations_response.features,explanations_response.xAxis,explanations_response.yAxis
            elif explanation_method == 'ale' :
                return explanations_response.explainability_type,explanations_response.explanation_method,explanations_response.explainability_model,explanations_response.plot_name,explanations_response.plot_descr,explanations_response.plot_type,explanations_response.features,explanations_response.xAxis,explanations_response.yAxis
            elif explanation_method == '2dpdp':
                return explanations_response.explainability_type,explanations_response.explanation_method,explanations_response.explainability_model,explanations_response.plot_name,explanations_response.plot_descr,explanations_response.plot_type,explanations_response.features,explanations_response.xAxis,explanations_response.yAxis,explanations_response.zAxis
            elif explanation_method == 'counterfactuals':
                return explanations_response.explainability_type,explanations_response.explanation_method,explanations_response.explainability_model,explanations_response.plot_name,explanations_response.plot_descr,explanations_response.plot_type,explanations_response.table_contents
            if explanation_method== 'influenceFunctions':
                return response.influences, response.positive, response.negative
            
        elif explanation_type == 'featureExplanation':
            if explanation_method == 'pdp' or explanation_method=='ale':
                explanations_response = self.stub.GetExplanation(xai_service_pb2.ExplanationsRequest(explanation_type=explanation_type,explanation_method=explanation_method,
                                                                                       model=model,model_id=model_id, feature1=feature1))
            elif explanation_method == '2dpdp':
                explanations_response = self.stub.GetExplanation(xai_service_pb2.ExplanationsRequest(explanation_type=explanation_type,explanation_method=explanation_method,
                                                                                       model=model,model_id=model_id, feature1=feature1,feature2=feature2))
            elif explanation_method == 'counterfactuals':
                explanations_response = self.stub.GetExplanation(xai_service_pb2.ExplanationsRequest(explanation_type=explanation_type,explanation_method=explanation_method,
                                                                                       model=model,model_id=model_id,query=query, target=target))
            
            if explanation_method == 'pdp' or explanation_method == '2dpdp':
                return explanations_response.explainability_type,explanations_response.explanation_method,explanations_response.explainability_model,explanations_response.plot_name,explanations_response.plot_descr,explanations_response.plot_type,explanations_response.features,explanations_response.xAxis,explanations_response.yAxis,explanations_response.zAxis
            elif explanation_method == 'counterfactuals':
                return explanations_response.explainability_type,explanations_response.explanation_method,explanations_response.explainability_model,explanations_response.plot_name,explanations_response.plot_descr,explanations_response.plot_type,explanations_response.table_contents
            elif explanation_method == 'ale':
                return explanations_response.explainability_type,explanations_response.explanation_method,explanations_response.explainability_model,explanations_response.plot_name,explanations_response.plot_descr,explanations_response.plot_type,explanations_response.features,explanations_response.xAxis,explanations_response.yAxis


