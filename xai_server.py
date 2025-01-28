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
from sklearn.inspection import permutation_importance
from modules.lib import _load_model

class ExplainabilityExecutor(ExplanationsServicer):

    def GetExplanation(self, request, context):
        print('Reading data')

        #for request in request_iterator:
        explanation_type = request.explanation_type
        explanation_method = request.explanation_method

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
            return handler.handle(request, explanation_type)
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


    def GetFeatureImportance(self, request, context):
        from ExplainabilityMethodsRepository.ExplanationsHandler import BaseExplanationHandler
        # def is_sklearn_model(model):
        #     return hasattr(model, 'predict') and hasattr(model, 'fit')
    
        # def is_tensorflow_model(model):
        #     return hasattr(model, 'predict') and 'tensorflow' in str(type(model)).lower()
        
        handler = BaseExplanationHandler()
        data_path = request.data
        target_columns = request.target_column
        test_index = request.test_index
        model_path = request.model
        model, name = _load_model(model_path[0])


        dataset = pd.read_csv(data_path,index_col=0)
        test_data = dataset.loc[list(test_index)]
        test_labels = test_data[target_columns]
        test_data = test_data.drop(columns=[target_columns])

        if name == 'sklearn':
            result = permutation_importance(model, test_data, test_labels,scoring='accuracy', n_repeats=10, random_state=42)
            feature_importances = list(zip(test_data.columns, result.importances_mean))
            sorted_features = sorted(feature_importances, key=lambda x: x[1], reverse=True)
        elif name == 'tensorflow':
            from sklearn.base import BaseEstimator

            class PredictionWrapper(BaseEstimator):
                def __init__(self, predict_func):
                    self.predict_func = predict_func
                
                def fit(self, X, y=None):
                    # Dummy fit method to satisfy the interface
                    pass
                
                def predict(self, X):
                    return self.predict_func(X)
                
            def predict_func(X):
                import tensorflow as tf
                predicted = model.predict(X)
                if predicted.shape[1] == 1:
                    return np.array([1 if x >= 0.5 else 0 for x in tf.squeeze(predicted)])
                else:
                    return np.argmax(model.predict(X),axis=1)
            
            wrapped_estimator = PredictionWrapper(predict_func)
            result = permutation_importance(wrapped_estimator, test_data, test_labels,scoring='accuracy', n_repeats=10, random_state=42)
            feature_importances = list(zip(test_data.columns, result.importances_mean))
            sorted_features = sorted(feature_importances, key=lambda x: x[1], reverse=True)

        return xai_service_pb2.FeatureImportanceResponse(feature_importances=[xai_service_pb2.FeatureImportance(feature_name=feature,importance_score=importance) for feature, importance in sorted_features])


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    xai_service_pb2_grpc.add_ExplanationsServicer_to_server(ExplainabilityExecutor(), server)
    #xai_service_pb2_grpc.add_InfluencesServicer_to_server(MyInfluencesService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()