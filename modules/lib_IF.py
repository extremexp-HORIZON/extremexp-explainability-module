from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import numpy as np
from pydvl.influence.torch import (
    DirectInfluence,
)
#from pydvl.influence.torch import TorchTwiceDifferentiable
import pandas as pd
from IPython.display import display
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder,MinMaxScaler


def preprocess_data(data,label_encoded_features,one_hot_encoded_features,numerical_features):
    label_encoder = LabelEncoder()
    for feature in label_encoded_features:
        data[feature] = label_encoder.fit_transform(data[feature])

    #one hot encoding of choosen features
    new_data = pd.concat([data, pd.get_dummies(data[one_hot_encoded_features])], axis=1)
    new_data = new_data.drop(columns=one_hot_encoded_features)

    scaler = StandardScaler()
    new_data[numerical_features] = scaler.fit_transform(new_data[numerical_features])
    
    return new_data


def compute_IF(
        model : nn.Module, 
        loss, 
        training_data : np.array, 
        test_data : np.array,
        train_labels : np.array,
        test_labels : np.array,
        influence_type : str,
        inversion_method : str,
        hessian_regularization : int       
) -> torch.tensor :
    
        if len(test_data.shape) == 1:

                train_data_loader = DataLoader(
                        TensorDataset(
                        torch.tensor(training_data.astype(float), dtype=torch.float), 
                        torch.tensor(train_labels.astype(float), dtype=torch.float)),
                        batch_size=32,
                )

                training_data = (torch.tensor(training_data.astype(float), dtype=torch.float), torch.tensor(train_labels, dtype=torch.float))
                test_data = (torch.tensor(test_data.astype(float), dtype=torch.float).reshape(1,-1), torch.tensor(test_labels, dtype=torch.float).unsqueeze(-1).unsqueeze(-1))

                influence_model = DirectInfluence(
                        model,
                        loss,
                        hessian_regularization=hessian_regularization,
                )
                influence_model = influence_model.fit(train_data_loader)
                influence_values = influence_model.influences(*test_data, *training_data, mode=influence_type)

        else:
                train_x = torch.as_tensor(training_data,dtype=torch.float)
                train_y = torch.as_tensor(train_labels, dtype=torch.float).unsqueeze(-1)
                test_x = torch.as_tensor(test_data,dtype=torch.float)
                test_y = torch.as_tensor(test_labels, dtype=torch.float).unsqueeze(-1)

                train_data_loader = DataLoader(
                        TensorDataset(train_x, train_y),
                        batch_size=32,
                )


                influence_model = DirectInfluence(
                        model,
                        loss,
                        hessian_regularization=hessian_regularization,
                )

                influence_model = influence_model.fit(train_data_loader)
                influence_values = influence_model.influences(test_x, test_y, train_x, train_y, mode=influence_type)
                print(influence_values)
        return influence_values

def show_neg_inf_instance(influences,num_instances,train_data,train_labels):
        influences = pd.DataFrame(influences)
        mean_train_influences = np.mean(influences, axis=0)
        mean_train_influences = pd.DataFrame(mean_train_influences)
        influential = train_data.loc[mean_train_influences.nsmallest(num_instances,0).index]
        influential['label'] = train_labels.loc[mean_train_influences.nsmallest(num_instances,0).index]
        influential['influence'] = mean_train_influences[0]
        return influential

def show_pos_inf_instance(influences,num_instances,train_data,train_labels):
        influences = pd.DataFrame(influences)
        mean_train_influences = np.mean(influences, axis=0)
        mean_train_influences = pd.DataFrame(mean_train_influences)
        influential = train_data.loc[mean_train_influences.nlargest(num_instances,0).index]
        influential['label'] = train_labels.loc[mean_train_influences.nlargest(num_instances,0).index]
        influential['influence'] = mean_train_influences[0]
        return influential
        

def show_influences(influences,num_instances,train_data,train_labels):

        print('Train instances with negative influence')
        display(show_neg_inf_instance(influences,num_instances,train_data,train_labels))
        print("")
        print("Train instances with positive influence")
        display(show_pos_inf_instance(influences,num_instances,train_data,train_labels))