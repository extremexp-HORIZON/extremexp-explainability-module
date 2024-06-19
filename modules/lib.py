import matplotlib.pyplot as plt
from skopt.plots import _cat_format
from matplotlib.ticker import MaxNLocator, FuncFormatter  # noqa: E402
from skopt.space import Categorical,Real,Integer
from functools import partial
import numpy as np
import os 
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from pandas import DataFrame
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from typing import List,Dict,Tuple
from skopt.space import Space
from modules.optimizer import ModelOptimizer
import copy
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from copy import deepcopy
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,RobustScaler,MinMaxScaler
from modules.config import config
import modules.clf_utilities as clf_ut
import pickle

def transform_grid_plt(param_grid: Dict
                   ) -> Dict:
    param_grid_copy = copy.deepcopy(param_grid)
    for key, value in param_grid.items():

        if isinstance(param_grid_copy[key],tuple):
            if (len(param_grid_copy[key]) == 3)  and (type(param_grid_copy[key][2]) == str):
                mins = param_grid_copy[key][0]
                maxs = param_grid_copy[key][1]
                param_grid_copy[key] = (mins,maxs)
            else:
                    mins = min(param_grid_copy[key])
                    maxs = max(param_grid_copy[key])
                    param_grid_copy[key] = (mins,maxs)

        if isinstance(value, list) and not isinstance(param_grid_copy[key][0],(str,int,float,type(None))):
            param_grid_copy[key] = [str(item) for item in value]          

    return param_grid_copy

def transform_grid(param_grid: Dict
                   ) -> Dict:
    param_grid_copy = copy.deepcopy(param_grid)
    for key, value in param_grid.items():

        if isinstance(param_grid_copy[key],tuple):
            if (len(param_grid_copy[key]) == 3)  and (type(param_grid_copy[key][2]) == str):
                mins = param_grid_copy[key][0]
                maxs = param_grid_copy[key][1]
                param_grid_copy[key] = Real(mins,maxs,prior='log-uniform',transform='normalize')
            elif type(param_grid_copy[key][0]) == float:
                mins = min(param_grid_copy[key])
                maxs = max(param_grid_copy[key])
                param_grid_copy[key] = Real(mins,maxs,prior='uniform',transform='normalize')
            else:
                mins = min(param_grid_copy[key])
                maxs = max(param_grid_copy[key])
                param_grid_copy[key] = Integer(mins,maxs,prior='uniform',transform='normalize')


        if isinstance(value, list) and not isinstance(param_grid_copy[key][0],(str,int,float,type(None))):
            param_grid_copy[key] = [str(item) for item in value]        

    return param_grid_copy


def dimensions_aslists(search_space : Dict
                       ):
    """Convert a dict representation of a search space into a list of
    dimensions, ordered by sorted(search_space.keys()).

    Parameters
    ----------
    search_space : dict
        Represents search space. The keys are dimension names (strings)
        and values are instances of classes that inherit from the class
        :class:`skopt.space.Dimension` (Real, Integer or Categorical)

    Returns
    -------
    params_space_list: list
        list of skopt.space.Dimension instances
    """
    params_space_list = [
        search_space[k] for k in sorted(search_space.keys())
    ]
    name = [
        k for k in sorted(search_space.keys())
    ]
    return params_space_list,name

def transform_samples(hyperparameters : List[Dict],
                      name: List
                      ) -> np.ndarray:
    rearranged_list = []

    for dictionary in hyperparameters:
        rearranged_dict = {key: dictionary[key] for key in name}
        rearranged_list.append(rearranged_dict)


    spaces = [list(rearranged_list[i].values()) for i in range(len(hyperparameters))]
    for sublist in spaces:
        for i in range(len(sublist)):
            if not isinstance(sublist[i], (int,float,str,type(None))):
                sublist[i] = str(sublist[i])
    #samples = space.transform(spaces)

    return spaces

def gaussian_objective(objective : str, 
                       optimizer : ModelOptimizer,
                       samples : np.ndarray):

    if objective == 'accuracy':
        gaussian = pd.DataFrame(samples)
        gaussian['label'] = optimizer.cv_results_['mean_test_score']
        gaussian = gaussian.dropna().reset_index(drop=True)
    elif objective == 'fit_time':
        gaussian = pd.DataFrame(samples)
        gaussian['label'] = optimizer.cv_results_['mean_fit_time']
        gaussian = gaussian[gaussian.label !=0 ]
    elif objective == 'score_time':
        gaussian = pd.DataFrame(samples)
        gaussian['label'] = optimizer.cv_results_['mean_score_time']
        gaussian = gaussian[gaussian.label !=0 ]

    X = gaussian.drop(columns='label')
    y = gaussian['label']

    return X,y

def is_logspaced(arr):
    if len(arr) < 3:
        return False  # Arrays with less than 3 elements are not log-spaced

    ratios = arr[1:] / arr[:-1]
    return np.allclose(ratios, ratios[0])



def convert_to_float32(train):
    return train.astype(np.float32)

def proxy_model(parameter_grid,optimizer,objective,clf):

    param_grid = transform_grid(parameter_grid)
    param_space, name = dimensions_aslists(param_grid)


    hyperparameters = optimizer.cv_results_['params']
    samples = transform_samples(hyperparameters,name)
    # Prepare the hyperparameters and corresponding accuracy scores

    # Convert hyperparameters to a feature matrix (X) and accuracy scores to a target vector (y)

    X1 , y1 = gaussian_objective(objective,optimizer,samples)
    cat_columns = X1.select_dtypes(exclude=[np.number]).columns.tolist()
    numeric_columns = X1.select_dtypes(exclude=['object']).columns.tolist()
    numerical_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])


    one_hot_encoded_transformer = Pipeline([
        ('one_hot_encoder', OneHotEncoder())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer,numeric_columns),
            # ('label',label_encoded_transformer,label_encoded_features),
            ('one_hot', one_hot_encoded_transformer, cat_columns)
        ])
    surrogate_model_accuracy = Pipeline([("preprocessor", preprocessor),
                            ("Model", clf_ut.clf_callable_map[clf].set_params(**clf_ut.clf_hyperparams_map[clf]))])



    # Fit the surrogate model on the hyperparameters and accuracy scores
    surrogate_model_accuracy.fit(X1, y1)

    return surrogate_model_accuracy


def instance_proxy(X_train,y_train,optimizer, misclassified_instance,params):
    MODELS_DICT_PATH = 'proxy_data_models/cf_trained_models.pkl'
    try:
        with open(MODELS_DICT_PATH, 'rb') as f:
            trained_models = pickle.load(f)
    except FileNotFoundError:
        trained_models = {}
    # Creating proxy dataset for each hyperparamet configuration - prediction of test instance
    proxy = pd.DataFrame(columns = ['hyperparameters','BinaryLabel'])
    # Iterate through each hyperparameter combination
    for i,params_dict in enumerate(optimizer.cv_results_['params']):
        if i in trained_models.keys():
            mdl = trained_models[i]
        else:
        # Retrain the model with the current hyperparameters
            mdl = deepcopy(optimizer.estimator)
            mdl.set_params(**params_dict)
            mdl.fit(X_train, y_train)
            trained_models[i] = mdl
        
        # Make prediction for the misclassified instance
        prediction = mdl.predict(misclassified_instance.to_frame().T)[0]
        proxy = proxy.append({'hyperparameters' : params_dict, 'BinaryLabel': prediction},ignore_index=True)
    if not os.path.isfile(MODELS_DICT_PATH):
        with open(MODELS_DICT_PATH, 'wb') as f:
            pickle.dump(trained_models, f)
    
    keys = list(proxy['hyperparameters'].iloc[0].keys())

    # Create new columns for each key
    for key in keys:
        proxy[key] = proxy['hyperparameters'].apply(lambda x: x.get(key, None))

# Drop the original "Hyperparameters" column
    proxy_dataset = proxy.drop(columns=['hyperparameters'])
    proxy_dataset['BinaryLabel'] = proxy_dataset['BinaryLabel'].astype(int)

    param_grid = transform_grid(params)
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

    # Create proxy model
    cat_transf = ColumnTransformer(transformers=[("cat", OneHotEncoder(), categorical)], remainder="passthrough")

    proxy_model = Pipeline([
        ("one-hot", cat_transf),
        ("svm", SVC(kernel='linear', C=2.0 ,probability=True))
    ])

    proxy_model = proxy_model.fit(proxy_dataset.drop(columns='BinaryLabel'), proxy_dataset['BinaryLabel'])

    return proxy_model , proxy_dataset


def min_max_scale(proxy_dataset,factual,counterfactuals):
    scaler = MinMaxScaler()
    dtypes_dict = counterfactuals.drop(columns='BinaryLabel').dtypes.to_dict()
    # Change data types of columns in factual based on dtypes of counterfactual
    for col, dtype in dtypes_dict.items():
        factual[col] = factual[col].astype(dtype)
        
    
#pd.concat([factual,counterfactuals])
    for feat in proxy_dataset.drop(columns='BinaryLabel').select_dtypes(include='number').columns.tolist():
        scaler.fit(proxy_dataset.drop(columns='BinaryLabel')[feat].values.reshape(-1,1))
        #scaler.fit(pd.concat([factual,counterfactuals]).drop(columns='BinaryLabel')[feat].values.reshape(-1,1))
        scaled_data = scaler.transform(factual[feat].values.reshape(-1,1))
        factual[feat] = scaled_data
        scaled_data = scaler.transform(counterfactuals[feat].values.reshape(-1,1))
        counterfactuals[feat] = scaled_data

    return factual,counterfactuals

def cf_difference(base_model, cf_df):
    """
    Calculate the difference between the base model and each row of the provided counterfactual DataFrame.
    
    Parameters:
    - base_model: DataFrame, representing the base model with hyperparameters
    - cf_df: DataFrame, representing the counterfactual DataFrame with hyperparameters
    
    Returns:
    - DataFrame with differences added as a new column
    """
    differences = []
    
    # Ensure the base_model DataFrame has only one row
    if len(base_model) != 1:
        raise ValueError("Base model DataFrame must have exactly one row.")

    # Get the single row of the base model
    base_row = base_model.iloc[0]
    
    # Iterate over each row in the counterfactual DataFrame
    for index, row in cf_df.iterrows():
        difference = 0
        
        # Iterate over each column in the counterfactual DataFrame
        for column, value in row.iteritems():
            # Exclude 'BinaryLabel' column
            if column == 'BinaryLabel':
                continue
            
            # Check if the column is numerical
            try:
                # Compute the absolute difference for numerical columns
                difference += abs(value - base_row[column])
            except:
                # For categorical values, difference is 1 if they are different
                if str(value) != str(base_row[column]):
                    difference += 1
                    
        # Append the difference for the current row
        differences.append(difference)
    
    # Add the differences as a new column in the counterfactual DataFrame
    cf_df['Difference'] = differences
    
    return cf_df['Difference']