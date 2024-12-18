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
        search_space[k] for k in search_space.keys()
    ]
    name = [
        k for k in search_space.keys()
    ]
    return params_space_list,name

def _evenly_sample(dim, n_points):
    """Return `n_points` evenly spaced points from a Dimension.

    Parameters
    ----------
    dim : `Dimension`
        The Dimension to sample from.  Can be categorical; evenly-spaced
        category indices are chosen in order without replacement (result
        may be smaller than `n_points`).

    n_points : int
        The number of points to sample from `dim`.

    Returns
    -------
    xi : np.array
        The sampled points in the Dimension.  For Categorical
        dimensions, returns the index of the value in
        `dim.categories`.

    xi_transformed : np.array
        The transformed values of `xi`, for feeding to a model.
    """
    cats = np.array(getattr(dim, 'categories', []), dtype=object)
    if len(cats):  # Sample categoricals while maintaining order
        xi = np.linspace(0, len(cats) - 1, min(len(cats), n_points),
                         dtype=int)
        #xi_transformed = dim.transform(cats[xi])
    else:
        bounds = dim.bounds
        # XXX use linspace(*bounds, n_points) after python2 support ends
        xi = np.linspace(bounds[0], bounds[1], n_points)
        #xi_transformed = dim.transform(xi)
    return xi

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

def transform_to_param_grid(data_list):
    parameter_grid = {}

    # Iterate over each entry in the data
    for entry in data_list:
            for key, value in entry.items():
                # Skip 'accuracy' as it is not part of the parameter grid
                if key == 'accuracy':
                    continue
                
                # If the key is not in the parameter_grid, initialize it
                if key not in parameter_grid:
                    parameter_grid[key] = []
                
                # Add the value to the list of values for this parameter
                if isinstance(value,list):
                    parameter_grid[key].append(value[0])
                else:
                    parameter_grid[key].append(value)

    # Convert to tuple for numeric values, keep lists/objects, and ensure uniqueness
    for key, values in parameter_grid.items():
        # Ensure unique values
        if isinstance(values[0], list):
            # Use a set to maintain uniqueness for lists of lists
            unique_values = []
            seen = set()  # To track seen tuples
            for v in values:
                # Convert inner lists to tuples for hashability
                tuple_v = tuple(v)
                if tuple_v not in seen:
                    seen.add(tuple_v)
                    unique_values.append(str(v))  # Convert back to list
            parameter_grid[key] = unique_values
        else:
            # For other types, ensure uniqueness by using a set
            parameter_grid[key] = list(set(values))

        # Convert numeric types to tuples
        if all(isinstance(val, (int, float)) for val in parameter_grid[key]):
            parameter_grid[key] = tuple(parameter_grid[key])  # Convert to tuple for numerical parameters

    return parameter_grid


# def proxy_model(workflows,clf):
#     proxy_data = []

#     # Iterate through the workflows in the JSON
#     for workflow_key, workflow_data in workflows.items():
#         # Find the 'TrainModel' task and extract parameters
#         for task in workflow_data['tasks']:
#             if task['id'] == 'TrainModel':
#                 parameters = {}
#                 for param in task['parameters']:
#                     value = param['value']
                    
#                     # Check if the parameter value is a list
#                     if isinstance(value, list):
#                         # If it's already a list, wrap it in another list to make a list of lists
#                         parameters[param['name']] = str(value)
#                     else:
#                         # Handle integers and floats appropriately
#                         if param['type'] == 'integer':
#                             parameters[param['name']] = int(value)
#                         elif param['type'] == 'float':
#                             parameters[param['name']] = float(value)
#                         else:
#                             parameters[param['name']] = value  # Handle other types normally
        
#         # Find the accuracy metric
#         for metric in workflow_data['metrics']:
#             for metric_key, metric_value in metric.items():
#                 if metric_value['name'] == 'accuracy':
#                     parameters['accuracy'] = float(metric_value['value'])

#         # Append parameters to the data list
#         proxy_data.append(parameters)


#     proxy_dataset = pd.DataFrame(proxy_data)
#     X1 , y1 = proxy_dataset.drop(columns='accuracy') , proxy_dataset['accuracy']
#     cat_columns = X1.select_dtypes(exclude=[np.number]).columns.tolist()
#     numeric_columns = X1.select_dtypes(exclude=['object']).columns.tolist()
#     numerical_transformer = Pipeline([
#         ('scaler', StandardScaler())
#     ])


#     one_hot_encoded_transformer = Pipeline([
#         ('one_hot_encoder', OneHotEncoder())
#     ])

#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', numerical_transformer,numeric_columns),
#             # ('label',label_encoded_transformer,label_encoded_features),
#             ('one_hot', one_hot_encoded_transformer, cat_columns)
#         ])
#     surrogate_model_accuracy = Pipeline([("preprocessor", preprocessor),
#                             ("Model", clf_ut.clf_callable_map[clf].set_params(**clf_ut.clf_hyperparams_map[clf]))])



#     # Fit the surrogate model on the hyperparameters and accuracy scores
#     surrogate_model_accuracy.fit(X1, y1)

#     return surrogate_model_accuracy, proxy_data
def proxy_model(parameter_grid,optimizer,objective,clf):

    param_grid = transform_grid(parameter_grid)
    _, name = dimensions_aslists(param_grid)


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
    MODELS_DICT_PATH = 'metadata/proxy_data_models/cf_trained_models.pkl'
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


def min_max_scale(proxy_dataset,factual,counterfactuals,label):
    scaler = MinMaxScaler()
    dtypes_dict = counterfactuals.drop(columns=label).dtypes.to_dict()
    # Change data types of columns in factual based on dtypes of counterfactual
    for col, dtype in dtypes_dict.items():
        factual[col] = factual[col].astype(dtype)
        
    
#pd.concat([factual,counterfactuals])
    for feat in proxy_dataset.drop(columns=label).select_dtypes(include='number').columns.tolist():
        scaler.fit(proxy_dataset.drop(columns=label)[feat].values.reshape(-1,1))
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
            if column == 'BinaryLabel' or column=='Label':
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