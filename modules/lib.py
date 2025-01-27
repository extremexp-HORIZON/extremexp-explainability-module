from skopt.space import Categorical,Real,Integer
import numpy as np
import os 
from sklearn.compose import ColumnTransformer
import pandas as pd
from sklearn.pipeline import Pipeline
from typing import Dict
from skopt.space import Space
import copy
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,RobustScaler,MinMaxScaler
import modules.clf_utilities as clf_ut
import joblib
import tensorflow as tf
import torch 

def _load_model(model_path):
  
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The specified model path does not exist: {model_path}")

    # Determine the file extension
    _, ext = os.path.splitext(model_path)

    # Handle sklearn models
    if ext in {".pkl", ".pickle"}:
        print("Sklearn model detected")
        name="sklearn"
        try:
            with open(model_path, "rb") as file:
                model = joblib.load(file)
                print("Sklearn model loaded")
        except ModuleNotFoundError as e:
            raise ImportError(
                f"Failed to load sklearn model. A version mismatch is likely. "
                f"Try using the same library version used to save the model. Original error: {e}"
            )
        except Exception as e:
            raise ValueError(f"Failed to load sklearn model from {model_path}: {e}")

    # Handle TensorFlow/Keras models
    elif ext in {".h5", ".keras"}:
        print("Tensorflow model detected")
        name = "tensorflow"
        try:
            model = tf.keras.models.load_model(model_path)
            print("Tensorflow model loaded")
        except Exception as e:
            raise ValueError(
                f"Failed to load TensorFlow/Keras model. Ensure you are using the same or a compatible "
                f"TensorFlow version. Original error: {e}"
            )

    # Handle PyTorch models
    elif ext in {".pt", ".pth"}:
        name = "pytorch"
        print("Pytorch model detected")
        try:
            model = torch.load(model_path)
            model.eval()  # Set the model to evaluation mode
            print("Pytorch model detected")
        except Exception as e:
            raise ValueError(
                f"Failed to load PyTorch model. Ensure you are using the same or a compatible "
                f"PyTorch version. Original error: {e}"
            )

    else:
        raise ValueError(f"Unsupported model format: {ext}")

    return model, name


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


def proxy_model(hyperparameters,metrics, clf):

    X1 = hyperparameters
    y1 = np.array(metrics)
    # for metric_name, metric_object in metrics.items():
    #     y1 = np.array(metric_object.value)

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

def instance_proxy(hyper_configs, misclassified_instance):
    import joblib
    # Creating proxy dataset for each hyperparamet configuration - prediction of test instance
    rows = []
    for config_name, config_data in hyper_configs.items():
        row = {}
        for key, value in config_data.hyperparameter.items():
            row[key] = cast_value(value.values, value.type)
        model, name = _load_model(config_name)
        if name == "sklearn":
            row['BinaryLabel'] = model.predict(misclassified_instance)[0] 
        elif name == "tensorflow":
            pass
            #row['BinaryLabel'] = np.argmax(model.predict(misclassified_instance),axis=1)
        rows.append(row)

    proxy_dataset = pd.DataFrame(rows)
    if name == 'tensorflow':
        proxy_dataset['BinaryLabel'] = np.random.choice([0, 1, 2], size=len(proxy_dataset))
    else:
        proxy_dataset['BinaryLabel'] = np.random.choice([0, 1], size=len(proxy_dataset))
    hyper_space = create_hyperspace(hyper_configs)
    param_grid = transform_grid(hyper_space)
    param_space, name = dimensions_aslists(param_grid)
    space = Space(param_space) 

    plot_dims = []
    for row in range(space.n_dims):
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
    print("Trained proxy model")
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


def cast_value(value, value_type):
    if value_type == "numeric":
        # Numeric types could include integers or floats
        return float(value) if '.' in value else int(value)
    elif value_type == "categorical":
        # Categorical values remain as strings
        return value
    else:
        # Default to string for unknown types
        return value
    
def create_hyperspace(model_configs):
    from collections import defaultdict

    aggregated_hyperparameters = defaultdict(set)

    # Parse each model configuration
    for model_config in model_configs.values():
        for key, value in model_config.hyperparameter.items():
            casted_value = cast_value(value.values, value.type)
            aggregated_hyperparameters[key].add(casted_value)

    # Convert sets to appropriate collections
    gridsearch_params = {
        key: list(value_set) if any(isinstance(v, str) for v in value_set) else tuple(sorted(value_set))
        for key, value_set in aggregated_hyperparameters.items()
    }

    return gridsearch_params

def create_hyper_df(model_configs,metrics):
    rows = []
    sorted_metrics = []
    for config_name, config_data in model_configs.items():
        row = {}
        sorted_metrics.append(metrics[config_name].value)
        for key, value in config_data.hyperparameter.items():
            row[key] = cast_value(value.values, value.type)
        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    return df, sorted_metrics

def create_cfquery_df(model_configs,model_name):
    rows = []
    for config_name, config_data in model_configs.items():
        row = {}
        if model_name == config_name:
            for key, value in config_data.hyperparameter.items():
                row[key] = cast_value(value.values, value.type)
                print(row)
            rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    return df