from skopt.space import Categorical,Real,Integer
import numpy as np
import os 
from sklearn.compose import ColumnTransformer
import pandas as pd
from sklearn.pipeline import Pipeline
from typing import Dict,List,Optional
from skopt.space import Space
import copy
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import modules.clf_utilities as clf_ut
import joblib
import tensorflow as tf
import torch
import torch.nn as nn
import logging
from dataclasses import dataclass
logging.basicConfig(level=logging.INFO,force=True)
logger = logging.getLogger(__name__)


def _load_dataset(file_path: str) -> pd.DataFrame:
    """Load a dataset from a file, automatically detecting its format."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    if ext == ".csv":
        logger.info("CSV file detected")
        df = pd.read_csv(file_path)
        if 'Unnamed: 0' in df.columns.tolist():
            df.drop(columns='Unnamed: 0',inplace=True)
            
        return df
    elif ext == ".parquet":
        logger.info("Parquet file detected")
        df =  pd.read_parquet(file_path)
        if 'Unnamed: 0' in df.columns.tolist():
            df.drop(columns='Unnamed: 0',inplace=True)
        return df
    elif ext == ".pkl" or ext == ".pickle":
        logger.info("Pickle file detected")
        df =  pd.read_pickle(file_path)
        if 'Unnamed: 0' in df.columns.tolist():
            df.drop(columns='Unnamed: 0',inplace=True)
        return df
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def _load_multidimensional_array(file_path: str) -> torch.Tensor:
    """
    Load a multidimensional array from a file into a pytorch tensor.
    Currently supports numpy arrays (.npy) format.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    if ext == ".npy":
        logger.info("Numpy file detected")
        data = np.load(file_path, allow_pickle=True)
        ret = torch.from_numpy(data)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    return ret

def _load_model_sklearn(model_path: str):
    """Load a scikit-learn model from a pickle file."""
    logger.info("Sklearn model detected")
    try:
        with open(model_path, "rb") as file:
            model = joblib.load(file)
            logger.info("Sklearn model loaded")
        return model, "sklearn"
    except ModuleNotFoundError as e:
        raise ImportError(
            f"Failed to load sklearn model. A version mismatch is likely. "
            f"Try using the same library version used to save the model. Original error: {e}"
        )
    except Exception as e:
        raise ValueError(f"Failed to load sklearn model from {model_path}: {e}")

def _load_model_tf(model_path: str):
    """Load a TensorFlow/Keras model and wrap it for sklearn compatibility."""
    logger.info("Tensorflow model detected")
    try:
        tf_model = tf.keras.models.load_model(model_path)
        from sklearn.base import BaseEstimator

        class PredictionWrapper(BaseEstimator):
            def __init__(self, predict_func):
                self.predict_func = predict_func
                self.classes_ = np.array([0, 1])
            
            def fit(self, X, y=None):
                pass
            
            def predict(self, X):
                return self.predict_func(X)
            
            def predict_proba(self, X):
                predicted = tf_model.predict(X)
                if predicted.shape[1] == 1:
                    prob_positive = predicted.flatten()
                    return np.vstack([1 - prob_positive, prob_positive]).T
                else:
                    return predicted

            def __sklearn_is_fitted__(self):
                return True
            
            @property
            def _estimator_type(self):
                return "classifier"
        
        def predict_func(X):
            import tensorflow as tf
            predicted = tf_model.predict(X)
            if predicted.shape[1] == 1:
                return np.array([1 if x >= 0.5 else 0 for x in tf.squeeze(predicted)])
            else:
                return np.argmax(tf_model.predict(X), axis=1)
        
        model = PredictionWrapper(predict_func)
        logger.info("Tensorflow model loaded")
        return model, "tensorflow"
    except Exception as e:
        raise ValueError(
            f"Failed to load TensorFlow/Keras model. Ensure you are using the same or a compatible "
            f"TensorFlow version. Original error: {e}"
        )

def _load_model_pytorch(model_path: str):
    """Load a PyTorch model from a .pt or .pth file."""
    @dataclass
    class DetectionResult:
        type: str
        model: Optional[nn.Module] = None
    def detect_pt_file(path: str) -> DetectionResult:
        try:
            ts_mod = torch.jit.load(path)
            return DetectionResult("torchscript", ts_mod)
        except Exception:
            pass
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
            return DetectionResult("state_dict")
        if isinstance(obj, nn.Module):
            return DetectionResult("full_module")
        return DetectionResult("unknown")
    
    logger.info("Pytorch model detected")
    try:
        pytorch_model_type_detection = detect_pt_file(model_path)
        pytorch_model_type = pytorch_model_type_detection.type
        if pytorch_model_type == "torchscript":
            model = pytorch_model_type_detection.model
            if model is None:
                raise ValueError("Torchscript model detection failed.")
            model.eval()
            logger.info("Torchscript model detected")
        elif pytorch_model_type == "state_dict":
            raise NotImplementedError("Loading state_dict models is not implemented yet.")
        elif pytorch_model_type == "full_module":
            model = torch.load(model_path, map_location="cpu")
            model.eval()
            logger.info("Full module model detected")
        else:
            raise ValueError(f"Could not determine PyTorch model type. {model_path} seems to be none out of: 'torchscript', 'state_dict', or 'full_module'.")
        model.eval()
        logger.info("Pytorch model loaded successfully")
        return model, "pytorch"
    except Exception as e:
        raise ValueError(
            f"Failed to load PyTorch model. Ensure you are using the same or a compatible "
            f"PyTorch version. Original error: {e}"
        )

def _load_model(model_path: List):
    """
    Loads a machine learning model from the specified file path.
    
    Supports the following model types:
    - Scikit-learn models (.pkl, .pickle)
    - TensorFlow/Keras models (.h5, .keras)
    - PyTorch models (.pt, .pth)
    
    Args:
        model_path (str): Path to the saved model file.

    Returns:
        tuple: (model, model_type)
            - model: The loaded model object.
            - model_type (str): The type of model ('sklearn', 'tensorflow', or 'pytorch').
    
    Raises:
        FileNotFoundError: If the specified model file does not exist.
        ImportError: If a scikit-learn model has a version mismatch issue.
        ValueError: If the model format is unsupported or loading fails.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The specified model path does not exist: {model_path}")

    # Determine the file extension
    _, ext = os.path.splitext(model_path)
    ext = ext.lower()

    loaders = {
        ".pkl": _load_model_sklearn,
        ".pickle": _load_model_sklearn,
        ".h5": _load_model_tf,
        ".keras": _load_model_tf,
        ".pt": _load_model_pytorch,
        ".pth": _load_model_pytorch,
    }

    # Try the loader for the detected extension first
    tried_loaders = []
    if ext in loaders:
        try:
            return loaders[ext](model_path)
        except Exception as e:
            tried_loaders.append(ext)
            # Try other loaders
            for other_ext, loader in loaders.items():
                if other_ext != ext and other_ext not in tried_loaders:
                    try:
                        model, model_type = loader(model_path)
                        logger.warning(
                            f"Model loaded as '{model_type}' but file extension '{ext}' suggests otherwise. "
                            f"File may have been saved with the wrong extension."
                        )
                        return model, model_type
                    except Exception:
                        continue
            # If none succeeded, re-raise original error
            raise e
    else:
        raise ValueError(f"Unsupported model format: {ext}")

def transform_grid(param_grid: Dict
                   ) -> Dict:
    """
    Transforms a parameter grid dictionary into a format suitable for hyperparameter optimization.
    
    Args:
        param_grid (Dict): A dictionary where keys are parameter names and values define the search space.
            - If a value is a tuple:
                - If it has three elements and the third is a string, it is treated as a log-uniform distribution.
                - If it contains floats, it is treated as a uniform Real distribution.
                - Otherwise, it is treated as an Integer distribution.
            - If a value is a list of complex types, it is converted into a list of string representations.

    Returns:
        Dict: A transformed parameter grid dictionary suitable for optimization.
    """
    param_grid_copy = copy.deepcopy(param_grid)
    for key, value in param_grid.items():

        if isinstance(param_grid_copy[key],tuple):
            if (len(param_grid_copy[key]) == 3)  and (type(param_grid_copy[key][2]) == str):
                mins = param_grid_copy[key][0]
                maxs = param_grid_copy[key][1]
                param_grid_copy[key] = Real(mins,maxs,prior='log-uniform',transform='normalize')
            elif len(param_grid_copy[key]) == 1:
                param_grid_copy[key] = tuple(param_grid_copy[key],)
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
    """
    Constructs and trains a surrogate model to approximate the relationship between hyperparameters 
    and model performance metrics.

    Args:
        hyperparameters (pd.DataFrame): A dataframe containing the hyperparameter values for different runs.
        metrics (array-like): An array of performance metrics corresponding to each set of hyperparameters.
        clf (str): The key corresponding to the desired model in `clf_ut.clf_callable_map`.

    Returns:
        sklearn.pipeline.Pipeline: A trained surrogate model pipeline that preprocesses hyperparameters 
                                   and predicts performance metrics.

    Raises:
        KeyError: If `clf` is not found in `clf_ut.clf_callable_map` or `clf_ut.clf_hyperparams_map`.
        ValueError: If `hyperparameters` is empty or improperly formatted.
    """
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
    """
    Creates a proxy dataset and trains a proxy model based on hyperparameter configurations and a misclassified instance.

    The function iterates through each hyperparameter configuration, predicts the label for the misclassified instance
    using the corresponding model, and constructs a proxy dataset. A proxy model (SVM with a linear kernel) is then trained
    on this dataset to approximate the behavior of the original models.

    Parameters:
    -----------
    hyper_configs : dict
        A dictionary where keys are configuration names and values are objects containing hyperparameter configurations.
        Each configuration object should have a `hyperparameter` attribute, which is a dictionary of hyperparameters.

    misclassified_instance : array-like or pandas.DataFrame
        The instance that was misclassified by the original models. This is used to generate predictions for each
        hyperparameter configuration.

    Returns:
    --------
    proxy_model : sklearn.pipeline.Pipeline
        A trained proxy model (SVM with a linear kernel) that approximates the behavior of the original models.

    proxy_dataset : pandas.DataFrame
        A DataFrame containing the proxy dataset. Each row corresponds to a hyperparameter configuration, and the columns
        include the hyperparameters and the predicted label for the misclassified instance.
    """
    rows = []
    for config_name, config_data in hyper_configs.items():
        row = {}
        for key, value in config_data.hyperparameter.items():
            row[key] = cast_value(value.values, value.type)
        model, name = _load_model(config_name)
        if name == "sklearn":
            row['BinaryLabel'] = model.predict(misclassified_instance)[0] 
            print(row['BinaryLabel'])
        elif name == "tensorflow":
            row['BinaryLabel'] = model.predict(misclassified_instance)[0]
            print(row['BinaryLabel'])
        rows.append(row)

    proxy_dataset = pd.DataFrame(rows)
    if name == 'tensorflow':
        # pass
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
    # categorical = [name[i] for i,value in enumerate(iscat) if value == True]
    categorical = []
    for config_name, config_data in hyper_configs.items():
        for key, value in config_data.hyperparameter.items():
            if value.type == 'categorical':
                categorical.append(key)
        break
    print(categorical)
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
    """
    Applies Min-Max scaling to numerical features in the `factual` and `counterfactuals` datasets based on the
    distribution of the `proxy_dataset`.

    The function scales numerical features in the `factual` and `counterfactuals` datasets to the range [0, 1] using
    the `MinMaxScaler`. The scaling is fitted on the `proxy_dataset` to ensure consistent scaling across all datasets.

    Parameters:
    -----------
    proxy_dataset : pandas.DataFrame
        The dataset used to fit the `MinMaxScaler`. It should contain the same numerical features as `factual` and
        `counterfactuals`.

    factual : pandas.DataFrame
        The dataset representing the factual instance (original instance). Its numerical features will be scaled.

    counterfactuals : pandas.DataFrame
        The dataset representing counterfactual instances. Its numerical features will be scaled.

    label : str
        The name of the target column in the datasets. This column is excluded from scaling.

    Returns:
    --------
    factual : pandas.DataFrame
        The scaled version of the `factual` dataset with numerical features transformed to the range [0, 1].

    counterfactuals : pandas.DataFrame
        The scaled version of the `counterfactuals` dataset with numerical features transformed to the range [0, 1].
    """
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
        for column, value in row.items():
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
    """
    Casts a given value to a specified type based on the provided `value_type`.

    This function is used to convert a value to either a numeric type (float or int) or keep it as a categorical type (string).
    If the `value_type` is not recognized, the value is returned as a string by default.

    Parameters:
    -----------
    value : str
        The value to be cast. This is typically provided as a string and will be converted based on `value_type`.

    value_type : str
        The type to which the value should be cast. Supported types are:
        - "numeric": Converts the value to a float if it contains a decimal point, otherwise to an int.
        - "categorical": Keeps the value as a string.
        - Any other type: Returns the value as a string by default.

    Returns:
    --------
    int, float, or str
        The value cast to the appropriate type based on `value_type`.
    """
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
    """
    Creates a hyperparameter search space from a collection of model configurations.

    This function aggregates hyperparameters from multiple model configurations into a unified search space.
    It ensures that each hyperparameter is represented as a list of possible values (for categorical types)
    or a sorted tuple (for numeric types). The resulting search space is suitable for use in hyperparameter
    optimization algorithms like grid search.

    Parameters:
    -----------
    model_configs : dict
        A dictionary where keys are model configuration names and values are objects containing hyperparameter
        configurations. Each configuration object should have a `hyperparameter` attribute, which is a dictionary
        of hyperparameters and their values.

    Returns:
    --------
    gridsearch_params : dict
        A dictionary where keys are hyperparameter names and values are lists (for categorical hyperparameters)
        or sorted tuples (for numeric hyperparameters) of possible values. This represents the unified search space.
    """
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

def create_hyper_df(model_configs):
    """
    Creates a DataFrame containing hyperparameter configurations and their corresponding metric values.

    This function iterates through a dictionary of model configurations, extracts hyperparameters and their values,
    and constructs a DataFrame where each row represents a unique configuration. It also collects the metric values
    associated with each configuration.

    Parameters:
    -----------
    model_configs : dict
        A dictionary where keys are configuration names and values are objects containing hyperparameter configurations
        and metric values. Each configuration object should have:
        - A `hyperparameter` attribute, which is a dictionary of hyperparameters and their values.
        - A `metric_value` attribute, which represents the performance metric for the configuration.

    Returns:
    --------
    df : pandas.DataFrame
        A DataFrame where each row corresponds to a hyperparameter configuration. Columns represent hyperparameters,
        and rows represent configurations.

    sorted_metrics : list
        A list of metric values corresponding to each configuration in the same order as the rows in the DataFrame.
    """
    rows = []
    sorted_metrics = []
    for config_name, config_data in model_configs.items():
        row = {}
        # sorted_metrics.append(metrics[config_name].value)
        for key, value in config_data.hyperparameter.items():
            row[key] = cast_value(value.values, value.type)
        sorted_metrics.append(config_data.metric_value)
        rows.append(row)


    # Create DataFrame
    df = pd.DataFrame(rows)

    return df, sorted_metrics

def create_cfquery_df(model_configs,model_name):
    """
    Creates a DataFrame containing hyperparameter configurations for a specific model.

    This function filters a dictionary of model configurations to extract the hyperparameters for a specified model.
    It constructs a DataFrame where each row represents the hyperparameter configuration for the specified model.

    Parameters:
    -----------
    model_configs : dict
        A dictionary where keys are configuration names and values are objects containing hyperparameter configurations.
        Each configuration object should have a `hyperparameter` attribute, which is a dictionary of hyperparameters
        and their values.

    model_name : str
        The name of the model configuration to filter and extract. Only configurations matching this name will be included
        in the resulting DataFrame.

    Returns:
    --------
    df : pandas.DataFrame
        A DataFrame where each row corresponds to the hyperparameter configuration for the specified model.
        Columns represent hyperparameters, and rows represent configurations.
    """
    rows = []
    for config_name, config_data in model_configs.items():
        row = {}
        if model_name == config_name:
            for key, value in config_data.hyperparameter.items():
                row[key] = cast_value(value.values, value.type)
            rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    return df

def keep_common_variability_points(model_configs):
    """
    Keeps only the common variability point keys across all model configurations.
    If a key is not present in all configurations, it is removed from each configuration.

    Parameters:
    model_configs : dict
        A dictionary where keys are configuration names and values are objects containing hyperparameter configurations.
        Each configuration object should have a `hyperparameter` attribute, which is a dictionary of hyperparameters
        and their values.
    Returns:
    dict
        The modified model_configs dictionary with only common variability point keys retained.
    """
    # Find common keys across all configurations
    common_keys = set.intersection(*(set(config_data.hyperparameter.keys()) for config_data in model_configs.values()))

    # Remove keys that are not common from each configuration
    for config_data in model_configs.values():
        keys_to_remove = set(config_data.hyperparameter.keys()) - common_keys
        for key in keys_to_remove:
            del config_data.hyperparameter[key]

    return model_configs

def shap_waterfall_payload(
    ex,
    idx: int,
    class_idx: int | None = None,   # required for multiclass
    top_k: int = 10,
    include_rest: bool = True
):
    """
    Build a frontend-ready payload for a SHAP waterfall of a single instance.

    Returns:
      {
        "index": idx,
        "class_idx": class_idx or 0,
        "expected_value": float,
        "prediction_value": float,          # base + sum(shap)
        "contributions": [                  # sorted by |shap| desc
           {"feature": str, "feature_value": float|str,
            "shap": float, "abs_shap": float, "direction": "up"|"down"},
           ...
           {"feature": "others", "shap": ..., ...}     # optional remainder bucket
        ]
      }
    """
    vals = np.asarray(ex.values)
    vals = np.asarray(ex.values)
    if vals.ndim == 3:
        if class_idx is None:
            class_idx = 1  # e.g., positive class; map via model.classes_ if needed
        row_shap = vals[idx, class_idx, :]
        base = np.asarray(ex.base_values)[idx, class_idx]
    else:
        row_shap = vals[idx, :]
        base = np.atleast_1d(ex.base_values)[idx] if np.ndim(ex.base_values) > 0 else ex.base_values

    feat_names = list(ex.feature_names)
    row_x = np.asarray(ex.data)[idx]


    row_shap = vals[idx, :]
    base = np.atleast_1d(ex.base_values)[idx] if np.ndim(ex.base_values) > 0 else ex.base_values

    # Build and sort contributions
    items = []
    for f, v, s in zip(feat_names, row_x, row_shap):
        items.append({
            "feature": f,
            "feature_value": float(v) if np.issubdtype(type(v), np.number) else str(v),
            "shap": float(s),
            "abs_shap": float(abs(s)),
        })
    items.sort(key=lambda d: d["abs_shap"], reverse=True)

    # Keep top_k and bucket the rest
    if top_k is not None and len(items) > top_k:
        top = items[:top_k]
        if include_rest:
            rest_shap = float(sum(d["shap"] for d in items[top_k:]))
            rest_abs  = float(sum(d["abs_shap"] for d in items[top_k:]))
            top.append({
                "feature": f"{len(items) - top_k} other features",
                "feature_value": "",
                "shap": rest_shap,
                "abs_shap": rest_abs,
            })
        items = top

    prediction_value = float(base + sum(d["shap"] for d in items))  # (approx if bucketed)

    return {
        "expected_value": float(base),
        "prediction_value": prediction_value,
        "contributions": items
    }
