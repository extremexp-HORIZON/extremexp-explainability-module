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
# from sklearn.gaussian_process.kernels import ConstantKernel,Matern
# from sklearn.gaussian_process.kernels import WhiteKernel
# from skopt.learning.gaussian_process.gpr import GaussianProcessRegressor
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from copy import deepcopy
# from sklearn.linear_model import LinearRegression
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
                # if is_logspaced(np.array(param_grid[key])) :
                #     mins = min(param_grid[key])
                #     maxs = max(param_grid[key])
                #     param_grid[key] = Real(mins,maxs,prior='log-uniform',transform='normalize')
                # else:
                    mins = min(param_grid_copy[key])
                    maxs = max(param_grid_copy[key])
                    param_grid_copy[key] = (mins,maxs)

        # if isinstance(param_grid[key][0],(int,float)) and isinstance(param_grid[key][2],(str)):
        #     continue
        # if isinstance(param_grid[key][0],(int,float)):
        #     param_grid[key] = tuple((min(param_grid[key]),max(param_grid[key])))

        if isinstance(value, list) and not isinstance(param_grid_copy[key][0],(str,int,float,type(None))):
            param_grid_copy[key] = [str(item) for item in value]
        # if isinstance(value, list) and isinstance(param_grid[key][0],(int,float)):
        #     mins = min(param_grid[key])
        #     maxs = max(param_grid[key])
        #     param_grid[key] = Integer(mins,maxs,prior='uniform',transform='normalize')             

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


def compute_scaled_datasets(data : DataFrame,
                            scaler : List = None,
                            imputer : List = None
                            ) -> Dict:

    numeric_cols=data.select_dtypes(include=np.number).columns
    scalers = {str(scaler): scaler for scaler in scaler}
    if imputer != None:
        imputers = {str(imputer): imputer for imputer in imputer}

    # Create an empty dictionary to store the scaled datasets
    scaled_datasets = {}

    # Iterate over the imputers
    if imputer != None:
        
        for imputer_name, imputer in imputers.items():
        # Iterate over the scalers
            for scaler_name, scaler in scalers.items():
            # Create a pipeline with the imputer and scaler
                num_transform = Pipeline([
                    ('imputer', SimpleImputer(strategy=imputer)),
                    ('scaler', scaler)
                ])
            
                preprocessor = ColumnTransformer(
                    transformers=[
                    ("num_transform", num_transform, numeric_cols)])
            # Fit and transform the pipeline on the data
                scaled_X = preprocessor.fit_transform(data)

            # Create a DataFrame with the scaled data
                scaled_df = pd.DataFrame(scaled_X.round(3), columns=data.columns)

            # Store the scaled dataset in the dictionary with a unique key
                key = f'{imputer_name}_{scaler_name}'
                scaled_datasets[key] = scaled_df
    else:
        for scaler_name, scaler in scalers.items():
            # Create a pipeline with the imputer and scaler
                num_transform = Pipeline(steps = [
                    ('scaler', scaler)
                ])
                preprocessor = ColumnTransformer(
                    transformers=[
                    ("num_transform", num_transform, numeric_cols)])
                
            # Fit and transform the pipeline on the data
                scaled_X = preprocessor.fit_transform(data)

            # Create a DataFrame with the scaled data
                scaled_df = pd.DataFrame(scaled_X.round(3), columns=data.columns)

            # Store the scaled dataset in the dictionary with a unique key
                key = f'{scaler_name}'
                scaled_datasets[key] = scaled_df


    return scaled_datasets

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
                      space : Space,
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

         
def plot_pdp_1D_grpc(xi,yi,param_grid):

    features = list(param_grid.keys())
    param_grid = transform_grid(param_grid)
    param_space, name = dimensions_aslists(param_grid)
    space = Space(param_space)
    space.set_transformer_by_type('label',Categorical)

    plot_dims = []
    for row in range(space.n_dims):
        if space.dimensions[row].is_constant:
            continue
        plot_dims.append((row, space.dimensions[row]))
    n_dims = len(plot_dims)

    fig, ax = plt.subplots(n_dims, 1,
                           figsize=(10, 2 * n_dims))

    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                        hspace=0.5, wspace=0.1)
    
    for i in range(n_dims):
            index, dim = plot_dims[i]
        # for x,y in zip(xi,yi):
            if n_dims > 1:
                ax_ = ax[i]
            else:
                ax_ = ax
            iscat = [isinstance(dim[1], Categorical) for dim in plot_dims]
            if not iscat[i]:
                low, high = dim.bounds
                ax_.set_xlim(low, high)
            #     ax_.plot(xi, yi)
            # else: 
            #      ax_.bar(xi,yi)
            ax_.yaxis.tick_left()
            ax_.yaxis.set_label_position('left')
            ax_.yaxis.set_ticks_position('both')
            ax_.xaxis.tick_bottom()
            ax_.xaxis.set_label_position('bottom')
            ax_.set_xlabel(dim)

            if dim.prior == 'log-uniform':
                ax_.set_xscale('log')
            else:
                ax_.xaxis.set_major_locator(MaxNLocator(6, prune='both',
                                                            integer=iscat[i]))
            # if iscat[i]:
            #     ax_.xaxis.set_major_formatter(FuncFormatter(
            #                 partial(_cat_format, dim)))
            ax_.plot(xi[i], yi[i])

            fig.suptitle('Partial Dependence Plots for each Hyperparameter')
            ax_.set_xlabel(name[i])
            #ax_.set_ylabel(objectives[0])

def plot_pdp_2D_grpc(xi,yi,zi,param_grid,feature1,feature2):
    
    features = list(param_grid.keys())
    param_grid = transform_grid(param_grid)
    param_space, name = dimensions_aslists(param_grid)
    space = Space(param_space)
    space.set_transformer_by_type('label',Categorical)

    plot_dims = []
    for row in range(space.n_dims):
        if space.dimensions[row].is_constant:
            continue
        plot_dims.append((row, space.dimensions[row]))
    n_dims = len(plot_dims)


    index1 = features.index(feature1)
    index2 = features.index(feature2)
    _ ,dim_1 = plot_dims[index1]
    _ ,dim_2 = plot_dims[index2]
    fig, ax = plt.subplots()
    iscat = [isinstance(dim[1], Categorical) for dim in plot_dims]
    # if not iscat[index1]:  # bounds not meaningful for categoricals 
    #         ax.set_ylim(*dim_1.bounds)
    # else:
    #     ax.yaxis.set_major_locator(MaxNLocator(6, integer=iscat[index1]))
    #     ax.yaxis.set_major_formatter(FuncFormatter(
    #                     partial(_cat_format, dim_1)))
    # if iscat[index2]:
    #             ax.xaxis.set_major_locator(MaxNLocator(6,integer=iscat[index2]))
    #             # partial() avoids creating closures in a loop
    #             ax.xaxis.set_major_formatter(FuncFormatter(
    #                 partial(_cat_format, dim_2)))
    # else:
    #         ax.set_xlim(*dim_2.bounds)

        
    im = ax.contourf(xi, yi, zi, 10,
                              cmap='viridis_r')
    ax.set_xlabel(feature2)
    ax.set_ylabel(feature1)
    fig.colorbar(im,label='Accuracy Score')


def plot_ale_grpc(data,param_grid):

    param_grid = transform_grid(param_grid)
    param_space, name = dimensions_aslists(param_grid)
    space = Space(param_space)
    space.set_transformer_by_type('label',Categorical)

    plot_dims = []
    for row in range(space.n_dims):
        if space.dimensions[row].is_constant:
            continue
        plot_dims.append((row, space.dimensions[row]))

    n_dims = len(plot_dims)

    fig, ax = plt.subplots(n_dims, 1,
                           figsize=(10, 3 * n_dims))

    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                        hspace=0.5, wspace=0.1)

    for i,feat in enumerate(name):
        ale_eff = data[i]
        index, dim = plot_dims[i]
        iscat = [isinstance(dim[1], Categorical) for dim in plot_dims]
        ax_ = ax[i]
    
        if not iscat[i]:
            #ax_.set_xscale('log')    
            # sample = space.rvs(n_samples=len(ale_eff)) # grid_size + 1
            # xi = space.transform(sample)

            # xi[:,i] = ale_eff.index.values
            # xi = space.inverse_transform(xi)
            ax_.plot(ale_eff.index,ale_eff['eff'])
        else:
            ax_.errorbar(
            ale_eff.index.astype(str),
            ale_eff["eff"],
            yerr=0,
            capsize=3,
            marker="o",
            linestyle="dashed",
        )
            ax2 = ax_.twinx()
            ax2.set_ylabel("Size")
            ax2.bar(ale_eff.index.astype(str), ale_eff["size"], alpha=0.1, align="center")
            ax2.tick_params(axis="y")
            #ax2.set_title("1D ALE Plot - Discrete/Categorical")
            #fig.tight_layout()
        if not iscat[i]:
            low, high = dim.bounds
            ax_.set_xlim(low, high)
        if dim.prior == 'log-uniform':
            ax_.set_xscale('log')
        else:
            ax_.xaxis.set_major_locator(MaxNLocator(6, prune='both',
                                                            integer=iscat[i]))
        if iscat[i]:
            ax_.xaxis.set_major_formatter(FuncFormatter(
                            partial(_cat_format, dim)))
        ax_.set_ylabel("Effect on prediction (centered)")
        ax_.set_xlabel(feat)
        fig.suptitle('Accumulated Local Effects Plots for each Hyperparameter')
        plt.tight_layout()

def convert_to_float32(train):
    return train.astype(np.float32)

def proxy_model(parameter_grid,optimizer,objective,clf):

    param_grid = transform_grid(parameter_grid)
    param_space, name = dimensions_aslists(param_grid)
    space = Space(param_space)

    # space.set_transformer_by_type('normalize',Categorical)
    # space.set_transformer_by_type('normalize',Integer)
    # space.set_transformer_by_type('normalize',Real)

    hyperparameters = optimizer.cv_results_['params']
    samples = transform_samples(hyperparameters,space,name)
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


    # kernel = ConstantKernel(1.0, (0.01, 1000.0)) \
    #         *Matern(
    #         length_scale=np.ones(n_dims),
    #         length_scale_bounds=[(0.01, 100)] * n_dims, nu=2.5) + WhiteKernel()

    # surrogate_model_accuracy = GaussianProcessRegressor(kernel=kernel,normalize_y=True,random_state=1,noise="gaussian",
    #             n_restarts_optimizer=2)

    # Fit the surrogate model on the hyperparameters and accuracy scores
    surrogate_model_accuracy.fit(X1, y1)

    return surrogate_model_accuracy

    # # Generate new hyperparameters for evaluation
    # new_hyperparameters = space.transform([[1, 'auto',2 ,150,'RobustScaler()']])  # Example hyperparameters

    # # Predict the accuracy scores using the surrogate model
    # predicted_scores = surrogate_model.predict(new_hyperparameters)

    # # Print the predicted accuracy scores
    # print("Predicted Accuracy Scores:", predicted_scores)

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