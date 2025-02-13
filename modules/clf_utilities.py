import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from modules.config import config


clf_callable_map = {
    'Linear Regression': LinearRegression(),
    'XGBoostRegressor': XGBRegressor()
    }

clf_hyperparams_map = {
    'Linear Regression': config.LinearRegression_hyperparameters,
    'XGBoostRegressor': config.XGBRegressor_hyperparameter
    }