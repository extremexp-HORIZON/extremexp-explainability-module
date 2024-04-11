import warnings
warnings.filterwarnings("ignore")

import numpy as np
import os
import itertools

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from modules.config import config


clf_callable_map = {
    'Linear Regression': LinearRegression(),
    'XGBoostRegressor': XGBRegressor()
    }

clf_hyperparams_map = {
    'Linear Regression': config.LinearRegression_hyperparameters,
    'XGBoostRegressor': config.XGBRegressor_hyperparameter
    }