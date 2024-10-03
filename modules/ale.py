import pandas as pd
import numpy as np
from modules.ALE_generic import ale 
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import t
from typing import List,Dict,Tuple
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter  # noqa: E402
from skopt.space import Categorical,Real
from functools import partial
from skopt.plots import _cat_format

def compute_ALE(data,model,feat,space,samples,name,include_CI=False, C=0.95):
    #d1 = pd.DataFrame(space.inverse_transform(samples),columns=[n for n in name])


    if data[feat].dtype in ['int','float']:
        # data = data.drop(columns=feat)
        # data[feat] = d1[feat]  
        ale_eff = ale(X=data, model=model, feature=[feat],plot=False, grid_size=50, include_CI=True, C=0.95)
        return ale_eff
    else:
        ale_eff = ale(X=data, model=model, feature=[feat],plot=False, grid_size=50,predictors=data.columns.tolist(), include_CI=True, C=0.95)
        return ale_eff
        

    
    
    