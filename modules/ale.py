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
        


def plot_ALE(data,model,space,samples,name,plot_dims):

    n_dims = len(plot_dims)
    fig, ax = plt.subplots(n_dims, 1,
                           figsize=(2 * n_dims, 3 * n_dims))

    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                        hspace=0.5, wspace=0.1)
    
    for i,feat in enumerate(name):
        index, dim = plot_dims[i]
        iscat = [isinstance(dim[1], Categorical) for dim in plot_dims]
        ax_ = ax[i]
        ale_eff = compute_ALE(data,model,feat,space,samples,name,include_CI=False, C=0.95)
    
        if not iscat[i]:
            #ax_.set_xscale('log')    
            sample = space.rvs(n_samples=51)#grid_size +1
            xi = space.transform(sample)
            xi[:,0] = ale_eff.index.values
            xi = space.inverse_transform(xi)
            ax_.plot(np.array(xi)[:,0].astype(float),ale_eff['eff'])
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
    
    
    