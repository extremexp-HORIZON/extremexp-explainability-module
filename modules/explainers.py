import json
import numpy as np
from modules.lib_IF import *
from modules.lib import *
from modules.pdp import partial_dependence_1D,partial_dependence_2D
from modules.ale import *

def ComputePDP(param_grid, model, feature):

        param_grid = transform_grid(param_grid)
        param_space, name = dimensions_aslists(param_grid)
        space = Space(param_space)
        feats = {}
        for index,n in enumerate(name):
            feats[n] = index

        plot_dims = []
        for row in range(space.n_dims):
            if space.dimensions[row].is_constant:
                continue
            plot_dims.append((row, space.dimensions[row]))
            
        # pdp_samples = space.transform(space.rvs(n_samples=1000,random_state=123456))
        pdp_samples = space.rvs(n_samples=1000,random_state=123456)

        # x_vals = list(param_grid.keys())

        # n_dims = len(plot_dims)
        xi = []
        yi=[]
        index, dim = plot_dims[feats[feature]]
        xi1, yi1 = partial_dependence_1D(space, model,
                                            index,
                                            samples=pdp_samples,
                                            n_points=100)
            # if isinstance(dim,Categorical):
            #      xi1 = np.array(space.dimensions[i].categories)
        xi.append(xi1)
        yi.append(yi1)
            
        x = [arr.tolist() for arr in xi]
        y = [arr for arr in yi]

        return x,y

def ComputePDP2D(param_grid, model,feature1,feature2):

        param_grid = transform_grid(param_grid)
        param_space, name = dimensions_aslists(param_grid)
        space = Space(param_space)

        index1 = name.index(feature1)
        index2 = name.index(feature2)


        plot_dims = []
        for row in range(space.n_dims):
            if space.dimensions[row].is_constant:
                continue
            plot_dims.append((row, space.dimensions[row]))
        
        # pdp_samples = space.transform(space.rvs(n_samples=1000,random_state=123456))
        pdp_samples = space.rvs(n_samples=1000,random_state=123456)

        _ ,dim_1 = plot_dims[index1]
        _ ,dim_2 = plot_dims[index2]
        xi, yi, zi = partial_dependence_2D(space, model,
                                                   index1, index2,
                                                   pdp_samples, 100)
        # if isinstance(dim_1,Categorical):
        #          xi = np.array(space.dimensions[index1].categories)

        # if isinstance(dim_2,Categorical):
        #          yi = np.array(space.dimensions[index2].categories)
        
        
        x = [arr.tolist() for arr in xi]
        y = [arr.tolist() for arr in yi]
        z = [arr.tolist() for arr in zi]
       
        return x,y,z
    
def ComputeALE(param_grid, model,feature):

        param_grid = transform_grid(param_grid)
        param_space, name = dimensions_aslists(param_grid)
        space = Space(param_space)

        plot_dims = []
        for row in range(space.n_dims):
            if space.dimensions[row].is_constant:
                continue
            plot_dims.append((row, space.dimensions[row]))

        pdp_samples = space.rvs(n_samples=1000,random_state=123456)
        data = pd.DataFrame(pdp_samples,columns=[n for n in name])

        dataframes_list = []

        
        ale_eff = compute_ALE(data,model,feature,space,pdp_samples,name,include_CI=False, C=0.95)
            # sample = space.rvs(n_samples=len(ale_eff))
            # xi = space.transform(sample)
            # xi[:,i] = ale_eff.index.values
            # xi = space.inverse_transform(xi)
            # feature_names = [x[i] for x in xi]
            # ale_eff.reset_index(inplace=True)
            # ale_eff[feat] = feature_names
            # ale_eff.set_index(feat,inplace=True)

        dataframes_list.append(ale_eff)

        d = json.dumps([df.to_json(orient='split') for df in dataframes_list])

        return d