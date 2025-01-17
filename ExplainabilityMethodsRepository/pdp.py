import numpy as np
from modules.lib import _evenly_sample
import pandas as pd 
from skopt.space import Categorical

def partial_dependence_1D(space, model, i, samples,name,
                          n_points=100):
    """
    Calculate the partial dependence for a single dimension.

    This uses the given model to calculate the average objective value
    for all the samples, where the given dimension is fixed at
    regular intervals between its bounds.

    This shows how the given dimension affects the objective value
    when the influence of all other dimensions are averaged out.

    Parameters
    ----------
    space : `Space`
        The parameter space over which the minimization was performed.

    model
        Surrogate model for the objective function.

    i : int
        The dimension for which to calculate the partial dependence.

    samples : np.array, shape=(n_points, n_dims)
        Randomly sampled and transformed points to use when averaging
        the model function at each of the `n_points` when using partial
        dependence.

    n_points : int, default=40
        Number of points at which to evaluate the partial dependence
        along each dimension `i`.

    Returns
    -------
    xi : np.array
        The points at which the partial dependence was evaluated.

    yi : np.array
        The average value of the modelled objective function at
        each point `xi`.

    """
    # The idea is to step through one dimension, evaluating the model with
    # that dimension fixed and averaging either over random values or over
    # the given ones in x_val in all other dimensions.
    # (Or step through 2 dimensions when i and j are given.)
    # Categorical dimensions make this interesting, because they are one-
    # hot-encoded, so there is a one-to-many mapping of input dimensions
    # to transformed (model) dimensions.

    # dim_locs[i] is the (column index of the) start of dim i in
    # sample_points.
    # This is usefull when we are using one hot encoding, i.e using
    # categorical values

    def _calc(x):
        """
        Helper-function to calculate the average predicted
        objective value for the given model, when setting
        the index'th dimension of the search-space to the value x,
        and then averaging over all samples.
        """
        rvs_ = pd.DataFrame(data=samples)  # copy
        # We replace the values in the dimension that we want to keep
        # fixed
        rvs_[i] = x
        rvs_.columns = name
        # In case of `x_eval=None` rvs conists of random samples.
        # Calculating the mean of these samples is how partial dependence
        # is implemented.
        return np.mean(model.predict(rvs_).astype(float))
    
    if isinstance(space.dimensions[i],Categorical):
        xi = np.array(space.dimensions[i].categories)
    else:
        if pd.DataFrame(samples)[i].nunique() <= n_points:
            xi = np.sort(pd.DataFrame(samples)[i].unique())
        else:
            xi = _evenly_sample(space.dimensions[i], 100)    # Calculate the partial dependence for all the points.
    yi = [_calc(x) for x in xi]

    return xi, yi


def partial_dependence_2D(space, model, i, j, samples,name,
                          n_points=100):
    """
    Calculate the partial dependence for two dimensions in the search-space.

    This uses the given model to calculate the average objective value
    for all the samples, where the given dimensions are fixed at
    regular intervals between their bounds.

    This shows how the given dimensions affect the objective value
    when the influence of all other dimensions are averaged out.

    Parameters
    ----------
    space : `Space`
        The parameter space over which the minimization was performed.

    model
        Surrogate model for the objective function.

    i : int
        The first dimension for which to calculate the partial dependence.

    j : int
        The second dimension for which to calculate the partial dependence.

    samples : np.array, shape=(n_points, n_dims)
        Randomly sampled and transformed points to use when averaging
        the model function at each of the `n_points` when using partial
        dependence.

    n_points : int, default=40
        Number of points at which to evaluate the partial dependence
        along each dimension `i` and `j`.

    Returns
    -------
    xi : np.array, shape=n_points
        The points at which the partial dependence was evaluated.

    yi : np.array, shape=n_points
        The points at which the partial dependence was evaluated.

    zi : np.array, shape=(n_points, n_points)
        The average value of the objective function at each point `(xi, yi)`.
    """
    # The idea is to step through one dimension, evaluating the model with
    # that dimension fixed and averaging either over random values or over
    # the given ones in x_val in all other dimensions.
    # (Or step through 2 dimensions when i and j are given.)
    # Categorical dimensions make this interesting, because they are one-
    # hot-encoded, so there is a one-to-many mapping of input dimensions
    # to transformed (model) dimensions.

    # dim_locs[i] is the (column index of the) start of dim i in
    # sample_points.
    # This is usefull when we are using one hot encoding, i.e using
    # categorical values

    def _calc(x, y):
        """
        Helper-function to calculate the average predicted
        objective value for the given model, when setting
        the index1'th dimension of the search-space to the value x
        and setting the index2'th dimension to the value y,
        and then averaging over all samples.
        """
        rvs_ = pd.DataFrame(samples)  # copy
        rvs_[j] = x
        rvs_[i] = y
        rvs_.columns = name
        return np.mean(model.predict(rvs_).astype(float))


    if isinstance(space.dimensions[j],Categorical):
        xi = np.array(space.dimensions[j].categories)
    else:
        if pd.DataFrame(samples)[j].nunique() <= n_points:  
            xi = np.sort(pd.DataFrame(samples)[j].unique())
        else:
            xi = _evenly_sample(space.dimensions[j], 40)
    if isinstance(space.dimensions[i],Categorical):
        yi = np.array(space.dimensions[i].categories)
    else:
        if pd.DataFrame(samples)[i].nunique() <= n_points:  
            yi = np.sort(pd.DataFrame(samples)[i].unique())
        else:
            yi = _evenly_sample(space.dimensions[i], 40)


    # Calculate the partial dependence for all combinations of these points.
    zi = [[_calc(x, y) for x in xi] for y in yi]

    # Convert list-of-list to a numpy array.
    zi = np.array(zi)

    return xi, yi, zi