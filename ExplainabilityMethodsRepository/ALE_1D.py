import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans

from PyALE._src.lib import quantile_ied, CI_estimate, order_groups


def aleplot_1D_continuous(X, model, feature, grid_size=20, include_CI=True, C=0.95):
    """Compute the accumulated local effect of a numeric continuous feature.

    This function divides the feature in question into grid_size intervals (bins)
    and computes the difference in prediction between the first and last value
    of each interval and then centers the results.

    Arguments:
    X -- A pandas DataFrame to pass to the model for prediction.
    model -- Any python model with a predict method that accepts X as input.
    feature -- String, the name of the column holding the feature being studied.
    grid_size -- An integer indicating the number of intervals into which the
    feature range is divided.
    include_CI -- A boolean, if True the confidence interval
    of the effect is returned with the results.
    C -- A float the confidence level for which to compute the confidence interval

    Return: A pandas DataFrame containing for each bin: the size of the sample in it
    and the accumulated centered effect of this bin.
    """

    quantiles = np.append(0, np.arange(1 / grid_size, 1 + 1 / grid_size, 1 / grid_size))
    # use customized quantile function to get the same result as
    # type 1 R quantile (Inverse of empirical distribution function)
    bins = [X[feature].min()] + quantile_ied(X[feature], quantiles).to_list()
    bins = np.unique(bins)
    feat_cut = pd.cut(X[feature], bins, include_lowest=True)

    bin_codes = feat_cut.cat.codes
    bin_codes_unique = np.unique(bin_codes)

    X1 = X.copy()
    X2 = X.copy()
    X1[feature] = [bins[i] for i in bin_codes]
    X2[feature] = [bins[i + 1] for i in bin_codes]
    try:
        y_1 = model.predict(X1).ravel()
        y_2 = model.predict(X2).ravel()
    except Exception as ex:
        raise Exception(
            "Please check that your model is fitted, and accepts X as input."
        )

    delta_df = pd.DataFrame({feature: bins[bin_codes + 1], "Delta": y_2 - y_1})
    res_df = delta_df.groupby([feature]).Delta.agg([("eff", "mean"), "size"])
    res_df["eff"] = res_df["eff"].cumsum()
    res_df.loc[min(bins), :] = 0
    # subtract the total average of a moving average of size 2
    mean_mv_avg = (
        (res_df["eff"] + res_df["eff"].shift(1, fill_value=0)) / 2 * res_df["size"]
    ).sum() / res_df["size"].sum()
    res_df = res_df.sort_index().assign(eff=res_df["eff"] - mean_mv_avg)
    if include_CI:
        ci_est = delta_df.groupby(feature).Delta.agg(
            [("CI_estimate", lambda x: CI_estimate(x, C=C))]
        )
        ci_est = ci_est.sort_index()
        lowerCI_name = "lowerCI_" + str(int(C * 100)) + "%"
        upperCI_name = "upperCI_" + str(int(C * 100)) + "%"
        res_df[lowerCI_name] = res_df[["eff"]].subtract(ci_est["CI_estimate"], axis=0)
        res_df[upperCI_name] = upperCI = res_df[["eff"]].add(
            ci_est["CI_estimate"], axis=0
        )
    return res_df


def aleplot_1D_discrete(X, model, feature, include_CI=True, C=0.95):
    """Compute the accumulated local effect of a numeric discrete feature.

    This function computes the difference in prediction when the value of the feature
    is replaced once with the value before it and once with the value after it, without
    the need to divide into interval like the case of aleplot_1D_continuous.

    Arguments:
    X -- A pandas DataFrame to pass to the model for prediction.
    model -- Any python model with a predict method that accepts X as input.
    feature -- String, the name of the column holding the feature being studied.
    include_CI -- A boolean, if True the confidence interval
    of the effect is returned with the results.
    C -- A float the confidence level for which to compute the confidence interval

    Return: A pandas DataFrame containing for each value of the feature: the size
    of the sample in it and the accumulated centered effect around this value.
    """

    groups = X[feature].unique()
    groups.sort()

    groups_codes = {groups[x]: x for x in range(len(groups))}
    feature_codes = X[feature].replace(groups_codes).astype(int)

    groups_counts = X.groupby(feature).size()
    groups_props = groups_counts / sum(groups_counts)

    K = len(groups)

    # create copies of the dataframe
    X_plus = X.copy()
    X_neg = X.copy()
    # all groups except last one
    last_group = groups[K - 1]
    ind_plus = X[feature] != last_group
    # all groups except first one
    first_group = groups[0]
    ind_neg = X[feature] != first_group
    # replace once with one level up
    X_plus.loc[ind_plus, feature] = groups[feature_codes[ind_plus] + 1]
    # replace once with one level down
    X_neg.loc[ind_neg, feature] = groups[feature_codes[ind_neg] - 1]
    try:
        # predict with original and with the replaced values
        y_hat = model.predict(X).ravel()
        y_hat_plus = model.predict(X_plus[ind_plus]).ravel()
        y_hat_neg = model.predict(X_neg[ind_neg]).ravel()
    except Exception as ex:
        raise Exception(
            "Please check that your model is fitted, and accepts X as input."
        )

    # compute prediction difference
    Delta_plus = y_hat_plus - y_hat[ind_plus]
    Delta_neg = y_hat[ind_neg] - y_hat_neg

    # compute the mean of the difference per group
    delta_df = pd.concat(
        [
            pd.DataFrame(
                {"eff": Delta_plus, feature: groups[feature_codes[ind_plus] + 1]}
            ),
            pd.DataFrame({"eff": Delta_neg, feature: groups[feature_codes[ind_neg]]}),
        ]
    )
    res_df = delta_df.groupby([feature]).mean()
    res_df["eff"] = res_df["eff"].cumsum()
    res_df.loc[groups[0]] = 0
    res_df = res_df.sort_index()
    res_df["eff"] = res_df["eff"] - sum(res_df["eff"] * groups_props)
    res_df["size"] = groups_counts
    if include_CI:
        ci_est = delta_df.groupby([feature]).eff.agg(
            [("CI_estimate", lambda x: CI_estimate(x, C=C))]
        )
        lowerCI_name = "lowerCI_" + str(int(C * 100)) + "%"
        upperCI_name = "upperCI_" + str(int(C * 100)) + "%"
        res_df[lowerCI_name] = res_df[["eff"]].subtract(ci_est["CI_estimate"], axis=0)
        res_df[upperCI_name] = upperCI = res_df[["eff"]].add(
            ci_est["CI_estimate"], axis=0
        )
    return res_df


def aleplot_1D_categorical(
    X, model, feature, predictors, include_CI=True, C=0.95
):
    """Compute the accumulated local effect of a categorical (or str) feature.

    The function computes the difference in prediction between the different groups
    similar to the function aleplot_1D_discrete.
    This function relies on an ordering of the unique values/groups of the feature,
    if the feature is not an ordered categorical, then an ordering is computed,
    which orders the groups by their similarity based on the distribution of the
    other features in each group.
    The function uses the given encoding function (for example a one-hot-encoding)
    and replaces the feature with the new generated (outputed) feature(s) before
    it calls the function model.predict.

    Arguments:
    X -- A pandas DataFrame containing all columns needed to pass to the model for
    prediction, however it should contain the original feature (before encoding)
    instead of the column(s) encoding it.
    model -- Any python model with a predict method that accepts X as input.
    feature -- String, the name of the column holding the feature being analysed.
    encode_fun -- Function, used to encode the categorical feature, usually a
    one-hot encoder. The function's input and output are as follows
        * input: a DataFrame with one column (the feature)
        * output: a DataFrame with the new column(s) encoding the feature.
        It is also important that this function should be able to handle missing
        categories (for example, a one-hot-encoder applied to a column, in which
        not all categories occur, should add a column of zeros for each missing
        category). Examples of use of this function could be found in the README
        file in github or in the description of the package in PyPI
        https://pypi.org/project/PyALE/.
    predictors -- List or array of strings containing the names of features used
    in the model, and in the right order.
    include_CI -- A boolean, if True the confidence interval of the effect is
    returned with the results.
    C -- A float the confidence level for which to compute the confidence interval

    Return: A pandas DataFrame containing for each value of the feature: the size
    of the sample in it and the accumulated centered effect around this value.
    """
    # if the values of the feature are not ordered, then order them by
    # their similarity to each other, based on the distributions of the other
    # features by group
    if (X[feature].dtype.name != "category") or (not X[feature].cat.ordered):
        X[feature] = X[feature].astype(str)

        # cast all columns that are object or string to category
        # for `order_groups` to work correctly
        X_temp = X.copy()
        for col in X_temp.columns:
            if X_temp[col].dtype.name in ["object", "string"]:
                X_temp[col] = X_temp[col].astype("category")

        groups_order = order_groups(X_temp, feature)
        groups = groups_order.index.values
        X[feature] = X[feature].astype(
            pd.api.types.CategoricalDtype(categories=groups, ordered=True)
        )

    groups = X[feature].unique()
    groups = groups.sort_values()
    feature_codes = X[feature].cat.codes
    groups_counts = X.groupby(feature).size()
    groups_props = groups_counts / sum(groups_counts)

    K = len(groups)

    # create copies of the dataframe
    X_plus = X.copy()
    X_neg = X.copy()
    # all groups except last one
    last_group = groups[K - 1]
    ind_plus = X[feature] != last_group
    # all groups except first one
    first_group = groups[0]
    ind_neg = X[feature] != first_group
    # replace once with one level up
    X_plus.loc[ind_plus, feature] = groups[feature_codes[ind_plus] + 1]
    # replace once with one level down
    X_neg.loc[ind_neg, feature] = groups[feature_codes[ind_neg] - 1]
    try:

        y_hat = model.predict(X[predictors]).ravel()

        y_hat_plus = model.predict(X_plus[ind_plus][predictors]).ravel()


        y_hat_neg = model.predict(X_neg[ind_neg][predictors]).ravel()
    except Exception as ex:
        raise Exception(
            """There seems to be a problem when predicting with the model.
            Please check the following: 
                - Your model is fitted.
                - The list of predictors contains the names of all the features"""
            """ used for training the model.
                - The encoding function takes the raw feature and returns the"""
            """ right columns encoding it, including the case of a missing category.
            """
        )

    # compute prediction difference
    Delta_plus = y_hat_plus - y_hat[ind_plus]
    Delta_neg = y_hat[ind_neg] - y_hat_neg

    # compute the mean of the difference per group
    delta_df = pd.concat(
        [
            pd.DataFrame(
                {"eff": Delta_plus, feature: groups[feature_codes[ind_plus] + 1]}
            ),
            pd.DataFrame({"eff": Delta_neg, feature: groups[feature_codes[ind_neg]]}),
        ]
    )
    res_df = delta_df.groupby([feature]).mean()
    res_df["eff"] = res_df["eff"].cumsum()
    res_df.loc[groups[0]] = 0
    # sort the index (which is at this point an ordered categorical) as a safety measure
    res_df = res_df.sort_index()
    res_df["eff"] = res_df["eff"] - sum(res_df["eff"] * groups_props)
    res_df["size"] = groups_counts
    if include_CI:
        ci_est = delta_df.groupby([feature]).eff.agg(
            [("CI_estimate", lambda x: CI_estimate(x, C=C))]
        )
        lowerCI_name = "lowerCI_" + str(int(C * 100)) + "%"
        upperCI_name = "upperCI_" + str(int(C * 100)) + "%"
        res_df[lowerCI_name] = res_df[["eff"]].subtract(ci_est["CI_estimate"], axis=0)
        res_df[upperCI_name] = upperCI = res_df[["eff"]].add(
            ci_est["CI_estimate"], axis=0
        )
    return res_df



