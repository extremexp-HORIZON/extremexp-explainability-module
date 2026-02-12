"""
Correlation and Relationship Measurement Utilities
===================================================
Provides unified functions for measuring relationships between variables
of different types using appropriate statistical measures:
- Kendall correlation for continuous-continuous
- Partial eta squared for categorical-continuous
- Mutual information for categorical-categorical
"""

import pandas as pd
import numpy as np
from scipy.stats import kendalltau
from sklearn.preprocessing import LabelEncoder


def compute_kendall_correlation(x, y):
    """
    Compute Kendall tau correlation for continuous-continuous relationships.
    More robust to outliers than Pearson correlation.
    
    Args:
        x: First continuous variable (array-like)
        y: Second continuous variable (array-like)
    
    Returns:
        Kendall tau correlation coefficient (-1 to 1)
    """
    tau, _ = kendalltau(x, y)
    # return np.abs(tau)
    return tau


def compute_partial_eta_squared(continuous, categorical):
    """
    Compute partial eta squared (η²) for categorical-continuous relationship.
    Measures effect size of categorical variable on continuous variable.
    
    Formula: η² = SS_between / SS_total
    
    Args:
        continuous: Continuous variable values (array-like)
        categorical: Categorical variable values (array-like)
    
    Returns:
        Square root of eta squared as correlation-like measure (0 to 1)
    """
    # Get unique categories
    categories = np.unique(categorical)
    
    # Compute overall mean
    grand_mean = np.mean(continuous)
    ss_total = np.sum((continuous - grand_mean) ** 2)
    
    # Compute between-group sum of squares
    ss_between = 0
    for cat in categories:
        group_data = continuous[categorical == cat]
        group_mean = np.mean(group_data)
        ss_between += len(group_data) * (group_mean - grand_mean) ** 2
    
    # Partial eta squared
    eta_squared = ss_between / ss_total if ss_total > 0 else 0
    return np.sqrt(eta_squared)  # Return as correlation-like measure (0-1)


def compute_mutual_information(x, y, n_bins=10):
    """
    Compute normalized mutual information for categorical-categorical relationships.
    Measures how much knowing one variable reduces uncertainty about the other.
    
    Args:
        x: First variable (array-like, categorical or will be discretized)
        y: Second variable (array-like, categorical or will be discretized)
        n_bins: Number of bins for discretizing continuous variables
    
    Returns:
        Normalized mutual information (0 to 1)
    """
    # Convert to numpy arrays if needed
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Discretize continuous variables if needed
    if x.dtype not in ['object', 'int32', 'int64']:
        x = pd.cut(x, bins=n_bins, labels=False, duplicates='drop').values
    if y.dtype not in ['object', 'int32', 'int64']:
        y = pd.cut(y, bins=n_bins, labels=False, duplicates='drop').values
    
    # Encode categorical if needed
    if len(x) > 0 and isinstance(x[0], str):
        le_x = LabelEncoder()
        x = le_x.fit_transform(x)
    if len(y) > 0 and isinstance(y[0], str):
        le_y = LabelEncoder()
        y = le_y.fit_transform(y)
    
    # Compute contingency table
    contingency = pd.crosstab(x, y)
    
    # Compute mutual information
    pxy = contingency / contingency.sum().sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    
    mi = 0
    for i in px.index:
        for j in py.index:
            if pxy.loc[i, j] > 0:
                mi += pxy.loc[i, j] * np.log(pxy.loc[i, j] / (px[i] * py[j]))
    
    # Normalize by maximum possible MI
    hx = -np.sum(px * np.log(px + 1e-10))
    hy = -np.sum(py * np.log(py + 1e-10))
    normalized_mi = mi / min(hx, hy) if min(hx, hy) > 0 else 0
    
    return np.abs(normalized_mi)


def detect_variable_types(X_df):
    """
    Detect whether each variable is continuous or categorical.
    
    Classification:
    - Categorical: object dtype, or numeric with < 10 unique values
    - Continuous: numeric with >= 10 unique values
    
    Args:
        X_df: Input DataFrame
    
    Returns:
        var_types: Dictionary mapping column names to 'continuous' or 'categorical'
    """
    var_types = {}
    for col in X_df.columns:
        if X_df[col].dtype == 'object':
            var_types[col] = 'categorical'
        # elif X_df[col].nunique() < 10:
        #     var_types[col] = 'categorical'
        else:
            var_types[col] = 'continuous'
    return var_types


def compute_relationship_measure(X_df, feature1, feature2, var_types):
    """
    Compute relationship measure based on variable types.
    
    Uses appropriate statistical measure:
    - Kendall tau correlation for continuous-continuous
    - Partial eta squared for categorical-continuous
    - Mutual information for categorical-categorical
    
    Following paper methodology: "The relationship between variables is measured 
    using Kendall's correlation coefficient, partial eta square, and mutual information,
    as it is done consistently throughout the methodology, depending on the types of variables."
    
    Args:
        X_df: Input DataFrame
        feature1: Name of first feature/column
        feature2: Name of second feature/column
        var_types: Dictionary mapping feature names to variable types
    
    Returns:
        Tuple of (relationship_strength, measure_type)
        - relationship_strength: Numeric value (0 to 1)
        - measure_type: String indicating which measure was used
    """
    type1 = var_types[feature1]
    type2 = var_types[feature2]
    
    x = X_df[feature1].values
    y = X_df[feature2].values

    # Both continuous: Kendall tau correlation
    if type1 == 'continuous' and type2 == 'continuous':
        return compute_kendall_correlation(x, y), 'Kendall'
    
    # One categorical, one continuous: Partial eta squared
    elif (type1 == 'categorical' and type2 == 'continuous') or \
         (type1 == 'continuous' and type2 == 'categorical'):
        cat_var = x if type1 == 'categorical' else y
        cont_var = y if type1 == 'categorical' else x
        return compute_partial_eta_squared(cont_var, cat_var), 'Partial η²'
    
    # Both categorical: Mutual information
    else:
        return compute_mutual_information(x, y), 'MI'


def compute_relationship_matrix(X, feature_names):
    """
    Compute relationship matrix between all pairs of features.
    
    Uses appropriate measures based on variable types for each pair.
    
    Args:
        X: Data matrix (can be array or DataFrame)
        feature_names: List of feature names
    
    Returns:
        relationship_matrix: DataFrame with relationship strengths between all feature pairs
    """
    # Create dataframe for easier type detection
    if not isinstance(X, pd.DataFrame):
        X_df = pd.DataFrame(X, columns=feature_names)
    else:
        X_df = X
    
    var_types = detect_variable_types(X_df)
    
    # Compute relationship matrix using appropriate measures
    relationship_matrix = pd.DataFrame(0.0, index=feature_names, columns=feature_names)
    
    for i, feat1 in enumerate(feature_names):
        for j, feat2 in enumerate(feature_names):
            if i < j:  # Only upper triangle to avoid duplicates
                try:
                    measure, measure_type = compute_relationship_measure(X_df, feat1, feat2, var_types)
                    relationship_matrix.loc[feat1, feat2] = measure
                    relationship_matrix.loc[feat2, feat1] = measure
                except Exception as e:
                    # On error, set to 0
                    relationship_matrix.loc[feat1, feat2] = 0
                    relationship_matrix.loc[feat2, feat1] = 0
    
    return relationship_matrix


def compute_correlations_with_components(X_original, components, component_names, correlation_threshold=0.5):
    """
    Compute correlations between original numerical variables and derived components.
    Identifies variables with correlation > threshold for component naming.
    
    Args:
        X_original: Original numerical variables (array or DataFrame)
        components: Derived component matrix (n_samples x n_components)
        component_names: Names of derived components
        correlation_threshold: Absolute correlation threshold for relevance
    
    Returns:
        correlations_dict: Dictionary mapping component names to relevant original variables
                          with their correlation values, sorted by absolute correlation
    """
    correlations = np.corrcoef(X_original.T, components.T)
    
    # Extract correlations between original and derived variables
    n_original = X_original.shape[1]
    n_components = components.shape[1]
    
    correlation_matrix = correlations[:n_original, n_original:]
    
    correlations_dict = {}
    for comp_idx, comp_name in enumerate(component_names):
        relevant_vars = []
        for var_idx, corr_value in enumerate(correlation_matrix[:, comp_idx]):
            if abs(corr_value) > correlation_threshold:
                relevant_vars.append({
                    'variable': X_original.columns[var_idx] if hasattr(X_original, 'columns') else f'var_{var_idx}',
                    'correlation': corr_value
                })
        correlations_dict[comp_name] = sorted(relevant_vars, key=lambda x: abs(x['correlation']), reverse=True)
    
    return correlations_dict


def compute_eta_squared_with_components(X_cat_original, components, component_names, eta_threshold=0.14):
    """
    Compute eta-squared values between categorical variables and derived components.
    Identifies categorical variables with eta-squared > threshold.
    
    Args:
        X_cat_original: Original categorical variables (DataFrame)
        components: Derived component matrix (n_samples x n_components)
        component_names: Names of derived components
        eta_threshold: Eta-squared threshold for relevance
    
    Returns:
        eta_dict: Dictionary mapping component names to relevant categorical variables
                 with their eta-squared values, sorted by eta-squared
    """
    eta_dict = {}
    
    for comp_idx, comp_name in enumerate(component_names):
        component_values = components[:, comp_idx]
        relevant_cats = []
        
        for cat_col in X_cat_original.columns:
            # Calculate eta-squared: sum of squared between-group deviations / total sum of squared deviations
            categories = X_cat_original[cat_col].unique()
            
            grand_mean = component_values.mean()
            ss_total = np.sum((component_values - grand_mean) ** 2)
            
            ss_between = 0
            for category in categories:
                mask = X_cat_original[cat_col] == category
                n_cat = mask.sum()
                if n_cat > 0:
                    cat_mean = component_values[mask].mean()
                    ss_between += n_cat * (cat_mean - grand_mean) ** 2
            
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            if eta_squared > eta_threshold:
                relevant_cats.append({
                    'variable': cat_col,
                    'eta_squared': eta_squared
                })
        
        eta_dict[comp_name] = sorted(relevant_cats, key=lambda x: x['eta_squared'], reverse=True)
    
    return eta_dict
