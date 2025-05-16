import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import FastICA
from sklearn.feature_selection import GenericUnivariateSelect, mutual_info_regression, f_regression
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from addmo.s1_data_tuning_auto.config.data_tuning_auto_config import DataTuningAutoSetup


def manual_feature_select(config: DataTuningAutoSetup, x):
    """
    Manual selection of features
    """
    return x[config.selected_features]


def filter_low_variance(config: DataTuningAutoSetup, x):
    """
    Pre-Filter removing features with low variance.
    For documentation see scikit-learning.org.
    """
    filter = VarianceThreshold(threshold=config.low_variance_threshold).fit(
        X=x
    ).set_output(transform="pandas")  # fit filter
    x_processed = filter.transform(X=x) # transform the system_data
    return x_processed


def filter_ica(x):
    """
    Filter Independent Component Analysis (ICA)
    """
    Ica = FastICA(max_iter=1000)
    x_transformed = Ica.fit_transform(X=x)
    x_processed = pd.DataFrame(x_transformed, columns=x.columns, index=x.index)
    return x_processed


def filter_univariate(config: DataTuningAutoSetup, x, y):
    """
    Filter univariate with scoring function f-test or mutual information
    and search mode : {‘percentile’, ‘k_best’, ‘fpr’, ‘fdr’, ‘fwe’}
    For documentation see scikit-learning.org.
    """
    score_function_map = {
        "mutual_info_regression": mutual_info_regression,
        "f_regression": f_regression,
    }

    score_func = score_function_map.get(config.univariate_score_function)
    if score_func is None:
        raise ValueError(
            f"Invalid score function '{config.univariate_score_function}'. "
            "Must be one of: 'mutual_info_regression', 'f_regression'."
        )

    selector = GenericUnivariateSelect(
        score_func=score_func,
        mode=config.univariate_search_mode,
        param=config.univariate_filter_params,
    ).set_output(transform="pandas")

    selector = selector.fit(X=x, y=y)
    x_processed = selector.transform(X=x)
    return x_processed

# embedded Feature Selection by recursive feature elimination (Feature Subset Selection, multivariate)
def recursive_feature_selection_by_count(config: DataTuningAutoSetup, x, y):
    """
    Embedded Feature Selection by recursive feature elimination (multivariate) based on the number of features to select.
    For documentation see scikit-learning.org.
    """

    model = RandomForestRegressor(random_state=42)
    min_features_to_select = config.recursive_embedded_number_features_to_select

    n_features = x.shape[1]
    current_features = list(range(n_features))

    while len(current_features) > min_features_to_select:
        selector = RFE(estimator=model, n_features_to_select=len(current_features))
        selector = selector.fit(x.iloc[:, current_features], y)

        # Optional: print CV score just for info, but no stopping condition
        scores = cross_val_score(model, x.iloc[:, current_features], y, cv=5, scoring='r2')
        mean_score = np.mean(scores)
        print(f"Features: {len(current_features)}, CV Score: {mean_score:.4f}")

        ranking = selector.ranking_
        least_important_feature = np.where(ranking == max(ranking))[0][0]
        current_features.pop(least_important_feature)

    print(f"Selected {len(current_features)} features after recursive elimination.")
    return x.iloc[:, current_features]


def recursive_feature_selection_by_score(config: DataTuningAutoSetup, x, y):
    """
    Recursive feature elimination based on score improvement.
    Stops when cross-validation score increase falls below the configured threshold.
    """

    model = RandomForestRegressor(random_state=42)
    min_increase = config.min_increase_for_wrapper

    n_features = x.shape[1]
    current_features = list(range(n_features))
    last_score = -np.inf
    best_features = current_features.copy()
    best_score = last_score

    while len(current_features) > 1:  # Stop when only one feature left
        selector = RFE(estimator=model, n_features_to_select=len(current_features))
        selector = selector.fit(x.iloc[:, current_features], y)

        # Evaluate with cross-validation
        scores = cross_val_score(model, x.iloc[:, current_features], y, cv=5, scoring='r2')
        mean_score = np.mean(scores)
        print(f"Features: {len(current_features)}, CV Score: {mean_score:.4f}")

        score_improvement = mean_score - last_score

        # Stop if score improvement is too small
        if score_improvement < min_increase:
            print("Score improvement below threshold. Stopping.")
            break

        # Update best set
        best_score = mean_score
        best_features = current_features.copy()
        last_score = mean_score

        # Eliminate features
        ranking = selector.ranking_
        least_important_feature = np.where(ranking == max(ranking))[0][0]
        current_features.pop(least_important_feature)

    print(f"Selected {len(best_features)} features with best CV score: {best_score:.4f}")
    return x.iloc[:, best_features]