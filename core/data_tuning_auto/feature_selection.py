import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import FastICA
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import RFE, RFECV
from sklearn.feature_selection import SequentialFeatureSelector

from core.util.data_handling import split_target_features
from core.model_tuning.scoring.validator_factory import ValidatorFactory
from core.model_tuning.models.model_factory import ModelFactory
from core.model_tuning.scoring.abstract_scorer import ValidationScoring
from core.model_tuning.models.abstract_model import AbstractMLModel
from core.data_tuning_auto.config.data_tuning_auto_config import DataTuningAutoSetup


def manual_feature_select(config: DataTuningAutoSetup, x):
    """Manual selection of features"""
    return x[config.selected_features]


def filter_low_variance(config: DataTuningAutoSetup, x):
    """Pre-Filter removing features with low variance.
    For documentation see scikit-learning.org."""
    filter = VarianceThreshold(threshold=config.low_variance_threshold).fit(
        X=x
    )  # fit filter
    x_processed = filter.transform(X=x)  # transform the data
    return x_processed


def filter_ica(x):
    """Filter Independent Component Analysis (ICA)"""
    Ica = FastICA(max_iter=1000)
    x_processed = Ica.fit_transform(X=x)
    return x_processed


def filter_univariate(config: DataTuningAutoSetup, x, y):
    """Filter univariate
    with scoring function f-test or mutual information
    and search mode : {‘percentile’, ‘k_best’, ‘fpr’, ‘fdr’, ‘fwe’}
    For documentation see scikit-learning.org."""
    filter = GenericUnivariateSelect(
        score_func=config.univariate_score_function,
        mode=config.univariate_search_mode,
        param=config.univariate_filter_params,
    )
    filter = filter.fit(X=x, y=y)
    x_processed = filter.transform(X=x)
    return x_processed


# embedded Feature Selection by recursive feature elemination (Feature Subset Selection, multivariate)
def recursive_feature_selection_embedded(config: DataTuningAutoSetup, x, y):
    """Embedded Feature Selection by recursive feature elemination (multivariate)
    For documentation see scikit-learning.org."""

    white_list_of_possible_models = ["Scikit_RF", "Scikit_Lasso"]
    # currently only supporting scikit-learn models
    if config.embedded_model not in white_list_of_possible_models:
        raise ValueError(
            "The model you chose is not supported for embedded feature selection. "
            "Please choose one of the following models: %s"
            % white_list_of_possible_models
        )

    model: AbstractMLModel = ModelFactory.model_factory(config.wrapper_model)

    scikit_model = model.model

    if config.recursive_embedded_number_features_to_select == 0:
        selector = RFECV(estimator=scikit_model)
    else:
        selector = RFE(estimator=scikit_model)
    selector = selector.fit(x, y)
    print("Ranks of all Features %s" % selector.ranking_)
    x_processed = selector.transform(x)

    return x_processed

def recursive_feature_selection_wrapper_scikit_learn(config: DataTuningAutoSetup, x, y) -> pd.DataFrame:
    # get model from the factory
    wrapper_model = ModelFactory.model_factory(config.wrapper_model)

    selector = SequentialFeatureSelector(
        estimator=wrapper_model, **config.filter_sequential_kwargs
    )
    selector.fit(X=x, y=y)  # fit the selector
    x_processed = selector.transform(X=x)  # transform the data
    return x_processed


# def custom_forward_feature_selector(config: DataTuningAutoSetup, xy) -> pd.DataFrame:
#     '''
#     Custom Sequential Feature Selector supporting other scoring functions.
#     This method selects features based on custom scoring functions.
#
#     Parameters:
#     config: DataTuningAutoSetup - configuration object with selection parameters
#     xy: pd.DataFrame - combined feature and target data
#
#     Returns:
#     x_selected: pd.DataFrame - selected feature data after feature selection
#     '''
#     x_selected = pd.DataFrame()
#
#     scorer: ValidationScoring = ScoringFactory.get_splitter(config.scoring_split_technique)
#     model: AbstractMLModel = ModelFactory.model_factory(config.wrapper_model)
#
#     # Splitting features and target
#     x, y = split_target_features(config.name_of_target, xy)
#
#     # Initialize score
#     old_score = scorer.score_validation(model, config.scoring_metric, x, y)
#
#     # Feature selection loop
#     for i in x.columns:
#         temp_score = old_score
#         for feature in x.columns:
#             # Add the feature for testing
#             x_temp = pd.concat([x_selected, x[[feature]]], axis=1)
#
#             # Calculate new score with the added feature
#             new_score = scorer.score_validation(model, config.scoring_metric, x_temp, y)
#
#
#
#             # choose the best feature for that iteration
#             if new_score > temp_score:
#                 temp_score = new_score
#                 x_best_lag = series
#
#             # Add feature permanently if score improves
#             if new_score > old_score:
#                 x_selected[feature] = x[feature]
#                 old_score = new_score
#
#     return x_selected


def forward_feature_selector(config: DataTuningAutoSetup, xy) -> pd.DataFrame:
    '''
    Forward Sequential Feature Selector.
    This method selects features based on custom scoring functions in a forward manner.

    Parameters:
    config: DataTuningAutoSetup - configuration object with selection parameters
    xy: pd.DataFrame - combined feature and target data

    Returns:
    x_selected: pd.DataFrame - selected feature data after feature selection
    '''
    x_selected = pd.DataFrame()

    scorer: ValidationScoring = ValidatorFactory.ValidatorFactory(config.scoring_split_technique)
    model: AbstractMLModel = ModelFactory.model_factory(config.wrapper_model)

    x, y = split_target_features(config.name_of_target, xy)

    features = set(x.columns)
    selected_features = set()
    old_score = scorer.score_validation(model, config.scoring_metric, x_selected, y)

    while True:
        score_changes = {}
        for feature in features - selected_features:
            x_test = pd.concat([x_selected, x[[feature]]], axis=1)
            new_score = scorer.score_validation(model, config.scoring_metric, x_test, y)
            score_changes[feature] = new_score - old_score

        best_feature = max(score_changes, key=score_changes.get, default=None)
        improvement = score_changes.get(best_feature, 0) > 0
        if not improvement:
            break

        x_selected[best_feature] = x[best_feature]
        selected_features.add(best_feature)
        old_score = scorer.score_validation(model, config.scoring_metric, x_selected, y)

    return x_selected

