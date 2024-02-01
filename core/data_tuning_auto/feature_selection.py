import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import FastICA
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import RFE, RFECV
from sklearn.feature_selection import SequentialFeatureSelector


from core.model_tuning.models.model_factory import ModelFactory
from core.model_tuning.scoring.metrics.metric_factory import MetricFactory
from core.model_tuning.scoring.validation_splitting.splitter_factory import (
    SplitterFactory,
)
from core.model_tuning.scoring.validator_factory import ValidatorFactory
from core.model_tuning.hyperparameter_tuning.hyparam_tuning_factory import HyperparameterTunerFactory
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
    ).set_output(transform="pandas")  # fit filter
    x_processed = filter.transform(X=x) # transform the data
    return x_processed


def filter_ica(x):
    """Filter Independent Component Analysis (ICA)"""
    Ica = FastICA(max_iter=1000).set_output(transform="pandas")
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
    ).set_output(transform="pandas")
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
        selector = RFECV(estimator=scikit_model).set_output(transform="pandas")
    else:
        selector = RFE(estimator=scikit_model).set_output(transform="pandas")
    selector = selector.fit(x, y)
    print("Ranks of all Features %s" % selector.ranking_)
    x_processed = selector.transform(x)

    return x_processed


def recursive_feature_selection_wrapper_scikit_learn(
    config: DataTuningAutoSetup, x, y
) -> pd.DataFrame:

    # erfolgloser Versuch, die Wrapper Funktionen von scikit-learn zu nutzen und gleichzeitig die
    # Modelle hyperparameter zu tunen, vermutlich am besten einen eigenen Wrapper schreiben
    # alternativ die feature selection mit in optuna integrieren?
    model = ModelFactory.model_factory(config.model)

    metric = MetricFactory.metric_factory(
        config.validation_score_metric, config.validation_score_metric_kwargs
    )
    splitter = SplitterFactory.splitter_factory(config)

    scorer = ValidatorFactory.ValidatorFactory(config)

    tuner = HyperparameterTunerFactory.tuner_factory(config, model, scorer)

    def tuning():
        best_params = tuner.tune(x, y, **config.hyperparameter_tuning_kwargs)
        model.set_params(best_params)
        return model.to_scikit_learn()


    selector = SequentialFeatureSelector(
        estimator=model.to_scikit_learn(), cv=splitter, scoring=metric,
        tol=config.min_increase_4_wrapper, direction=config.sequential_direction
    ).set_output(transform="pandas")
    selector.fit(X=x, y=y)  # fit the selector
    x_processed = selector.transform(X=x)  # transform the data
    return x_processed
