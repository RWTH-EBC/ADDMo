import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import FastICA
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import RFE, RFECV
from sklearn.feature_selection import SequentialFeatureSelector

from core.util.data_handling import split_target_features
from core.data_tuning.wrapper_model import DataTunerWrapperModel
from core.model_tuning.models.model_factory import ModelFactory

from core.data_tuning_auto.config.data_tuning_auto_config import DataTuningAutoSetup

def manual_feature_select(config: DataTuningAutoSetup, x):
    '''Manual selection of features'''
    return x[config.selected_features]

def filter_low_variance(config: DataTuningAutoSetup, x):
    '''Pre-Filter removing features with low variance.
    For documentation see scikit-learning.org.'''
    filter = VarianceThreshold(
        threshold=config.low_variance_threshold
    ).fit(X=x)  # fit filter
    x_processed = filter.transform(X=x)  # transform the data
    return x_processed

def filter_ica(x):
    '''Filter Independent Component Analysis (ICA)'''
    Ica = FastICA(max_iter=1000)
    x_processed = Ica.fit_transform(X=x)
    return x_processed

def filter_univariate(config: DataTuningAutoSetup, x, y):
    '''Filter univariate
    with scoring function f-test or mutual information
    and search mode : {‘percentile’, ‘k_best’, ‘fpr’, ‘fdr’, ‘fwe’}
    For documentation see scikit-learning.org.'''
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
    '''Embedded Feature Selection by recursive feature elemination (multivariate)
    For documentation see scikit-learning.org.'''
    if config.recursive_embedded_number_features_to_select == 0:
        selector = RFECV(
            estimator=config.embedded_model,
            step=1,
            cv=config.recursive_embedded_scoring,
        )
    else:
        selector = RFE(
            estimator=config.embedded_model,
            n_features_to_select=config.recursive_embedded_number_features_to_select,
            step=1,
        )
    selector = selector.fit(x, y)
    print("Ranks of all Features %s" % selector.ranking_)
    x_processed = selector.transform(X)

    return x_processed

def recursive_feature_selection_wrapper(config: DataTuningAutoSetup, x,
                                                  y) -> pd.DataFrame:
    '''
    Filter using Sequential Feature Selection.
    This method selects features based on the resulting score of the estimator.
    For documentation see scikit-learn.org.
    '''


    x_created = pd.DataFrame()

    Wrapper = DataTunerWrapperModel(config)

    # prepare data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    old_score = Wrapper.train_score(x_train, x_test, y_train, y_test)

    # loop through to create lags as long as they improve the result
    for i in range(config.minimum_target_lag, len(x_train)):
        series = feature_constructor.create_lag(y, i)
        x_processed = pd.concat([x, series], axis=1, join="inner")

        # redo identical splitting
        x_train_processed = x_processed.loc[x_train.index]
        x_test_processed = x_processed.loc[x_test.index]

        new_score = Wrapper.train_score(x_train_processed, x_test_processed, y_train,
                                        y_test)

        if new_score <= old_score + config.min_increase_4_wrapper:
            break
        else:
            x_created[series.name] = series
            old_score = new_score

    return x_created

def recursive_feature_selection_wrapper_scikit_learn(config: DataTuningAutoSetup, x,
                                                  y) -> pd.DataFrame:

    # get model from the factory
    wrapper_model = ModelFactory.model_factory(config.wrapper_model)

    selector = SequentialFeatureSelector(
        estimator=wrapper_model, **config.filter_sequential_kwargs)\

    selector.fit(X=x, y=y)  # fit the selector
    x_processed = selector.transform(X=x)  # transform the data
    return x_processed