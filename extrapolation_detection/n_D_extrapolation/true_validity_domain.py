import pandas as pd


def infer_threshold(outlier_fraction: float, errors_train_val_test: pd.Series) -> float:
    '''Infer the absolute threshold from the fraction of outliers.'''
    absolute_threshold = errors_train_val_test.quantile(1 - outlier_fraction)
    return absolute_threshold

def classify_errors_2_true_validity(errors: pd.Series, absolute_validity_threshold: float):
    '''The regressor error is used to classify the data points to be in the true validity domain
    or not
    absolute error in respective error metric
    '''
    true_validity = errors.gt(absolute_validity_threshold)

    # convert bool to 0 and 1
    true_validity = true_validity.astype(int)

    return true_validity