import pandas as pd
from pandas import DataFrame


def split_simulation_data(
    df: DataFrame,
    train_val_test_indices: list,
    val_fraction: float,
    test_fraction: float,
    shuffle: bool = False,
) -> dict:
    """Splits data in training, validation, test and remaining splits; last column of dataframe will be used as y

    Parameters
    ----------
    df: DataFrame
        data to be process
    train_val_test_indices: list
        list of indices to be used as training, validation and test split
    val_fraction: float
        relative fraction of data points from dataIndices to be used as validation split
    test_fraction: float
            relative fraction of data points from dataIndices to be used as test split
    random_state: int
        random state for shuffling
    shuffle: bool
        If set to True, training, validation and test datapoints will be shuffled

    Returns
    -------
    dict
        {'available_data': available_data, 'x_remaining', 'y_remaining', 'header'
    """

    # select list of indices used as training, validation and test split
    data = df.iloc[train_val_test_indices]

    # Shuffle data
    if shuffle:
        data_shuffled = data.sample(frac=1, random_state=1)
    else:
        data_shuffled = data

    # training split
    training_split = 1 - val_fraction - test_fraction
    xy_training = data_shuffled.iloc[0 : round(len(data_shuffled) * training_split)]

    # validation split
    xy_validation = data_shuffled.iloc[
        round(len(data_shuffled) * training_split) : round(
            len(data_shuffled) * (training_split + val_fraction)
        )
    ]

    # test split
    xy_test = data_shuffled.iloc[
        round(len(data_shuffled) * (training_split + val_fraction)) :
    ]

    # remaining split (not train/val/test)
    xy_remaining = df.drop(train_val_test_indices)

    return xy_training, xy_validation, xy_test, xy_remaining


def move_true_invalid_from_training_2_validation(
    xy_train, xy_val, true_validity_train, true_validity_val
):
    """Moves true invalid from training data to validation data"""

    # Convert 0s and 1s to False and True first for easier handling, as well df to series via squeeze
    true_validity_train = true_validity_train.astype(bool).squeeze()
    true_validity_val = true_validity_val.astype(bool).squeeze()

    # Select the "true invalid" data points from xy_train
    invalid_data = xy_train[~true_validity_train]
    invalid_labels = true_validity_train[~true_validity_train]

    # Remove the "true invalid" data points from xy_train
    xy_train_new = xy_train[true_validity_train]
    true_validity_train_new = true_validity_train[true_validity_train]

    # Append the "true invalid" data points to xy_val
    xy_val_new = pd.concat([xy_val, invalid_data])
    true_validity_val_new = pd.concat([true_validity_val, invalid_labels])

    return xy_train_new, xy_val_new, true_validity_train_new, true_validity_val_new
