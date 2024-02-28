import os
from core.util.definitions import root_dir

# 1_config_ann_train.py
# saves data.pkl -> all arrays have common index origin from the imported data
# {
#         "available_data": {
#             "x_train": numpy.ndarray,
#             "y_train": numpy.ndarray,
#             "x_val": numpy.ndarray,
#             "y_val": numpy.ndarray,
#             "x_test": numpy.ndarray,
#             "y_test": numpy.ndarray,
#         },
#         "non_available_data": {
#             "x_remaining": numpy.ndarray,
#             "y_remaining": numpy.ndarray,
#         },
#         "header": np.array(df.columns), # probably unnecessary?
#     }
#
# also saves errors.pkl -> follow index of data.pkl and contains the mae of the ann for each data point
# {
#         "train_error": numpy.ndarray,
#         "val_error": numpy.ndarray,
#         "test_error": numpy.ndarray}

# 2_config_score_ann.py
# score() saves remaining_error.pkl (prior data_error)-> contains the mae of the ann for each
# non-available data point
# {
#     "errors": numpy.ndarray
# }

# score_2D() saves data_error_2D.pkl -> contains the error_on_mesh of the ann for each point in
# the 2D plot used for the validity domain boundary
# works only for 2 dimensions currently
# {
#     "var1_meshgrid": numpy.ndarray,   # meshgrid of the first variable
#     "var2_meshgrid": numpy.ndarray,   # meshgrid of the second variable
#     ... and so on
#     "error_on_mesh": numpy.ndarray # ann error on the corresponding meshgrid points
# }


# 3_config_validity_domain.py
# validity_domain.pkl
# {
#     "error_threshold": float,   # error threshold
#     "ground_truth_train": numpy.ndarray,   # ground truth for training data
#     "ground_truth_val": numpy.ndarray,   # ground truth for validation data
#     "ground_truth_test": numpy.ndarray,   # ground truth for test data
# }