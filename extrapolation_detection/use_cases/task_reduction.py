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

experiment_name = "Carnot_Test"

from extrapolation_detection.machine_learning_util import data_handling

###################################################################################################
# Load data
xy_tot = data_handling.load_csv("Carnot_mid", path="data")

# Specify data indices used for training, validation and testing
train_val_test = list(range(0, 744))

xy_tot_splitted = data_handling.split_simulation_data(
    xy_tot,
    train_val_test,
    val_fraction=0.1,
    test_fraction=0.1,
    random_state=1,
    shuffle=True,
)

# save to pickle
data_handling.write_pkl(xy_tot_splitted, "data", directory=experiment_name, override=False)

###################################################################################################
# tune model
from core.s3_model_tuning.models.abstract_model import AbstractMLModel
from core.s3_model_tuning.model_tuner import ModelTuner
from core.s3_model_tuning.config.model_tuning_config import ModelTuningSetup

# Path to the config file
path_to_yaml = os.path.join(root_dir(), 'core', 's3_model_tuning', 'config',
                            'model_tuning_config.yaml')
# Create the config object
config = ModelTuningSetup()

# Load the config from the yaml file
# config.load_yaml_to_class(path_to_yaml)

# training data of extrapolation experiment is used for model tuning
x_train_val = xy_tot_splitted["available_data"]["x_train"]
y_train_val = xy_tot_splitted["available_data"]["y_train"]

model_tuner = ModelTuner(config=config)
model_dict = model_tuner.tune_all_models(x_train_val, y_train_val)
best_model = model_tuner.get_best_model(model_dict)
regressor: AbstractMLModel = best_model

# Score regressor per sample
from extrapolation_detection.n_D_extrapolation.score_regressor_per_data_point import \
    score_train_val_test, score_remaining_data, score_meshgrid

train_val_test_errors = score_train_val_test(regressor, xy_tot_splitted["available_data"], metric="mae")
data_handling.write_pkl(train_val_test_errors, "errors_train_test_val", directory=experiment_name,
                        override=False)


# 2_config_score_ann.py
# score() saves remaining_error.pkl (prior data_error)-> contains the mae of the ann for each
# non-available data point
# {
#     "errors": numpy.ndarray
# }

remaining_errors = score_remaining_data(regressor, xy_tot_splitted["non_available_data"], metric="mae")
data_handling.write_pkl(remaining_errors, "errors_remaining", directory=experiment_name,
                        override=False)


# score_2D() saves data_error_2D.pkl -> contains the error_on_mesh of the ann for each point in
# the 2D plot used for the validity domain boundary
# works only for 2 dimensions currently
# {
#     "var1_meshgrid": numpy.ndarray,   # meshgrid of the first variable
#     "var2_meshgrid": numpy.ndarray,   # meshgrid of the second variable
#     ... and so on
#     "error_on_mesh": numpy.ndarray # ann error on the corresponding meshgrid points
# }
from extrapolation_detection.use_cases.score_ann import carnot_model

mesh_points_per_axis = 100
system_simulation = carnot_model
score_meshgrid_dct = score_meshgrid(regressor, xy_tot_splitted, system_simulation, mesh_points_per_axis)

data_handling.write_pkl(score_meshgrid_dct, "errors_meshgrid", experiment_name, override=False)

# 3_config_validity_domain.py
# validity_domain.pkl
# {
#     "error_threshold": float,   # error threshold
#     "ground_truth_train": numpy.ndarray,   # ground truth for training data
#     "ground_truth_val": numpy.ndarray,   # ground truth for validation data
#     "ground_truth_test": numpy.ndarray,   # ground truth for test data
# }

#