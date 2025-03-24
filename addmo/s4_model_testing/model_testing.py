#TODO; Notes: load previously saved model, ask if model tuning is needed, if yes then do automatic data tuning based on config and then make predictions and score
import json
import os
import pandas as pd
from addmo.util.definitions import results_dir_data_tuning, results_dir
from addmo.util.load_save_utils import root_dir
from addmo.s1_data_tuning_auto.data_tuner_auto import DataTunerAuto
from addmo.s1_data_tuning_auto.config.data_tuning_auto_config import DataTuningAutoSetup
from addmo.util.experiment_logger import ExperimentLogger
from addmo.util.experiment_logger import LocalLogger
from addmo.util.experiment_logger import WandbLogger
from addmo.util.definitions import  return_results_dir_model_tuning, return_best_model
from addmo.s3_model_tuning.models.model_factory import ModelFactory
from sklearn.metrics import root_mean_squared_error
from addmo.s2_data_tuning.config.data_tuning_config import DataTuningFixedConfig
from addmo.s2_data_tuning.data_tuner_fixed import DataTunerByConfig
from addmo.util.load_save import load_data
from addmo.util.data_handling import split_target_features

def model_test(model_config, input_data_path, input_data_exp_name, model_tuning=False):

    # Load regressor
    path_to_regressor = return_best_model(return_results_dir_model_tuning())
    regressor = ModelFactory.load_model(path_to_regressor)

    # Load data tuning config used for the model
    name_of_raw_data = model_config['name_of_raw_data']
    name_of_tuning = "data_tuning_experiment_auto"    #model_config['name_of_data_tuning_experiment']
    # Up Here we assume that the tuning config is in the same dir as model tuning results
    # However, auto data tuning is saved in a different directory by default
    # Todo: add a comment in data auto tuning to change the dir to 'test_data_tuning' so that everything is saved in the same dir and fetched directly from config

    path_data_tuning_config = os.path.join(root_dir(), results_dir(), name_of_raw_data, name_of_tuning, "config.json")
    with open(path_data_tuning_config, "r") as file:
        saved_config_data = json.load(file)


    if model_tuning:
        # Load new dataset here
        saved_config_data["abs_path_to_data"] = input_data_path
        saved_config_data["name_of_raw_data"] = input_data_exp_name
        saved_config_data["name_of_data_tuning_experiment"] = "data_tuning_experiment_auto"

        # Convert the dictionary back to the DataTuningAutoSetup object
        new_config = DataTuningAutoSetup(**saved_config_data)

        LocalLogger.directory = results_dir_data_tuning(new_config)
        LocalLogger.active = True
        WandbLogger.project = "addmo-test_data_auto_tuning"
        WandbLogger.directory = results_dir_data_tuning(new_config)
        WandbLogger.active = False

        ExperimentLogger.start_experiment(config=new_config)

        # Create a new tuner instance with the same tuning parameters
        new_tuner = DataTunerAuto(config=new_config)

        # Apply the same tuning process to the new data
        tuned_x_new = new_tuner.tune_auto()
        y_new = new_tuner.y

        tuned_xy_new = pd.concat([y_new, tuned_x_new], axis=1, join="inner").bfill()

        # Log the tuned system data
        ExperimentLogger.log_artifact(tuned_xy_new, name="tuned_xy_testing", art_type="system_data")
        y_pred = pd.Series(regressor.predict(tuned_x_new), index=tuned_x_new.index)
        fit_error= root_mean_squared_error(y_new, y_pred)


    else:
        xy = load_data(input_data_path)
        x = xy.drop(saved_config_data['name_of_target'], axis=1)
        y = xy[saved_config_data['name_of_target']]
        y_pred = pd.Series(regressor.predict(x), index=x.index)
        fit_error= root_mean_squared_error(y, y_pred)



    print("error is : ", fit_error)

if __name__ == "__main__":
    dir = return_results_dir_model_tuning()
    # Read config
    config_path = os.path.join(dir, "config.json")
    with open(config_path, 'r') as f:
        model_config = json.load(f)
    input_data_path = os.path.join(root_dir(),'addmo_examples','raw_input_data','InputData.xlsx')
    input_data_exp_name = 'model_testing'
    model_test(model_config, input_data_path, input_data_exp_name, True)


# If using fixed config then use this:
#TODO: make common variables in fixed and auto tuning identical and then ask user to specify tuning type and do this dynamically
   # Split the system_data

        # Using fixed config
        # tuner = DataTunerByConfig(config=new_config)
        # xy_raw = load_data(new_config.abs_path_to_data)
        # x, y = split_target_features(new_config.target, xy_raw)
        #
        # # Tune the system_data
        # tuned_x = tuner.tune_fixed(xy_raw)
        # ExperimentLogger.log({"x_tuned": tuned_x.iloc[[0, 1, 2, -3, -2, -1]]})
        #
        # # Merge target and features
        # xy_tuned = tuned_x.join(y)
        # ExperimentLogger.log({"xy_tuned": xy_tuned.iloc[[0, 1, 2, -3, -2, -1]]})
        #
        # # Drop NaNs
        # xy_tuned = xy_tuned.dropna()
        # ExperimentLogger.log({"xy_tuned": xy_tuned.iloc[[0, 1, 2, -3, -2, -1]]})
        #
        # # Log the tuned system_data
        # ExperimentLogger.log_artifact(xy_tuned, name='xy_tuned_fixed', art_type='system_data')

        # Now make predictions on new data using saved regressor
        # x_tuned = tuned_x_new.drop(new_config.name_of_target, axis=1)

