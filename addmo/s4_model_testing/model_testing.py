import json
import os
import pandas as pd
from addmo.util.definitions import results_dir_data_tuning, results_dir
from addmo.util.load_save_utils import root_dir
from addmo.s1_data_tuning_auto.data_tuner_auto import DataTunerAuto
from addmo.s1_data_tuning_auto.config.data_tuning_auto_config import DataTuningAutoSetup
from addmo.util.experiment_logger import ExperimentLogger, LocalLogger, WandbLogger
from addmo.util.definitions import  results_dir_data_tuning_fixed, return_best_model,results_dir_data_tuning_auto
from addmo.s3_model_tuning.models.model_factory import ModelFactory
from sklearn.metrics import root_mean_squared_error
from addmo.s2_data_tuning.config.data_tuning_config import DataTuningFixedConfig
from addmo.s2_data_tuning.data_tuner_fixed import DataTunerByConfig
from addmo.util.load_save import load_data
from addmo.util.data_handling import split_target_features
from addmo.s5_insights.model_plots.scatter_plot import scatter
from addmo.util.plotting_utils import save_pdf
from addmo.util.load_save_utils import create_or_clean_directory

def model_test(dir, model_config, input_data_path, input_data_exp_name, data_tuning_type):

    # Load regressor
    path_to_regressor = return_best_model(dir)
    regressor = ModelFactory.load_model(path_to_regressor)


    if data_tuning_type.lower()== 'auto':
        # Load data tuning config used for the model
        name_of_raw_data = model_config['name_of_raw_data']
        path_data_tuning_config = os.path.join(results_dir_data_tuning_auto(name_of_raw_data), "config.json")
        with open(path_data_tuning_config, "r") as file:
            saved_config_data = json.load(file)


        tuned_x_new,y_new, new_config = data_tuning_recreate_auto(saved_config_data, input_data_path, input_data_exp_name)
        y_pred = pd.Series(regressor.predict(tuned_x_new), index=tuned_x_new.index)
        fit_error= root_mean_squared_error(y_new, y_pred)
        plt = scatter(y_new, y_pred, model_config['name_of_target'],fit_error)
        saving_dir = results_dir_data_tuning(new_config)
        save_pdf(plt, os.path.join(saving_dir, 'model_fit_scatter'))
        plt.show()

    elif data_tuning_type.lower()== 'fixed':
        # Load data tuning config used for the model
        name_of_raw_data = model_config['name_of_raw_data']
        path_data_tuning_config = os.path.join(results_dir_data_tuning_fixed(name_of_raw_data),"config.json")
        with open(path_data_tuning_config, "r") as file:
            saved_config_data = json.load(file)


        tuned_x_new, y, new_config = data_tuning_recreate_fixed(saved_config_data, input_data_path, input_data_exp_name)
        y_pred = pd.Series(regressor.predict(tuned_x_new), index=tuned_x_new.index)
        fit_error= root_mean_squared_error(y, y_pred)
        plt = scatter(y, y_pred, model_config['name_of_target'],fit_error)
        saving_dir = results_dir_data_tuning(new_config)
        save_pdf(plt, os.path.join(saving_dir,'model_fit_scatter'))
        plt.show()

    else:
        # No separate config file needed for data in case of no tuning
        xy = load_data(input_data_path)
        x = xy.drop(model_config['name_of_target'], axis=1)
        y = xy[model_config['name_of_target']]
        y_pred = pd.Series(regressor.predict(x), index=x.index)
        fit_error= root_mean_squared_error(y, y_pred)
        plt=scatter(y, y_pred, model_config['name_of_target'],fit_error)
        saving_dir = os.path.join(root_dir(), results_dir(), input_data_exp_name,"data_tuning_experiment_raw")
        create_or_clean_directory(saving_dir)
        save_pdf(plt, os.path.join(saving_dir,'model_fit_scatter'))
        plt.show()


    return  fit_error, saving_dir

def data_tuning_recreate_auto(saved_config_data, input_data_path, input_data_exp_name):
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
    return tuned_x_new, y_new, new_config


def data_tuning_recreate_fixed(saved_config_data, input_data_path, input_data_exp_name):
    # Load new dataset here
    saved_config_data["abs_path_to_data"] = input_data_path
    saved_config_data["name_of_raw_data"] = input_data_exp_name
    saved_config_data["name_of_data_tuning_experiment"] = "data_tuning_experiment_fixed"

    # Convert the dictionary back to the DataTuningAutoSetup object
    new_config = DataTuningFixedConfig(**saved_config_data)

    LocalLogger.directory = results_dir_data_tuning(new_config)
    LocalLogger.active = True
    WandbLogger.project = "addmo-test_data_auto_tuning"
    WandbLogger.directory = results_dir_data_tuning(new_config)
    WandbLogger.active = False

    ExperimentLogger.start_experiment(config=new_config)

    #Using fixed config
    tuner = DataTunerByConfig(config=new_config)
    xy_raw = load_data(new_config.abs_path_to_data)
    x, y = split_target_features(new_config.name_of_target, xy_raw)

    # Tune the system_data
    tuned_x = tuner.tune_fixed(xy_raw)
    ExperimentLogger.log({"x_tuned": tuned_x.iloc[[0, 1, 2, -3, -2, -1]]})

    # Merge target and features
    xy_tuned = tuned_x.join(y)
    ExperimentLogger.log({"xy_tuned": xy_tuned.iloc[[0, 1, 2, -3, -2, -1]]})

    # Drop NaNs
    xy_tuned = xy_tuned.dropna()
    ExperimentLogger.log({"xy_tuned": xy_tuned.iloc[[0, 1, 2, -3, -2, -1]]})

    # Log the tuned system_data
    ExperimentLogger.log_artifact(xy_tuned, name='xy_tuned_fixed', art_type='system_data')

    tuned_y_new = xy_tuned[new_config.name_of_target]
    tuned_x_new = xy_tuned.drop(new_config.name_of_target, axis=1)

    return tuned_x_new, tuned_y_new, new_config


