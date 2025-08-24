import os
import pandas as pd

from addmo.util.definitions import results_dir_model_tuning, results_dir_data_tuning_auto, results_dir_data_tuning_fixed
from addmo.util.load_save_utils import root_dir
from addmo.util.experiment_logger import LocalLogger
from addmo.util.experiment_logger import WandbLogger
from addmo.util.experiment_logger import ExperimentLogger
from addmo.s3_model_tuning.models.keras_models import SciKerasSequential
from addmo.s3_model_tuning.config.model_tuning_config import ModelTuningExperimentConfig
from addmo.s3_model_tuning.config.model_tuning_config import ModelTunerConfig
from addmo.s3_model_tuning.model_tuner import ModelTuner
from addmo.util.load_save import load_data
from addmo.util.load_save import load_config_from_json
from addmo.util.data_handling import split_target_features
from addmo.s5_insights.model_plots.scatter_plot import scatter
from addmo.util.plotting_utils import save_pdf


def exe_model_tuning(user_input='y', config_exp=None, config_tuner=None):
    """
    Executes model tuning process and returns the best model.
    Parameters:
        user_input : str, optional
            If 'y', the contents of the target results directory will be overwritten.
            If 'd', the directory contents will be deleted. Default is 'y'.
        config_exp : DataTuningExperimentConfig
        config_tuner : ModelTunerConfig
    """

    # Configure the logger
    LocalLogger.active = True
    if LocalLogger.active:
        LocalLogger.directory = results_dir_model_tuning( config_exp,user_input)
    WandbLogger.project = "addmo-test_model_tuning"
    WandbLogger.active = False
    if WandbLogger.active:
        WandbLogger.directory = results_dir_model_tuning(config_exp,user_input)

    # Initialize logging
    ExperimentLogger.start_experiment(config=config_exp)

    # Create the model tuner
    model_tuner = ModelTuner(config=config_tuner)

    # Load the system_data
    xy_tuned = load_data(config_exp.abs_path_to_data)

    # Select training and validation period
    xy_tuned_train_val = xy_tuned.loc[config_exp.start_train_val:config_exp.stop_train_val]
    x_train_val, y_train_val = split_target_features(config_exp.name_of_target, xy_tuned_train_val)

    # log start and end of the system_data
    ExperimentLogger.log({"xy_tuned_train_val": pd.concat([xy_tuned_train_val.head(5), xy_tuned_train_val.tail(5)])})

    # Tune the models
    model_dict = model_tuner.tune_all_models(x_train_val, y_train_val)

    # Get the best model
    best_model_name = model_tuner.get_best_model_name(model_dict)
    best_model = model_tuner.get_model(model_dict, best_model_name)
    y_pred = best_model.predict(x_train_val)
    # Log the best model
    if isinstance(best_model, SciKerasSequential):
        art_type = 'keras'
    else:
        art_type = 'joblib'
    name = 'best_model'
    ExperimentLogger.log_artifact(best_model, name, art_type)
    saved_data_name = config_exp.abs_path_to_data.split(".")[0]
    ExperimentLogger.log_artifact(xy_tuned,saved_data_name , "system_data")
    plt = scatter(y_train_val, y_pred, config_exp.name_of_target, best_model.fit_error)
    save_pdf(plt, os.path.join(LocalLogger.directory, 'model_fit_scatter'))
    plt.show()


    print("Finished")

if __name__ == "__main__":
    # Read config from existing json file
    path_to_config_exp = os.path.join(root_dir(), 'addmo', 's3_model_tuning', 'config',
                                  'model_tuner_experiment_config.json')
    path_to_config_tuner= os.path.join(root_dir(), 'addmo', 's3_model_tuning', 'config',
                                  'model_tuner_config.json')
    config_exp = load_config_from_json(path_to_config_exp, ModelTuningExperimentConfig)
    config_tuner = load_config_from_json(path_to_config_tuner, ModelTunerConfig)

    user_input = input(
        "To overwrite the existing content type in 'addmo_examples/results/test_raw_data/test_data_tuning/test_model_tuning' results directory <y>, for deleting the current contents type <d>: ")
    # run
    exe_model_tuning(user_input, config_exp, config_tuner)