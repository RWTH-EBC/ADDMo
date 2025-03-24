import os
import pandas as pd
import json
from addmo.util.definitions import results_dir_model_tuning
from addmo.s5_insights.plots.time_series import plot_data
from addmo.s5_insights.plots.parallel_plots import parallel_plots
from addmo.s3_model_tuning.config.model_tuning_config import ModelTuningExperimentConfig
from addmo.util.plotting import save_pdf
from addmo.s5_insights.plots.carpet_plots import prediction_func_4_regressor, plot_carpets
from addmo.util.definitions import  return_results_dir_model_tuning, return_best_model
from addmo.s3_model_tuning.models.model_factory import ModelFactory


def exe_time_series_plot(config, plot_name, plot_dir, save=False):
    """
    Executes plotting of data.
    """

    plt = plot_data(config)
    if save:
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, plot_name)
        save_pdf(plt, plot_path)
    else:
        plt.show()

def exe_carpet_plots(model_config, plot_name, plot_dir, save = False):

    # Create and show the plot
    plt = plot_carpets(model_config)

    if save:
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, plot_name)
        save_pdf(plt, plot_path)
    else:
        plt.show()


def exe_parallel_plot(model_config, plot_name, plot_dir, save = False):



    plt = parallel_plots(model_config)
    if save:
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, plot_name)
        save_pdf(plt, plot_path)
    else:
        plt.show()


if __name__ == '__main__':


    dir = return_results_dir_model_tuning()
    # Read config
    config_path = os.path.join(dir, "config.json")
    with open(config_path, 'r') as f:
        model_config = json.load(f)

    plot_dir = os.path.join(results_dir_model_tuning( ModelTuningExperimentConfig()), 'plots')

    # Execute plotting functions
    exe_time_series_plot(model_config,"training_data_time_series",plot_dir,save=False)
    exe_carpet_plots(model_config, "predictions_carpet", plot_dir,save=False)
    exe_parallel_plot(model_config,"parallel_plot", plot_dir, save=False)