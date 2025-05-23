import os
import json
from addmo.util.definitions import results_dir_model_tuning
from addmo.s5_insights.model_plots.time_series import plot_timeseries_combined
from addmo.s5_insights.model_plots.parallel_plots import parallel_plots
from addmo.s3_model_tuning.config.model_tuning_config import ModelTuningExperimentConfig
from addmo.util.plotting import save_pdf
from addmo.s5_insights.model_plots.carpet_plots import  plot_carpets
from addmo.util.definitions import  return_results_dir_model_tuning


def exe_time_series_plot(model_config, plot_name, plot_dir, save=False):
    """
    Executes plotting of input data.
    """

    figures = plot_timeseries_combined(model_config, data_path=model_config['abs_path_to_data'])

    if not isinstance(figures, list):
        figures = [figures]
    if save:
        os.makedirs(plot_dir, exist_ok=True)
        for idx, fig in enumerate(figures):
            suffix = "_2weeks" if idx == 1 else ""
            plot_path = os.path.join(plot_dir, f"{plot_name}{suffix}")
            save_pdf(fig, plot_path)
    else:
        for fig in figures:
            fig.show()

def exe_carpet_plots(dir, plot_name, plot_dir, save = False, bounds= None, defaults_dict= None, combinations= None):
    """
    Executes carpet model_plots of input data features along with predictions using saved model.
    """
    # Create and show the plot
    config_path = os.path.join(dir, "config.json")
    with open(config_path, 'r') as f:
        model_config = json.load(f)

    path_to_regressor = os.path.join(dir, "best_model.joblib")

    plt = plot_carpets(model_config,  bounds=bounds , defaults_dict = defaults_dict, combinations= combinations, path_to_regressor= path_to_regressor)

    if save:
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, plot_name)
        plt.show()
        save_pdf(plt, plot_path)
    else:
        plt.show()


def exe_parallel_plot(model_config, plot_name, plot_dir, save = False, path_to_regressor=None):
    """
    Executes parallel model_plots of input data features along with predictions using saved model.
    """


    plt = parallel_plots(model_config, path_to_regressor=path_to_regressor)
    plt.show()
    if save:
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, plot_name)
        save_pdf(plt, plot_path)


if __name__ == '__main__':

    # Default saved directory for loading the saved model
    # dir = return_results_dir_model_tuning('test_raw_data', 'test_data_tuning', 'test_model_tuning')
    dir = r'C:\Users\mre-rpa\Desktop\PycharmProjects\addmo-automated-ml-regression\addmo_examples\results\plots'
    path_to_regressor = r"C:\Users\mre-rpa\Desktop\PycharmProjects\addmo-automated-ml-regression\addmo_examples\results\plots\best_model.joblib"
    # Read config
    config_path = os.path.join(dir, "config.json")
    with open(config_path, 'r') as f:
        model_config = json.load(f)

    # Path for saving the model_plots
    plot_dir = os.path.join(dir, 'plots')
    # bounds = {"Total active power": [0, 45], "Schedule": [0, 1], "FreshAir Temperature": [10.6, 28.1],
    #           "Space Temperature T1": [22.8, 26],
    #           "Space Temperature T2": [20.6, 23.5], "Av. Space Temperature": [21.7, 24.75],
    #           "Supply Temperature": [14.1, 31], "Empty trial schedule": [0, 0], "Shut off schedule": [0, 1]}
    # defaults_dict = {"Total active power": 15, "Schedule": 1, "FreshAir Temperature": 15, "Space Temperature T1": 24,
    #                  "Space Temperature T2": 21, "Av. Space Temperature": 22, "Supply Temperature": 18,
    #                  "Empty trial schedule": 0, "Shut off schedule": 1}

    # Execute plotting functions
    # exe_time_series_plot(model_config,"training_data_time_series",plot_dir,save=False)
    exe_carpet_plots(dir, "predictions_carpet", plot_dir,save=True)
    # exe_parallel_plot(model_config,"parallel_plot", plot_dir, save=True)