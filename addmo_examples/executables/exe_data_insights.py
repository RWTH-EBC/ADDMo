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

def exe_carpet_plots(model_config, plot_name, plot_dir, save = False):
    """
    Executes carpet model_plots of input data features along with predictions using saved model.
    """
    # Create and show the plot
    plt = plot_carpets(model_config)

    if save:
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, plot_name)
        plt.show()
        save_pdf(plt, plot_path)
    else:
        plt.show()


def exe_parallel_plot(model_config, plot_name, plot_dir, save = False):
    """
    Executes parallel model_plots of input data features along with predictions using saved model.
    """


    plt = parallel_plots(model_config)
    plt.show()
    if save:
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, plot_name)
        save_pdf(plt, plot_path)


if __name__ == '__main__':

    # Default saved directory for loading the saved model
    dir = return_results_dir_model_tuning('test_raw_data', 'test_data_tuning', 'test_model_tuning')

    # Read config
    config_path = os.path.join(dir, "config.json")
    with open(config_path, 'r') as f:
        model_config = json.load(f)

    # Path for saving the model_plots
    plot_dir = os.path.join(dir, 'plots')

    # Execute plotting functions
    exe_time_series_plot(model_config,"training_data_time_series",plot_dir,save=False)
    # exe_carpet_plots(model_config, "predictions_carpet", plot_dir,save=True)
    # exe_parallel_plot(model_config,"parallel_plot", plot_dir, save=True)