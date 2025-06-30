import os
import json
from addmo.s5_insights.model_plots.time_series import plot_timeseries_combined
from addmo.s5_insights.model_plots.parallel_plots import parallel_plots, parallel_plots_interactive
from addmo.util.plotting import save_pdf
from addmo.s5_insights.model_plots.carpet_plots import  plot_carpets, plot_carpets_with_buckets
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

def exe_scatter_carpet_plots(dir, plot_name, plot_dir, save = False, bounds= None, defaults_dict= None, combinations= None):
    """
        Executes carpet model_plots of input data features along with predictions using saved model.
    """
    # Create and show the plot
    config_path = os.path.join(dir, "config.json")
    with open(config_path, 'r') as f:
        model_config = json.load(f)

    path_to_regressor = os.path.join(dir, "best_model.joblib")

    plt = plot_carpets_with_buckets(model_config,bounds=bounds, defaults_dict=defaults_dict, combinations=combinations,path_to_regressor=path_to_regressor)
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

    plt = parallel_plots(model_config, path_to_regressor= path_to_regressor)
    # plt = parallel_plots_interactive(model_config, path_to_regressor=path_to_regressor)
    import plotly.io as pio
    pio.renderers.default = "browser"
    plt.show()
    if save:
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, plot_name)
        save_pdf(plt, plot_path)

def exe_interactive_parallel_plot(model_config, plot_name, plot_dir, save = False, path_to_regressor=None):
    """
    Executes parallel model_plots of input data features along with predictions using saved model.
    """

    plt = parallel_plots_interactive(model_config, path_to_regressor=path_to_regressor)

    if save:
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, plot_name)
        plt.write_html(plot_path)

    return plt
if __name__ == '__main__':

    _path_to_input_dir = r"R:\_Dissertationen\mre\Diss\08_Data_Plots_Analysis\0_ADDMo_TrueValidityVSExtrapolationCovargeScores\8_bes_VLCOPcorr_random_NovDez\fullANN\8_bes_VLCOPcorr_random_absurd-sweep-172"

    # Default saved directory for loading the saved model
    results_dir = r"D:\04_GitRepos\addmo-extra\aixtra_use_case\results"
    path_to_regressor = r"R:\_Dissertationen\mre\Diss\08_Data_Plots_Analysis\0_ADDMo_TrueValidityVSExtrapolationCovargeScores\8_bes_VLCOPcorr_random_NovDez\fullANN\8_bes_VLCOPcorr_random_absurd-sweep-172\regressors\regressor.keras"
    # Read config
    config_path = os.path.join(_path_to_input_dir, "local_logger", "config.json")
    with open(config_path, 'r') as f:
        model_config = json.load(f)
        model_config['abs_path_to_data'] = r"R:\_Dissertationen\mre\Diss\08_Data_Plots_Analysis\0_ADDMo_TrueValidityVSExtrapolationCovargeScores\8_bes_VLCOPcorr_random_NovDez\fullANN\8_bes_VLCOPcorr_random_absurd-sweep-172\regressors\xy_regressor_fit.csv"

    # Path for saving the model_plots
    plot_dir = os.path.join(results_dir, 'plots')

    # Execute plotting functions
    # exe_time_series_plot(model_config,"training_data_time_series",plot_dir,save=False)
    # exe_carpet_plots(results_dir, "predictions_carpet", plot_dir,save=False)
    # exe_parallel_plot(model_config,"parallel_plot", plot_dir, save=False)
    # exe_interactive_parallel_plot(model_config,"interactive_parallel_plot", plot_dir, save=False)
    # exe_scatter_carpet_plots(_path_to_input_dir, "predictions_scatter_carpet_bucket=4", plot_dir, save=True) #TODO: whats that string thing?

    plt = plot_carpets_with_buckets(model_config,bounds=None, defaults_dict=None, combinations=None,path_to_regressor=path_to_regressor, num_buckets=4)

    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, "test")
    plt.show()
    save_pdf(plt, plot_path)
    print(plot_path)

