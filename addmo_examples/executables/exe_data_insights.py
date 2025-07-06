import os
import json

import pandas as pd

from addmo.s5_insights.model_plots.time_series import plot_timeseries_combined
from addmo.s5_insights.model_plots.parallel_plots import parallel_plots, parallel_plots_interactive
from addmo.util.plotting import save_pdf
from addmo.s5_insights.model_plots.carpet_plots import  plot_carpets, plot_carpets_with_buckets, prediction_func_4_regressor
from addmo.util.definitions import  return_results_dir_model_tuning, return_best_model, load_model_config
from addmo.s3_model_tuning.models.model_factory import ModelFactory
from util.load_save import load_data


def exe_time_series_plot(dir, plot_name, plot_dir, save=True):
    """
    Executes plotting of input data.
    """

    model_config = load_model_config(dir)
    # Load data
    data_path = model_config['abs_path_to_data']
    data = pd.read_csv(data_path, delimiter=",", index_col=0, encoding="latin1", header=0)

    figures = plot_timeseries_combined(model_config, data)


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

def exe_carpet_plots(dir, plot_name, plot_dir, save = True, bounds= None, defaults_dict= None, combinations= None, path_to_regressor=None):
    """
    Executes carpet model_plots of input data features along with predictions using saved model.
    """
    model_config = load_model_config(dir)

    if path_to_regressor is None:
        path_to_regressor = return_best_model(dir)  # return default path where model is saved
    # Load regressor
    regressor = ModelFactory.load_model(path_to_regressor)
    pred_func_1 = prediction_func_4_regressor(regressor)

    plt= plot_carpets(model_config, regressor, pred_func_1, pred_func_2 = None, bounds = bounds, combinations = combinations, defaults_dict = defaults_dict)

    if save:
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, plot_name)
        plt.show()
        save_pdf(plt, plot_path)
    else:
        plt.show()

def exe_scatter_carpet_plots(dir, plot_name, plot_dir, save = True, bounds= None, defaults_dict= None, combinations= None, path_to_regressor= None):
    """
        Executes carpet model_plots of input data features along with predictions using saved model.
    """
    model_config = load_model_config(dir)
    # model_config['abs_path_to_data'] = r"R:\_Dissertationen\mre\Diss\08_Data_Plots_Analysis\0_ADDMo_TrueValidityVSExtrapolationCovargeScores\8_bes_VLCOPcorr_random_NovDez\fullANN\8_bes_VLCOPcorr_random_absurd-sweep-172\regressors\xy_regressor_fit.csv"

    if path_to_regressor is None:
        path_to_regressor = return_best_model(dir)  #return default path where model is saved
    # Load regressor
    regressor = ModelFactory.load_model(path_to_regressor)
    pred_func_1= prediction_func_4_regressor(regressor)

    plt = plot_carpets_with_buckets(model_config=model_config, regressor=regressor, pred_func_1=pred_func_1, pred_func_2=None,
                                    bounds=bounds,  combinations=combinations, defaults_dict=defaults_dict)

    if save:
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, plot_name)
        plt.show()
        save_pdf(plt, plot_path)
    else:
        plt.show()

def exe_parallel_plot(dir, plot_name, plot_dir, save = True, path_to_regressor=None):
    """
    Executes parallel model_plots of input data features along with predictions using saved model.
    """

    # Load config
    model_config = load_model_config(dir)
    # Load regressor
    if path_to_regressor is None:
        path_to_regressor = return_best_model(dir)
    regressor = ModelFactory.load_model(path_to_regressor)
    plt = parallel_plots(model_config, regressor)

    if save:
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, plot_name)
        save_pdf(plt, plot_path)

def exe_interactive_parallel_plot(dir, plot_name, plot_dir, save = True, path_to_regressor=None):
    """
    Executes parallel model_plots of input data features along with predictions using saved model.
    """
    # Load config
    model_config = load_model_config(dir)
    # Load regressor
    if path_to_regressor is None:
        path_to_regressor = return_best_model(dir)
    regressor = ModelFactory.load_model(path_to_regressor)

    plt = parallel_plots_interactive(model_config, regressor)

    if save:
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, plot_name)
        plt.write_html(plot_path)

    return plt


if __name__ == '__main__':

    #
    # # Default saved directory for loading the saved model
    # results_dir = r"D:\04_GitRepos\addmo-extra\aixtra_use_case\results"
    # path_to_regressor = r"R:\_Dissertationen\mre\Diss\08_Data_Plots_Analysis\0_ADDMo_TrueValidityVSExtrapolationCovargeScores\8_bes_VLCOPcorr_random_NovDez\fullANN\8_bes_VLCOPcorr_random_absurd-sweep-172\regressors\regressor.keras"
    # # Read config
    # config_path = os.path.join(_path_to_input_dir, "local_logger", "config.json")
    # with open(config_path, 'r') as f:
    #     model_config = json.load(f)
    #     model_config['abs_path_to_data'] = r"R:\_Dissertationen\mre\Diss\08_Data_Plots_Analysis\0_ADDMo_TrueValidityVSExtrapolationCovargeScores\8_bes_VLCOPcorr_random_NovDez\fullANN\8_bes_VLCOPcorr_random_absurd-sweep-172\regressors\xy_regressor_fit.csv"
    #

    # Define directory where the model config and regressor is saved:
    _path_to_input_dir = return_results_dir_model_tuning('test_raw_data', 'test_data_tuning', 'test_model_tuning_fixed')
    # _path_to_input_dir = r"R:\_Dissertationen\mre\Diss\08_Data_Plots_Analysis\0_ADDMo_TrueValidityVSExtrapolationCovargeScores\8_bes_VLCOPcorr_random_NovDez\fullANN\8_bes_VLCOPcorr_random_absurd-sweep-172"

    # Path for saving the model_plots
    plot_dir = os.path.join(_path_to_input_dir, 'plots')
    # Define regressor path if it is not saved as 'best_model.ext'
    # path_to_regressor = os.path.join(results_dir, 'models.keras')


    # Execute plotting functions
    # exe_time_series_plot(_path_to_input_dir, plot_name = "training_data_time_series", plot_dir= plot_dir,save=False)
    # exe_carpet_plots(_path_to_input_dir, plot_name = "predictions_carpet_test", plot_dir= plot_dir, save=True)
    # exe_parallel_plot(_path_to_input_dir, plot_name =  "parallel_plot",  plot_dir= plot_dir, save=False)
    # exe_interactive_parallel_plot(_path_to_input_dir, plot_name =  "interactive_parallel_plot",  plot_dir= plot_dir, save=False)
    exe_scatter_carpet_plots(_path_to_input_dir, plot_name = "predictions_scatter_carpet_bucket=4", plot_dir= plot_dir, save=False)



# bounds= {'Total active power': [0.0, 31.2], 'Schedule': [0, 1], 'Space Temperature T1': [22.8, 26.0],
#              'Space Temperature T2': [20.6, 23.5], 'Av. Space Temperature': [21.7, 24.75],
#              'Supply Temperature': [14.1, 31.0], 'Empty trial schedule': [0, 0], 'Shut off schedule': [0, 1]}