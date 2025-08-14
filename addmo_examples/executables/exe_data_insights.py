import os
import pandas as pd, csv
import datetime
from addmo.s5_insights.model_plots.time_series import plot_timeseries_combined
from addmo.s5_insights.model_plots.parallel_plots import parallel_plots, parallel_plots_interactive
from addmo.util.plotting_utils import save_pdf
from addmo.s5_insights.model_plots.carpet_plots import  plot_carpets, plot_carpets_with_buckets, prediction_func_4_regressor
from addmo.util.definitions import  return_results_dir_model_tuning, return_best_model, load_model_config
from addmo.util.load_save import ensure_datetime_index
from addmo.s3_model_tuning.models.model_factory import ModelFactory

def exe_time_series_plot(dir, plot_name, plot_dir, save=True):
    """
    Executes plotting of input data.
    """
    # Load config
    model_config = load_model_config(dir)

    # Load data
    data_path = model_config['abs_path_to_data']
    if data_path.endswith(".xlsx"):
        data = pd.read_excel(data_path, index_col=0,
                             header=0)  # change loading of data as per the input data file ext and delimiter
    elif data_path.endswith(".csv"):
        data = pd.read_csv(data_path, delimiter=csv.Sniffer().sniff(open(data_path).read(1024), delimiters=";,").delimiter, index_col=0, encoding="latin1", header=0)
    else:
        print('No data file found.')

    data = ensure_datetime_index(data,origin=datetime.datetime(2019, 1, 1), fmt="%Y-%m-%d %H:%M:%S")
    # Execute plotting
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
    # Load config
    model_config = load_model_config(dir)

    if path_to_regressor is None:
        path_to_regressor =  return_best_model(dir) # return default path where model is saved
    # Load regressor
    regressor = ModelFactory.load_model(path_to_regressor)
    pred_func_1 = prediction_func_4_regressor(regressor)
    # Load target variable
    target = model_config["name_of_target"]

    # No need to use input data if user provides bounds and default dictionary
    if bounds is not None and defaults_dict is not None:
        variables = regressor.metadata["features_ordered"]
        measurements_data = None

    # Load the input data and fetch variables from it
    else:
        data_path = model_config['abs_path_to_data']
        if data_path.endswith(".xlsx"):
            data = pd.read_excel(data_path, index_col=0, header=0)  # change loading of data as per the input data file ext and delimiter
        elif data_path.endswith(".csv"):
            data = pd.read_csv(data_path,
                               delimiter=csv.Sniffer().sniff(open(data_path).read(1024), delimiters=";,").delimiter,
                               index_col=0, encoding="latin1", header=0)

        else:
            print('No data file found.')

        measurements_data = data.drop(target, axis=1)
        variables = list(measurements_data.columns)

    # Execute plotting
    plt= plot_carpets(variables=variables, measurements_data= measurements_data, regressor_func= pred_func_1, bounds = bounds, combinations = combinations, defaults_dict = defaults_dict)

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

    # If model config is saved in same directory:
    # Load config
    model_config = load_model_config(dir)

    if path_to_regressor is None:
        path_to_regressor = return_best_model(dir)  # return default path where model is saved
    # Load regressor
    regressor = ModelFactory.load_model(path_to_regressor)
    pred_func_1 = prediction_func_4_regressor(regressor)

    # Load target variable
    target = model_config["name_of_target"]

    # Load the input data and fetch variables from it

    data_path = model_config['abs_path_to_data']
    if data_path.endswith(".xlsx"):
        data = pd.read_excel(data_path, index_col=0, header=0)  # change loading of data as per the input data file ext and delimiter
    elif data_path.endswith(".csv"):
        data = pd.read_csv(data_path, delimiter=csv.Sniffer().sniff(open(data_path).read(1024), delimiters=";,").delimiter, index_col=0, encoding="latin1", header=0)

    else:
        print('No data file found.')

    measurements_data = data.drop(target, axis=1)
    variables = list(measurements_data.columns)
    target_values = pd.DataFrame(data[target])

    # Execute plotting
    plt = plot_carpets_with_buckets(variables=variables, measurements_data= measurements_data, target_values= target_values, regressor_func= pred_func_1, bounds = bounds, combinations = combinations, defaults_dict = defaults_dict)

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

    # Load target and data
    target = model_config["name_of_target"]
    data_path = model_config['abs_path_to_data']
    if data_path.endswith(".xlsx"):
        data = pd.read_excel(data_path, index_col=0,
                             header=0)  # change loading of data as per the input data file ext and delimiter
    elif data_path.endswith(".csv"):
        data = pd.read_csv(data_path, delimiter=csv.Sniffer().sniff(open(data_path).read(1024), delimiters=";,").delimiter, index_col=0, encoding="latin1", header=0)

    else:
        print('No data file found.')

    # Execute plotting
    plt = parallel_plots(target, data, regressor)

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

    # Load target and data
    target = model_config["name_of_target"]
    data_path = model_config['abs_path_to_data']
    if data_path.endswith(".xlsx"):
        data = pd.read_excel(data_path, index_col=0, header=0)  # change loading of data as per the input data file ext and delimiter
    elif data_path.endswith(".csv"):
        data = pd.read_csv(data_path, delimiter=csv.Sniffer().sniff(open(data_path).read(1024), delimiters=";,").delimiter, index_col=0, encoding="latin1", header=0)

    else:
        print('No data file found.')
        return None
    # Execute plotting
    plt = parallel_plots_interactive(target, data, regressor)

    if save:
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, plot_name)
        plt.write_html(plot_path)

    return plt


if __name__ == '__main__':


    # Define directory where the model config and regressor is saved:
    _path_to_input_dir = return_results_dir_model_tuning('test_raw_data', 'test_data_tuning', 'test_model_tuning')

    # Path for saving the model_plots
    plot_dir = os.path.join(_path_to_input_dir, 'plots')
    # Define regressor path if it is not saved as 'best_model.ext'
    # path_to_regressor = os.path.join(_path_to_input_dir, 'regressor.keras')


    # Execute plotting functions
    exe_scatter_carpet_plots(_path_to_input_dir, plot_name = "carpet_scatter_plot", plot_dir= plot_dir,save=True, path_to_regressor=None)  #Todo: Note for user: pls check input data path inside the exe func
    exe_time_series_plot(_path_to_input_dir, plot_name = "training_data_time_series", plot_dir= plot_dir,save=False)
    exe_carpet_plots(_path_to_input_dir, plot_name = "predictions_carpet_test", plot_dir= plot_dir, save=True)
    exe_parallel_plot(_path_to_input_dir, plot_name =  "parallel_plot",  plot_dir= plot_dir, save=False)
    exe_interactive_parallel_plot(_path_to_input_dir, plot_name =  "interactive_parallel_plot",  plot_dir= plot_dir, save=False)



