import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from addmo.util import plotting as d
from addmo.util.load_save import load_data


def plot_timeseries(config, data_path):
    """
    Function to plot the data from the given model config file path.
    """

    # Load and preprocess data
    data = load_data(data_path)
    # Fetch time column from dataset
    time_column = next((col for col in data.columns if pd.api.types.is_datetime64_any_dtype(data[col])), None)
    # Filter data based on start and end date from model config
    if time_column:
        # Set time column as index
        data.set_index(time_column, inplace=True)

    start_date = config.get('start_train_val', data.index.min())
    end_date = config.get('end_test', data.index.max())
    data = data[(data.index >= start_date) & (data.index <= end_date)]

    time_ser= data[(data.index >= start_date) & (data.index <= end_date)]
    columns_to_plot = data.columns.tolist()

    # Dynamically set figure height
    fig_height = max(5, len(columns_to_plot) * 2.5)
    fig, axes = plt.subplots(len(columns_to_plot), 1, figsize=(d.cm2inch(15.5), d.cm2inch(fig_height)),sharex=True, gridspec_kw={'wspace': 0, 'hspace': 0.08})
    plt.subplots_adjust(left=0.12, right=0.97, bottom=0.05, top=0.97)

    for ax, column in zip(axes, columns_to_plot):
        if column== config['name_of_target']:
            color= d.red
            ax.plot(time_ser.index, time_ser[column], color=color, linewidth=0.75, label='Target')
            ax.legend(loc='upper right', fontsize=7)
            plt.setp(ax.get_yticklabels(), fontsize=7)

        else:
            color= d.black
            ax.plot(time_ser.index, time_ser[column], color=color, linewidth=0.75)
        ax.set_ylabel(column.replace(' ', '\n').replace('__', '\n'), fontsize=7, labelpad=-1.1)
        plt.setp(ax.get_yticklabels(), fontsize=7)
        ax.grid()

    axes[-1].set_xlabel("Time", fontsize=7)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d '))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=0, ha="center",fontsize=7)
    fig.align_labels()

    return fig
