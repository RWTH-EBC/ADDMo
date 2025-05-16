import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from addmo.util import plotting as d
from addmo.util.load_save import load_data


def plot_timeseries_combined(config, data_path):
    """
    Returns:
    - Full time range defined in config.
    - First 2-week window (if the data spans more than 21 days).

    Returns one or two matplotlib figures.
    """
    # Load data
    data = load_data(data_path)

    # Fetch time column from dataset
    time_column = next((col for col in data.columns if pd.api.types.is_datetime64_any_dtype(data[col])), None)
    # Filter data based on start and end date from model config
    if time_column:
        # Set time column as index
        data.set_index(time_column, inplace=True)

    # Get date range from config
    start_date = pd.to_datetime(config.get('start_train_val', data.index.min()))
    end_date = pd.to_datetime(config.get('end_test', data.index.max()))
    data = data[(data.index >= start_date) & (data.index <= end_date)]

    # Check data duration
    duration_days = (data.index.max() - data.index.min()).days
    columns_to_plot = data.columns.tolist()

    fig_height = max(5, len(columns_to_plot) * 2.5)
    fig_full, axes_full = plt.subplots(len(columns_to_plot), 1,
                                       figsize=(d.cm2inch(15.5), d.cm2inch(fig_height)),
                                       sharex=True, gridspec_kw={'wspace': 0, 'hspace': 0.08})
    plt.subplots_adjust(left=0.12, right=0.97, bottom=0.05, top=0.97)

    for ax, column in zip(axes_full, columns_to_plot):
        color = d.red if column == config['name_of_target'] else d.black
        ax.plot(data.index, data[column], color=color, linewidth=0.75,
                label='Target' if column == config['name_of_target'] else None)
        if column == config['name_of_target']:
            ax.legend(loc='upper right', fontsize=7)

        ax.set_ylabel(column.replace(' ', '\n').replace('__', '\n'), fontsize=7, labelpad=-1.1)
        ax.grid()
        plt.setp(ax.get_yticklabels(), fontsize=7)

    axes_full[-1].set_xlabel("Time", fontsize=7)
    axes_full[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.setp(axes_full[-1].xaxis.get_majorticklabels(), rotation=0, ha="center", fontsize=7)
    fig_full.align_labels()

    figures = [fig_full]

    if duration_days > 10:
        window_start = data.index.min()
        window_end = window_start + pd.Timedelta(days=5)
        data_2weeks = data[(data.index >= window_start) & (data.index < window_end)]

        fig_height_2w = max(5, len(columns_to_plot) * 2.5)
        fig_2weeks, axes_2weeks = plt.subplots(len(columns_to_plot), 1,
                                               figsize=(d.cm2inch(15.5), d.cm2inch(fig_height_2w)),
                                               sharex=True, gridspec_kw={'wspace': 0, 'hspace': 0.08})
        plt.subplots_adjust(left=0.12, right=0.97, bottom=0.05, top=0.97)

        for ax, column in zip(axes_2weeks, columns_to_plot):
            color = d.red if column == config['name_of_target'] else d.black
            ax.plot(data_2weeks.index, data_2weeks[column], color=color, linewidth=0.75,
                    label='Target' if column == config['name_of_target'] else None)
            if column == config['name_of_target']:
                ax.legend(loc='upper right', fontsize=7)

            ax.set_ylabel(column.replace(' ', '\n').replace('__', '\n'), fontsize=7, labelpad=-1.1)
            ax.grid()
            plt.setp(ax.get_yticklabels(), fontsize=7)

        axes_2weeks[-1].set_xlabel("Time", fontsize=7)
        axes_2weeks[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.setp(axes_2weeks[-1].xaxis.get_majorticklabels(), rotation=0, ha="center", fontsize=7)
        fig_2weeks.align_labels()

        figures.append(fig_2weeks)

    return figures if len(figures) > 1 else figures[0]

