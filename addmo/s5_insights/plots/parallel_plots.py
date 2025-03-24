import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from addmo.util import plotting as d
from addmo.util.definitions import  return_results_dir_model_tuning, return_best_model
from addmo.s3_model_tuning.models.model_factory import ModelFactory


def parallel_plots(model_config):

    # Load target and data
    target = model_config["name_of_target"]
    data_path = model_config['abs_path_to_data']
    data = pd.read_excel(data_path)

    # Load regressor
    path_to_regressor = return_best_model(return_results_dir_model_tuning())
    regressor = ModelFactory.load_model(path_to_regressor)

    # Pre-process data
    time_column = next((col for col in data.columns if pd.api.types.is_datetime64_any_dtype(data[col])), None)
    if time_column:
        data.set_index(time_column, inplace=True)

    xy_grid = data.drop(target, axis=1)
    y_pred = pd.Series(regressor.predict(xy_grid), index=xy_grid.index)
    xy_grid[target] = data[target]
    xy_grid['y_pred'] = y_pred


    # columns to plot:
    cols = []
    for var in xy_grid.columns:
        min_val, max_val = xy_grid[var].min(), xy_grid[var].max()
        if min_val != max_val:  # Only keep variables with a valid range
            cols.append(var)

    xy_grid = xy_grid[cols]
    ys_grid = xy_grid.to_numpy()[:, :]
    ymins_grid = ys_grid.min(axis=0)
    ymax_grid = ys_grid.max(axis=0)
    dys_grid = ymax_grid - ymins_grid
    ymins_grid -= dys_grid * 0.05 # Add padding
    ymax_grid += dys_grid * 0.05

    zs_grid = np.zeros_like(ys_grid)
    zs_grid[:, 0] = ys_grid[:, 0]
    zs_grid[:, 1:] = (ys_grid[:, 1:] - ymins_grid[1:]) / dys_grid[1:] * dys_grid[0] + ymins_grid[0]

    dys = ymax_grid - ymins_grid
    zs = np.zeros_like(ys_grid)
    zs[:, 0] = ys_grid[:, 0]
    zs[:, 1:] = (ys_grid[:, 1:] - ymins_grid[1:]) / dys[1:] * dys[0] + ymins_grid[0]

    num_vars= len(xy_grid.columns) + 1
    figure_width =  max(5, num_vars * 2.5)
    fig_size = (d.cm2inch(figure_width), d.cm2inch(8))  # Adjusted figure size
    fig, host = plt.subplots(figsize=fig_size)
    plt.subplots_adjust(left=0.05, right=0.92, bottom=0.08, top=0.8)


    axes = [host] + [host.twinx() for i in range(ys_grid.shape[1] - 1)]
    for i, ax in enumerate(axes):

        ax.set_ylim(ymins_grid[i], ymax_grid[i])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        if ax != host:
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_ticks_position('right')
            ax.spines['right'].set_position(("axes", i / (ys_grid.shape[1] - 1)))
    host.set_xlim(0, ys_grid.shape[1] - 1)
    host.set_xticks(range(ys_grid.shape[1]))
    host.set_xticklabels([col.replace(' ', '\n') for col in xy_grid.columns])
    host.tick_params(axis='x', which='major', pad=7, labelsize=9)
    host.spines['right'].set_visible(False)
    host.xaxis.tick_top()

    for j in range(zs_grid.shape[0]):
        host.plot(np.arange(ys_grid.shape[1]), zs_grid[j, :], color=d.red, linewidth=0.5, alpha=0.7)

    return plt




