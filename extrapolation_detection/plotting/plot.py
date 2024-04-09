import os.path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, colors, axes
from matplotlib.cm import ScalarMappable
from matplotlib.tri import Triangulation

from extrapolation_detection.util import loading_saving
from extrapolation_detection.util import data_handling as dh


class PlotData2D:
    def __init__(self):
        self.plot_title: str = None

        self.x_training: pd.DataFrame = None
        self.x_val: pd.DataFrame = None
        self.x_test: pd.DataFrame = None
        self.x_remaining: pd.DataFrame = None
        self.x_grid: pd.DataFrame = None

        self.errors_train: pd.DataFrame = None
        self.errors_val: pd.DataFrame = None
        self.errors_test: pd.DataFrame = None
        self.errors_remaining: pd.DataFrame = None
        self.errors_grid: pd.DataFrame = None

        self.true_validity_train: pd.Series = None
        self.true_validity_val: pd.Series = None
        self.true_validity_test: pd.Series = None
        self.true_validity_remaining: pd.Series = None
        self.true_validity_grid: pd.Series = None
        self.true_validity_threshold: float = None

        self.n_score_train: pd.Series = None
        self.n_score_val: pd.Series = None
        self.n_score_test: pd.Series = None
        self.n_score_remaining: pd.Series = None
        self.n_score_grid: pd.Series = None
        self.n_score_threshold: float = None

        # constructed from the data depending on the use case
        self.x_train_ml_model = None
        self.errors_train_ml_model = None
        self.x_test_ml_model = None
        self.errors_test_ml_model = None

    def load_plot_data(self, experiment_folder: str, detector: str):
        # read experiment data
        self.x_train = loading_saving.read_csv("x_train", directory=experiment_folder)
        self.y_train = loading_saving.read_csv("y_train", directory=experiment_folder)
        self.x_val = loading_saving.read_csv("x_val", directory=experiment_folder)
        self.y_val = loading_saving.read_csv("y_val", directory=experiment_folder)
        self.x_test = loading_saving.read_csv("x_test", directory=experiment_folder)
        self.y_test = loading_saving.read_csv("y_test", directory=experiment_folder)
        self.x_remaining = loading_saving.read_csv(
            "x_remaining", directory=experiment_folder
        )
        self.y_remaining = loading_saving.read_csv(
            "y_remaining", directory=experiment_folder
        )
        self.x_grid = loading_saving.read_csv("x_grid", directory=experiment_folder)
        self.y_grid = loading_saving.read_csv("y_grid", directory=experiment_folder)

        self.xy_training = loading_saving.read_csv(
            "xy_train", directory=experiment_folder
        )
        self.xy_validation = loading_saving.read_csv(
            "xy_val", directory=experiment_folder
        )
        self.xy_test = loading_saving.read_csv("xy_test", directory=experiment_folder)
        self.xy_remaining = loading_saving.read_csv(
            "xy_remaining", directory=experiment_folder
        )
        self.xy_grid = loading_saving.read_csv("xy_grid", directory=experiment_folder)

        # read errors
        self.errors_train = loading_saving.read_csv(
            "errors_train", directory=experiment_folder
        )
        self.errors_val = loading_saving.read_csv(
            "errors_val", directory=experiment_folder
        )
        self.errors_test = loading_saving.read_csv(
            "errors_test", directory=experiment_folder
        )
        self.errors_remaining = loading_saving.read_csv(
            "errors_remaining", directory=experiment_folder
        )
        self.errors_grid = loading_saving.read_csv(
            "errors_grid", directory=experiment_folder
        )

        # read true_validity
        self.true_validity_train = loading_saving.read_csv(
            "true_validity_train", experiment_folder
        ).squeeze()
        self.true_validity_val = loading_saving.read_csv(
            "true_validity_val", experiment_folder
        ).squeeze()
        self.true_validity_test = loading_saving.read_csv(
            "true_validity_test", experiment_folder
        ).squeeze()
        self.true_validity_remaining = loading_saving.read_csv(
            "true_validity_remaining", experiment_folder
        ).squeeze()
        self.true_validity_grid = loading_saving.read_csv(
            "true_validity_grid", experiment_folder
        ).squeeze()
        self.true_validity_threshold = loading_saving.read_csv(
            "true_validity_threshold", experiment_folder
        ).iloc[0, 0]

        # read detector data
        self.n_score_train = loading_saving.read_csv(
            f"n_score_train_{detector}", directory=experiment_folder
        ).squeeze()
        self.n_score_val = loading_saving.read_csv(
            f"n_score_val_{detector}", directory=experiment_folder
        ).squeeze()
        self.n_score_test = loading_saving.read_csv(
            f"n_score_test_{detector}", directory=experiment_folder
        ).squeeze()
        self.n_score_remaining = loading_saving.read_csv(
            f"n_score_remaining_{detector}", directory=experiment_folder
        ).squeeze()
        self.n_score_grid = loading_saving.read_csv(
            f"n_score_grid_{detector}", directory=experiment_folder
        ).squeeze()
        self.n_score_threshold = loading_saving.read_csv(
            f"{detector}_threshold",
            directory=os.path.join(experiment_folder, "detectors"),
        ).iloc[0, 0]

        # read detector data
        self.x_train_detector = loading_saving.read_csv(
            f"{detector}_x_fit",
            directory=os.path.join(experiment_folder, "detectors"),
        )
        self.x_val_detector = loading_saving.read_csv(
            f"{detector}_x_val",
            directory=os.path.join(experiment_folder, "detectors"),
        )

    def infer_ml_model_data_splits(self):
        """Use if the model is tuned on train and validiation data, and after tuning trained on
        the same train+val split."""
        self.x_train_ml_model = pd.concat([self.x_train, self.x_val])
        self.errors_train_ml_model = pd.concat([self.errors_train, self.errors_val])
        self.x_test_ml_model = self.x_test
        self.errors_test_ml_model = self.errors_test

    def infer_detector_data_splits(self):
        index_fit = self.x_train_detector.index
        index_val = self.x_val_detector.index

        x_train_ml_model = self.x_train_ml_model.loc[index_fit]
        errors_train_ml_model = self.errors_train_ml_model.loc[index_fit]
        x_test_ml_model = pd.concat([self.x_train, self.x_val, self.x_test]).loc[
            index_val
        ]
        errors_test_ml_model = pd.concat(
            [self.errors_train, self.errors_val, self.errors_test]
        ).loc[index_val]
        # self.x_test_ml_model = self.x_train_ml_model
        # self.errors_test_ml_model = self.errors_train_ml_model
        #
        # self.x_test_ml_model = pd.DataFrame(columns=self.x_train_ml_model.columns)
        # self.errors_test_ml_model = pd.DataFrame(columns=self.errors_train_ml_model.columns)

        # for better visualization switch train and test
        self.x_train_ml_model = x_test_ml_model
        self.errors_train_ml_model = errors_test_ml_model
        self.x_test_ml_model = x_train_ml_model
        self.errors_test_ml_model = errors_train_ml_model


def show_plot(plt):
    # if plt is matplotlib plot
    if hasattr(plt, "show"):
        plt.show()
    # if plt is a plotly plot
    if hasattr(plt, "show"):
        plt.show()


def save_plot(plt, file_name: str, experiment_folder: str):
    folder = os.path.join(experiment_folder, "plots")
    if not os.path.exists(folder):
        os.mkdir(folder)

    # if plt is matplotlib plot
    if hasattr(plt, "savefig"):
        plt.savefig(os.path.join(folder, file_name), bbox_inches="tight")
        plt.close()

    # if plt is a plotly plot, save it
    if hasattr(plt, "write_html"):
        plt.write_html(os.path.join(folder, file_name))


def _plot_subplot(plt_data: PlotData2D, axs: axes.Axes):
    """Plots 2D UseCases"""

    # Define colors
    green = (87 / 255, 171 / 255, 39 / 255)
    darkgrey = (78 / 255, 79 / 255, 80 / 255)
    red = (221 / 255, 64 / 255, 45 / 255)
    black = (0, 0, 0)
    blue = (0 / 255, 84 / 255, 159 / 255)
    petrol = (0 / 255, 97 / 255, 101 / 255)
    violett = (97 / 255, 33 / 255, 88 / 255)

    color_dict = {
        "green": green,
        "darkgrey": darkgrey,
        "red": red,
        "black": black,
        "blue": blue,
        "petrol": petrol,
        "violett": violett,
    }

    # Color Norm
    divnorm = colors.TwoSlopeNorm(
        vmin=min(
            np.amin(plt_data.errors_train["error"]),
            np.amin(plt_data.errors_val["error"]),
            np.amin(plt_data.errors_test["error"]),
            np.amin(plt_data.errors_remaining["error"]),
        ),
        vcenter=plt_data.true_validity_threshold,
        vmax=max(
            np.amax(plt_data.errors_train["error"]),
            np.amax(plt_data.errors_val["error"]),
            np.amax(plt_data.errors_test["error"]),
            np.amax(plt_data.errors_remaining["error"]),
        ),
    )

    markersize = 20

    # scatter remaining data points with color indicating the error
    axs.scatter(
        plt_data.x_remaining.iloc[:, 0],  # axis 1
        plt_data.x_remaining.iloc[:, 1],  # axis 2
        c=plt_data.errors_remaining["error"],  # color of points (error)
        norm=divnorm,
        cmap=colors.LinearSegmentedColormap.from_list(
            "gwr", [green, darkgrey, red], N=256
        ),
        marker=".",
        label="Evaluation Data",
        alpha=1,
    )
    # scatter training data points of ml model with color indicating the error and edgecolor
    axs.scatter(
        plt_data.x_train_ml_model.iloc[:, 0],  # axis 1
        plt_data.x_train_ml_model.iloc[:, 1],  # axis 2
        c=plt_data.errors_train_ml_model["error"],  # color of points (error)
        norm=divnorm,
        cmap=colors.LinearSegmentedColormap.from_list(
            "gwr", [green, darkgrey, red], N=256
        ),
        edgecolors=black,
        linewidths=1.5,
        label="Training Data",
        s=markersize,
    )

    # scatter test data points with color indicating the error and edgecolor
    axs.scatter(
        plt_data.x_test_ml_model.iloc[:, 0],  # axis 1
        plt_data.x_test_ml_model.iloc[:, 1],  # axis 2
        c=plt_data.errors_test_ml_model["error"],  # color of points (error)
        norm=divnorm,
        cmap=colors.LinearSegmentedColormap.from_list(
            "gwr", [green, darkgrey, red], N=256
        ),
        edgecolors=blue,
        linewidths=1.5,
        label="Test Data",
        s=markersize,
    )

    triang = Triangulation(plt_data.x_grid.iloc[:, 0], plt_data.x_grid.iloc[:, 1])
    c1 = axs.tricontour(
        triang,
        plt_data.errors_grid["error"],
        levels=[plt_data.true_validity_threshold],
        linewidths=2.5,
        colors=[petrol],
    )
    h1, _ = c1.legend_elements()

    # Classifier decision boundary
    c2 = axs.tricontour(
        triang,
        plt_data.n_score_grid,
        levels=[plt_data.n_score_threshold],
        linewidths=2.5,
        colors=[violett],
    )
    h2, _ = c2.legend_elements()

    # Legend
    handles, labels = axs.get_legend_handles_labels()
    handles.append(h1[0])
    labels.append("Validity Domain")
    handles.append(h2[0])
    labels.append("Extrapolation Boundary")

    # Axs limits
    axs.set_xlim(-7.5, 20)
    axs.set_ylim(0, 4.5)

    axs.set_title(plt_data.plot_title)
    print(plt_data.plot_title)

    # Axs labels
    # axs.set_xlabel(r'$\mathrm{T}_\mathrm{amb}$ in °C')
    # axs.set_ylabel(r'$\mathrm{P}_\mathrm{el}$ in kW')

    return labels, handles, divnorm, color_dict


def plot_single(plt_data: PlotData2D):
    # Subplots
    fig, axs = plt.subplots(1, 1, figsize=(5.4, 4.1))
    plt.subplots_adjust(
        left=0.11, right=0.89, bottom=0.12, top=0.85, wspace=0.05, hspace=0.2
    )

    labels, handles, divnorm, color_dict = _plot_subplot(plt_data, axs)

    # Colorbar
    ticks = (
        np.linspace(divnorm.vmin + 0.01, divnorm.vcenter, 3).tolist()
        + np.linspace(divnorm.vcenter, divnorm.vmax - 0.01, 3).tolist()
    )
    # ticks = [0.01, 0.05, 0.1, 2, 4] # custom ticks
    cb = plt.colorbar(
        ScalarMappable(
            norm=divnorm,
            cmap=colors.LinearSegmentedColormap.from_list(
                "gwr",
                [color_dict["green"], color_dict["darkgrey"], color_dict["red"]],
                N=256,
            ),
        ),
        ax=axs,
        fraction=0.05,
        # orientation='horizontal',
        ticks=[round(i, 2) for i in ticks],
        label="Absolute Prediction Error of ANN in kW",
    )
    # cb.ax.set_yticklabels([0, 1, 2, 3, 4, 5, 6, 7, 8])

    # Legend
    handles[0], handles[1], handles[2], handles[3], handles[4] = (
        handles[1],
        handles[3],
        handles[2],
        handles[4],
        handles[0],
    )
    labels[0], labels[1], labels[2], labels[3], labels[4] = (
        labels[1],
        labels[3],
        labels[2],
        labels[4],
        labels[0],
    )
    axs.legend(
        handles,
        labels,
        bbox_to_anchor=(0.55, 1.02),
        ncol=3,
        loc="lower center",
        columnspacing=0.3,
    )

    # Axs labels
    axs.set_xlabel(plt_data.x_train.columns[0])
    # axs.set_xlabel(r"$\mathrm{T}_\mathrm{amb}$ in °C")
    axs.set_ylabel(plt_data.x_train.columns[1])
    # axs.set_ylabel(r"$\mathrm{P}_\mathrm{el}$ in kW")

    plt.title(plt_data.plot_title)

    return plt


def plot_3(plt_data_list: list[PlotData2D]):
    assert len(plt_data_list) == 3

    fig, axs = plt.subplots(3, 2, figsize=(5.4, 4.8), height_ratios=[1, 0, 1])
    plt.subplots_adjust(
        left=0.075, right=0.98, bottom=0.09, top=0.95, wspace=0.05, hspace=0.2
    )

    handles = None
    labels = None
    divnorm = None
    for row in range(0, 3):
        for column in range(0, 2):
            if row == 1 or row == 3:
                axs[row, column].remove()
            else:
                if row == 0:
                    indx = column
                elif row == 2:
                    indx = column + 2
                # elif row == 4:
                #     indx = column + 4
                if indx < 3:
                    labels, handles, divnorm, color_dict = _plot_subplot(
                        plt_data_list[indx], axs[row, column]
                    )
                    if indx != 1 and indx != 2:  # and indx != 2:
                        axs[row, column].set_xticklabels([])
                    else:
                        axs[row, column].set_xlabel(
                            plt_data_list[0].x_train.columns[0],
                            labelpad=0
                            # r"$\mathrm{T}_\mathrm{amb}$ in °C", labelpad=0
                        )

                    if indx != 0 and indx != 2 and indx != 4:
                        axs[row, column].set_yticklabels([])
                    else:
                        axs[row, column].set_ylabel(
                            plt_data_list[0].x_train.columns[1],
                            labelpad=0
                            # r"$\mathrm{P}_\mathrm{el}$ in kW", labelpad=0
                        )
                else:
                    axs[row, column].axis("off")

    handles[0], handles[1], handles[2] = handles[1], handles[2], handles[0]
    labels[0], labels[1], labels[2] = labels[1], labels[2], labels[0]

    axs[2, 1].legend(handles, labels)
    ticks = (
        np.linspace(divnorm.vmin + 0.01, divnorm.vcenter, 3).tolist()
        + np.linspace(divnorm.vcenter, divnorm.vmax - 0.01, 3).tolist()
    )
    # ticks = [0.01, 0.05, 0.1, 4, 8]
    cb = plt.colorbar(
        ScalarMappable(
            norm=divnorm,
            cmap=colors.LinearSegmentedColormap.from_list(
                "gwr",
                [color_dict["green"], color_dict["darkgrey"], color_dict["red"]],
                N=256,
            ),
        ),
        ax=axs[2, 1],
        fraction=0.05,
        orientation="horizontal",
        ticks=[round(i, 2) for i in ticks],
        label="Absolute Prediction Error \n of ANN in kW",
    )
    cb.ax.xaxis.set_label_position("top")

    return plt


def plot_5(plt_data_list: list[PlotData2D]):
    assert len(plt_data_list) == 5

    fig, axs = plt.subplots(5, 2, figsize=(5.4, 7.2), height_ratios=[1, 0, 1, 0, 1])
    plt.subplots_adjust(
        left=0.075, right=0.98, bottom=0.065, top=0.965, wspace=0.05, hspace=0.2
    )

    handles = None
    labels = None
    divnorm = None

    for row in range(0, 5):
        for column in range(0, 2):
            if row == 1 or row == 3:
                axs[row, column].remove()
            else:
                if row == 0:
                    indx = column
                elif row == 2:
                    indx = column + 2
                elif row == 4:
                    indx = column + 4

                if indx < 5:
                    labels, handles, divnorm, color_dict = _plot_subplot(
                        plt_data_list[indx], axs[row, column]
                    )
                    if indx != 3 and indx != 4:  # and indx != 2:
                        axs[row, column].set_xticklabels([])
                    else:
                        axs[row, column].set_xlabel(
                            plt_data_list[0].x_train.columns[0],
                            labelpad=0
                            # r"$\mathrm{T}_\mathrm{amb}$ in °C", labelpad=0
                        )

                    if indx != 0 and indx != 2 and indx != 4:
                        axs[row, column].set_yticklabels([])
                    else:
                        axs[row, column].set_ylabel(
                            plt_data_list[0].x_train.columns[1],
                            labelpad=0
                            # r"$\mathrm{P}_\mathrm{el}$ in kW", labelpad=0
                        )
                else:
                    axs[row, column].axis("off")

    handles[0], handles[1], handles[2] = handles[1], handles[2], handles[0]
    labels[0], labels[1], labels[2] = labels[1], labels[2], labels[0]

    axs[4, 1].legend(handles, labels)
    ticks = (
        np.linspace(divnorm.vmin + 0.01, divnorm.vcenter, 3).tolist()
        + np.linspace(divnorm.vcenter, divnorm.vmax - 0.01, 3).tolist()
    )
    # ticks = [0.01, 0.04, 0.08, 4, 8]
    cb = plt.colorbar(
        ScalarMappable(
            norm=divnorm,
            cmap=colors.LinearSegmentedColormap.from_list(
                "gwr",
                [color_dict["green"], color_dict["darkgrey"], color_dict["red"]],
                N=256,
            ),
        ),
        ax=axs[4, 1],
        fraction=0.05,
        orientation="horizontal",
        ticks=[round(i, 2) for i in ticks],
        label="Absolute Prediction Error \n of ANN in kW",
    )
    cb.ax.xaxis.set_label_position("top")

    return plt


def plot_bar_plot(bars, clf_labels, bars_ideal=None, labels=None, ylabel=None):
    # todo:
    blue = (0 / 255, 84 / 255, 159 / 255)
    green = (87 / 255, 171 / 255, 39 / 255)
    violett = (97 / 255, 33 / 255, 88 / 255)
    black = (0, 0, 0)
    magenta = (227 / 255, 0 / 255, 102 / 255)
    orange = (246 / 255, 168 / 255, 0 / 255)
    darkred = (114 / 255, 29 / 255, 19 / 255)
    color_list = [blue, black, green, darkred, orange, magenta, violett]

    n_uc = bars.shape[1]
    n_clfs = bars.shape[0]
    bar_width = 0.8 / n_clfs
    r1 = np.arange(n_uc)

    fig = plt.figure(figsize=(5.4, 4))
    plt.ylim(0, 1)

    for i in range(n_clfs):
        r = [x + bar_width * i for x in r1]
        bar = bars[i, :]
        if bars_ideal is not None:
            bar_ideal = bars_ideal[i, :]
            yerr = np.zeros((1, n_uc))
            yerr[0, :] = bar_ideal - bar
            # for j in range(len(bar)):
            #     plt.hlines(bar[j] + yerr[0][j], r[j] - bar_width/2, r[j] + bar_width/2, color=color_list[i])
            #     plt.vlines(r[j] - bar_width/2 * 0.93, bar[j], bar[j] + yerr[0][j], color=color_list[i])
            #     plt.vlines(r[j] + bar_width / 2 * 0.93, bar[j], bar[j] + yerr[0][j], color=color_list[i])
            plt.bar(
                r,
                bar_ideal,
                # bottom=bar,
                hatch="//",
                width=bar_width * 0.97,
                color="none",
                edgecolor=color_list[i],
            )

        plt.bar(
            r,
            bar,
            width=bar_width * 0.97,
            color=color_list[i],
            edgecolor=color_list[i],
            label=clf_labels[i],
        )

    if labels is None:
        plt.xticks([x + bar_width * (n_clfs - 1) / 2 for x in r1])
    else:
        plt.xticks([x + bar_width * (n_clfs - 1) / 2 for x in r1], labels)

    if ylabel is not None:
        plt.ylabel(ylabel)

    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig("plots\\barplot.pdf", bbox_inches="tight")
    plt.show()
