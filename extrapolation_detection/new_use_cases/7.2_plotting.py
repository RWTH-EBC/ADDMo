import os

from extrapolation_detection.machine_learning_util import data_handling
from extrapolation_detection.new_use_cases.ed_experiment_config import (
    ExtrapolationExperimentConfig,
)
from core.util.data_handling import split_target_features

from extrapolation_detection.plotting import plot


def exe_plot_2D(experiment_name, detector_experiment_names):


    for detector in detector_experiment_names:
        plot_data = plot.PlotData()
        plot_data.load_plot_data(experiment_name, detector)
        plot_data.infer_ml_model_data_splits()

        # plt = plot.plot_single_customized(plot_data)
        plt = plot.plot_single_via_subplot(plot_data)
        plot.show_plot(plt)

    # plot_data.plot_title = "test"
    # plot_data_list = [plot_data, plot_data, plot_data]#, plot_data, plot_data]
    # plt = plot.plot_3(plot_data_list)
    # plot.show_plot(plt)



if __name__ == '__main__':
    experiment_name = "Carnot_Test2"
    detector_experiment_names = ["KNN_val+test"]

    exe_plot_2D(experiment_name, detector_experiment_names)
