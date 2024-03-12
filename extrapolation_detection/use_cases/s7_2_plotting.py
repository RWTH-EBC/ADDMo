import os

from extrapolation_detection.plotting import plot
from extrapolation_detection.use_cases.config.ed_experiment_config import (
    ExtrapolationExperimentConfig,
)


def exe_plot_2D_all(config):
    # plot for all detectors that are saved in the directory
    directory = os.path.join(config.experiment_folder, "detectors")
    for detector_file in os.listdir(directory):
        if detector_file.endswith(".pkl"):
            # get name without file ending
            detector_name = detector_file.split(".")[0]

            plot_data = plot.PlotData2D()
            plot_data.load_plot_data(config.experiment_folder, detector_name)
            plot_data.infer_ml_model_data_splits()
            plt = plot.plot_single(plot_data)
            plot.save_plot(plt, detector_name, config.experiment_folder)
            plot.show_plot(plt)


def exe_plot_2D(experiment_name, detector_experiment_names:list[str]):
    for detector in detector_experiment_names:
        plot_data = plot.PlotData2D()
        plot_data.load_plot_data(experiment_name, detector)
        plot_data.infer_ml_model_data_splits()
        plt = plot.plot_single(plot_data)
        plot.show_plot(plt)


if __name__ == "__main__":
    config = ExtrapolationExperimentConfig()
    exe_plot_2D_all(config)

    # experiment_name = "Carnot_Test2"
    # detector_experiment_names = ["KNN_val+test"]
    #
    # exe_plot_2D(experiment_name, detector_experiment_names)
