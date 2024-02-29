import os

from extrapolation_detection.machine_learning_util import data_handling
from extrapolation_detection.new_use_cases.ed_experiment_config import (
    ExtrapolationExperimentConfig,
)
from core.util.data_handling import split_target_features

from extrapolation_detection.plotting.plot import Plotter


def exe_plot_2D(experiment_name, detector_experiment_names):
    if not os.path.exists("plots"):
        os.mkdir("plots")

    p = Plotter()


    # read experiment data
    p.x_train = data_handling.read_csv("x_train", directory=experiment_name)
    p.y_train = data_handling.read_csv("y_train", directory=experiment_name)
    p.x_val = data_handling.read_csv("x_val", directory=experiment_name)
    p.y_val = data_handling.read_csv("y_val", directory=experiment_name)
    p.x_test = data_handling.read_csv("x_test", directory=experiment_name)
    p.y_test = data_handling.read_csv("y_test", directory=experiment_name)
    p.x_remaining = data_handling.read_csv("x_remaining", directory=experiment_name)
    p.y_remaining = data_handling.read_csv("y_remaining", directory=experiment_name)
    p.x_grid = data_handling.read_csv("x_grid", directory=experiment_name)
    p.y_grid = data_handling.read_csv("y_grid", directory=experiment_name)

    p.xy_training = data_handling.read_csv("xy_train", directory=experiment_name)
    p.xy_validation = data_handling.read_csv("xy_val", directory=experiment_name)
    p.xy_test = data_handling.read_csv("xy_test", directory=experiment_name)
    p.xy_remaining = data_handling.read_csv("xy_remaining", directory=experiment_name)
    p.xy_grid = data_handling.read_csv("xy_grid", directory=experiment_name)



    # read errors
    p.errors_train = data_handling.read_csv("errors_train", directory=experiment_name)
    p.errors_val = data_handling.read_csv("errors_val", directory=experiment_name)
    p.errors_test = data_handling.read_csv("errors_test", directory=experiment_name)
    p.errors_remaining = data_handling.read_csv(
        "errors_remaining", directory=experiment_name
    )
    p.errors_grid = data_handling.read_csv("errors_grid", directory=experiment_name)

    # read true_validity
    p.true_validity_train = data_handling.read_csv(
        "true_validity_train", experiment_name
    ).squeeze()
    p.true_validity_val = data_handling.read_csv(
        "true_validity_val", experiment_name
    ).squeeze()
    p.true_validity_test = data_handling.read_csv(
        "true_validity_test", experiment_name
    ).squeeze()
    p.true_validity_remaining = data_handling.read_csv(
        "true_validity_remaining", experiment_name
    ).squeeze()

    for detector in detector_experiment_names:
        # read detector data
        p.n_score_train = data_handling.read_csv(
            f"n_score_train_{detector}", directory=experiment_name
        ).squeeze()
        p.n_score_val = data_handling.read_csv(
            f"n_score_val_{detector}", directory=experiment_name
        ).squeeze()
        p.n_score_test = data_handling.read_csv(
            f"n_score_test_{detector}", directory=experiment_name
        ).squeeze()
        p.n_score_remaining = data_handling.read_csv(
            f"n_score_remaining_{detector}", directory=experiment_name
        ).squeeze()
        p.novelty_threshold = data_handling.read_csv(
            f"{detector}_threshold", directory=os.path.join(experiment_name, "detectors")
        ).iloc[0, 0]

        # Plot data
        p._plot_subplot("a", "B")

if __name__ == '__main__':
    experiment_name = "Carnot_Test2"
    detector_experiment_names = ["KNN_val+test"]

    exe_plot_2D(experiment_name, detector_experiment_names)
