import os

from addmo.util.experiment_logger import ExperimentLogger
from aixtra.util import loading_saving_aixtra
from aixtra_use_case.config.ed_experiment_config import (
    ExtrapolationExperimentConfig,
)

from aixtra.plotting import plot

from aixtra.exploration_quantification.exploration_quantifier import ExplorationQuantifier
from aixtra.exploration_quantification.coverage_plotting import (
    plot_scatter_average_coverage_per_2D,
)


def exe(config: ExtrapolationExperimentConfig):

    grid_path = os.path.join(config.experiment_folder, "explo_quant")
    x_grid = loading_saving_aixtra.read_csv(f"grid_points", grid_path)


    directory = os.path.join(config.experiment_folder, "detectors")
    for detector_file in os.listdir(directory):
        if detector_file.endswith(".pkl") and (detector_file.replace("_test+fit", "").replace(".pkl", "") in config.config_detector.detectors): #TODO:Attention hard code only works for _test+fit
            # get name without file ending
            detector_name = detector_file.split(".")[0]
            detector = loading_saving_aixtra.read_pkl(detector_name, directory=directory)

            quantifier = ExplorationQuantifier()
            quantifier.explo_clf = detector # set the tuned extrapolation detector as the classifier
            y_grid = quantifier.calc_labels(x_grid=x_grid)
            coverage = quantifier.calculate_coverage()

            plots_per_axes = plot_scatter_average_coverage_per_2D(
                x_grid=x_grid,
                y_grid=y_grid,
                title_header=f"Extra-{detector_name}\n"
                             f"Coverage = {coverage.loc['Inside']:.2f} %",
            )

            for i, plt in enumerate(plots_per_axes):
                plot.save_plot(
                    plt,
                    f"coverage_extra_{detector_name}_{i}",
                    config.experiment_folder,
                )
                # plot.show_plot(plt)

            # save
            save_path = os.path.join(config.experiment_folder, "explo_quant")
            loading_saving_aixtra.write_csv(
                quantifier.labels_grid,
                f"labels_grid_extra_{detector_name}",
                save_path,
            )
            loading_saving_aixtra.write_csv(
                coverage,
                f"coverage_percentage_extra_{detector_name}",
                save_path,
            )

            # log coverage
            ExperimentLogger.log({f"coverage_extra_{detector_name}": coverage.loc["Inside"]})
            # yield {f"coverage_extra_{detector_name}": coverage.loc["Inside"]} # Todo: delete, only for rerunning


    print(f"{__name__} executed")


if __name__ == "__main__":
    config = ExtrapolationExperimentConfig()
    exe(config)
