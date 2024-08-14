import os

from addmo.util.load_save import load_config_from_json

from aixtra.util import loading_saving_aixtra
from aixtra_use_case.config.ed_experiment_config import (
    ExtrapolationExperimentConfig,
)

from aixtra.plotting import plot
from aixtra.plotting import plot_carpets
from addmo.util.experiment_logger import ExperimentLogger
from aixtra.util import loading_saving_aixtra
from addmo.s3_model_tuning.models.abstract_model import AbstractMLModel
from addmo.util.data_handling import split_target_features
from aixtra.extrapolation_detection.n_D_extrapolation.score_regressor_per_data_point import (
    score_per_sample,
)
from aixtra_use_case.config.ed_experiment_config import (
    ExtrapolationExperimentConfig,
)
from aixtra.exploration_quantification import point_generator
from aixtra.system_simulations.system_simulations import system_factory


def exe(config: ExtrapolationExperimentConfig):
    # load model
    regressor: AbstractMLModel = loading_saving_aixtra.load_regressor("regressor", directory=os.path.join(config.experiment_folder, "regressors"))

    x_grid = loading_saving_aixtra.read_csv(
        "x_grid", directory=config.experiment_folder
    )

    # define bounds
    bounds = point_generator.infer_or_forward_bounds(
        config.config_explo_quant.exploration_bounds, x_grid
    )

    # delete the target variable from bounds
    bounds.pop(config.name_of_target, None)
    bounds.pop("y_pred", None)

    # get system function
    system_function = system_factory(config.system_simulation)
    regressor_function = plot_carpets.prediction_func_4_regressor(regressor)

    # hard code defaults for nicer appearance
    if config.simulation_data_name.startswith("bes"):
        combinations = [
            ("u_hp", "t_amb"),
            ("u_hp", "t_room"),
            ("u_hp", "rad_dir"),
            ("t_amb", "t_room"),
            ("t_amb", "rad_dir"),
            ("rad_dir", "t_room"),
        ]
        defaults_dict = {"t_amb": 273.15, "rad_dir": 0, "u_hp": 0.5, "t_room": 273.15+20}
    else:
        combinations = None
        defaults_dict = None

    # plot
    carpets_plot = plot_carpets.plot_system_carpets(bounds, system_function, regressor_function, combinations=combinations, defaults_dict=defaults_dict)

    plot.save_plot(carpets_plot, "carpets_system", config.experiment_folder)

    print(f"{__name__} executed")

if __name__ == "__main__":
    config = ExtrapolationExperimentConfig()

    path = r"/aixtra/extrapolation_detection\use_cases\results\Boptest_TAir_mid_ODE_test1_supersmallANN\config.json"
    config = load_config_from_json(path, ExtrapolationExperimentConfig())
    exe(config)
