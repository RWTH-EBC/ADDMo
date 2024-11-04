import os
import wandb
from addmo.util.experiment_logger import ExperimentLogger
from addmo.util.experiment_logger import LocalLogger
from addmo.util.experiment_logger import WandbLogger
from addmo.util.definitions import create_or_clean_directory

from aixtra_use_case.config.ed_experiment_config import (
    ExtrapolationExperimentConfig,
)
from addmo.util.load_save import load_config_from_json
from aixtra.util import update_wandb_sweep as uws
from aixtra_use_case import (
    s1_split_data,
    s2_tune_ml_regressor,
    s3_regressor_error_calculation,
    s4_gradient,
    s4_true_validity_domain,
    s5_tune_detector,
    s6_detector_score_calculation,
    s7_2_plotting,
    s8_0_generate_grid,
    s8_1_coverage_convex_hull,
    s8_2_coverage_grid_occupancy,
    s8_3_coverage_tuned_ND,
    s8_4_coverage_true_validity,
    s8_5_coverage_gradient,
    s9_data_coverage,
    s9_data_coverage_grid,
    s9_carpet_plots,
)

def rerun(input_path):
    class ExtrapolationExperimentConfigRepeater(ExtrapolationExperimentConfig):
        @property
        def experiment_folder(self):
            return input_path

    config = load_config_from_json(os.path.join(input_path, "local_logger", "config.json"), ExtrapolationExperimentConfigRepeater)

    # Configure the logger
    LocalLogger.directory = os.path.join(config.experiment_folder, "local_logger")
    LocalLogger.active = True
    WandbLogger.active = False

    # Run scripts
    # ExperimentLogger.start_experiment(config=config)  # log config
    # s1_split_data.exe(config)
    # s2_tune_ml_regressor.exe(config)
    # s3_regressor_error_calculation.exe(config)
    # s4_true_validity_domain.exe(config)
    # s4_gradient.exe(config)
    # s5_tune_detector.exe(config)
    # s6_detector_score_calculation.exe(config)
    # s8_0_generate_grid.exe(config)
    # # s8_1_coverage_convex_hull.exe(config)
    # # s8_2_coverage_grid_occupancy.exe(config)
    log = {}
    for d in s8_3_coverage_tuned_ND.exe(config):
        log.update(d)
    # s8_4_coverage_true_validity.exe(config)
    # s8_5_coverage_gradient.exe(config)
    # s9_data_coverage.exe(config)
    # s9_data_coverage_grid.exe(config)
    # s9_carpet_plots.exe(config)

    # s7_2_plotting.exe_plot_2D_all(config)
    # s7_2_plotting.exe_plot_2D_detector(config)

    return log


if __name__ == '__main__':
    USER_NAME = "team-martinraetz"
    PROJECT_NAME = "8_Carnot_mid_noise_m0_std0.02"
    SWEEP_ID = "bl6kkkwh"

    local_folder = r"R:\_Dissertationen\mre\Diss\08_Data_Plots_Analysis\0_ADDMo_TrueValidityVSExtrapolationCovargeScores\8_Carnot_mid_noise_m0_std002\ANN"

    for run in uws.yield_runs_per_sweep(USER_NAME, PROJECT_NAME, SWEEP_ID):
        # with wandb.init(id=run.id, project=PROJECT_NAME, entity=USER_NAME, resume="must") as resumed_run:
        config = uws.get_config_from_run(run)
        exp_folder = os.path.join(local_folder, config["experiment_name"])
        print(config["experiment_name"])

        log = rerun(exp_folder)

        uws.update_run(run, summary_dict=log, config_dict=False)






