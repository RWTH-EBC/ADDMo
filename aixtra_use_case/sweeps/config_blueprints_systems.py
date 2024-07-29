from aixtra_use_case.config.ed_experiment_config import ExtrapolationExperimentConfig

def config_ODEel_steady(config: ExtrapolationExperimentConfig):
    config.simulation_data_name = "ODEel_steady"
    config.experiment_name = "Empty"
    config.name_of_target = "delta_reaTZon_y"
    config.train_val_test_period = (29185, 35040) #November und Dezember
    # config.train_val_test_period = (23329, 26208, 32065, 35040) #September und Dezember
    config.shuffle = False
    config.grid_points_per_axis = 10
    config.system_simulation = "BopTest_TAir_ODEel"
    config.true_outlier_threshold = 0.1

    config.config_explo_quant.exploration_bounds = {
        "TDryBul": (263.15, 303.15),
        "HDirNor": (0, 1000),
        "oveHeaPumY_u": (0, 1),
        "reaTZon_y": (290.15, 300.15),
        "delta_reaTZon_y": (-0.5, 0.5),
    }
    return config

def config_carnot(config: ExtrapolationExperimentConfig):
    config.simulation_data_name = "Carnot_mid_noise_m0_std0.02"
    config.experiment_name = "Empty"
    config.name_of_target = "$\dot{Q}_{heiz}$ in kW"
    config.train_val_test_period = (0, 744)
    config.shuffle = False
    config.grid_points_per_axis = 10
    config.system_simulation = "carnot"
    config.true_outlier_threshold = 0.2

    config.config_explo_quant.exploration_bounds = {
        "$T_{umg}$ in °C": (-10, 30),
        "$P_{el}$ in kW": (0, 5),
        "$\dot{Q}_{heiz}$ in kW": (0, 35)
    }
    return config