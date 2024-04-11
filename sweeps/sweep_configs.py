def sweep_hidden_layer_sizes():
    hidden_layer_sizes = []

    # Single layer possibilities
    for neurons in [5, 10, 100, 1000]:
        hidden_layer_sizes.append([neurons])

    # Two layer possibilities
    for neurons1 in [5, 10, 100, 1000]:
        for neurons2 in [5, 10, 100, 1000]:
            hidden_layer_sizes.append([neurons1, neurons2])

    sweep_configuration = {
        "name": "trial_sweep_ed_usecase",
        "method": "grid",
        "metric": {"name": "coverage_true_validity", "goal": "maximize"},
        "parameters": {
            "repetition": {"values": [1, 2, 3, 4, 5, 6, 7, 8]},
            "config_model_tuning": {
                "parameters": {
                    "hyperparameter_tuning_kwargs": {
                        "parameters": {
                            "hyperparameter_set": {
                                "parameters": {
                                    "hidden_layer_sizes": {"values": hidden_layer_sizes}
                                }
                            }
                        }
                    }
                }
            },
        },
    }
    return sweep_configuration


def sweep_several_tunings():
    sweep_configuration = {
        "name": "trial_sweep_ed_usecase",
        "method": "grid",
        "metric": {"name": "coverage_true_validity", "goal": "maximize"},
        "parameters": {
            "repetition": {"values": [1, 2, 3, 4, 5, 6, 7, 8]},
            "config_model_tuning": {
                "parameters": {
                    "validation_score_splitting": {
                        "values": ["UnivariateSplitter", "KFold"]
                    }
                }
            }
        }
    }

    return sweep_configuration
