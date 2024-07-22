def sweep_hidden_layer_sizes():
    hidden_layer_sizes = []

    # Single layer possibilities
    for neurons in [5, 10, 50, 100, 1000]:
        hidden_layer_sizes.append([neurons])

    # Two layer possibilities
    for neurons1 in [5, 10, 50, 100]:
        for neurons2 in [5, 10, 50, 100]:
            hidden_layer_sizes.append([neurons1, neurons2])

    # # Three layer possibilities
    # for neurons1 in [5, 10, 100, 1000]:
    #     for neurons2 in [5, 10, 100, 1000]:
    #         for neurons3 in [5, 10, 100, 1000]:
    #             hidden_layer_sizes.append([neurons1, neurons2, neurons3])

    sweep_configuration = {
        "name": "hidden_layer_sizes",
        "method": "grid",
        "metric": {"name": "coverage_true_validity", "goal": "maximize"},
        "parameters": {
            "repetition": {"values": [1, 2, 3]},
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
        "name": "tuned_splitting_sweep",
        "method": "grid",
        "metric": {"name": "coverage_true_validity", "goal": "maximize"},
        "parameters": {
            "repetition": {"values": [1, 2, 3]},
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

def sweep_repetitions_only():
    sweep_configuration = {
        "name": "repetitions_only_sweep",
        "method": "grid",
        "metric": {"name": "coverage_true_validity", "goal": "maximize"},
        "parameters": {
            "repetition": {"values": [1, 2, 3]},
        }
    }
    return sweep_configuration