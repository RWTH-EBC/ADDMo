import os
import unittest
import inspect
import tempfile
import pandas as pd
from unittest.mock import  patch, MagicMock
from addmo.util.experiment_logger import ExperimentLogger,LocalLogger, WandbLogger
from addmo.s1_data_tuning_auto.config.data_tuning_auto_config import DataTuningAutoSetup
from addmo.s1_data_tuning_auto.data_tuner_auto import DataTunerAuto
from addmo.s2_data_tuning.config.data_tuning_config import DataTuningFixedConfig
from addmo.s2_data_tuning.data_tuner_fixed import DataTunerByConfig
from addmo.util.load_save import load_data
from addmo.util.data_handling import split_target_features


# Define all the tuning classes here for testing:
TUNER_CLASSES = [DataTunerAuto, DataTunerByConfig]

CONFIG_MAP = {
    DataTunerAuto:   DataTuningAutoSetup,
    DataTunerByConfig: DataTuningFixedConfig,}
class TestAllDataTuners(unittest.TestCase):
    """
    For each tuner class, discover all methods named `tune_*` e.g. tune_auto, tune_fixed
    and invoke them with the “correct” arguments. Then assert the returned `tuned_x` is a non-empty DataFrame & `y` is non-empty Series.
    """

    def _instantiate_config(self, tuner_cls):
        sig = inspect.signature(tuner_cls.__init__)
        param = sig.parameters.get("config")
        anno = getattr(param, "annotation", inspect._empty)

        if anno not in (inspect._empty, object):
            config_cls = anno
        else:
            # fallback to explicit map
            config_cls = CONFIG_MAP[tuner_cls]

        return config_cls()

    def _make_data(self, config):
        raw = load_data(config.abs_path_to_data)
        x, y = split_target_features(config.name_of_target, raw)
        return raw, x, y

    def test_tuners(self):
        for tuner_cls in TUNER_CLASSES:
            with self.subTest(tuner=tuner_cls.__name__):
                config = self._instantiate_config(tuner_cls)
                tuner = tuner_cls(config=config)
                raw, x, y = self._make_data(config)
                for name, method in inspect.getmembers(tuner, predicate=inspect.ismethod):
                    if not name.startswith("tune_"):
                        continue

                    with self.subTest(method=name):
                        sig = inspect.signature(method)
                        num_args = len(sig.parameters)

                        if num_args == 0:
                            tuned_x = method()
                        elif num_args == 1:
                            tuned_x = method(raw)
                        elif num_args == 2:
                            tuned_x = method(x, y)
                        else:
                            self.skipTest(f"{name} has unexpected signature {sig}")

                        self.assertIsInstance(tuned_x, pd.DataFrame,
                                              f"{name} must return DataFrame")
                        self.assertFalse(tuned_x.empty,
                                         f"{name} returned empty DataFrame")

                        y_out = getattr(tuner, "y", y)
                        self.assertIsInstance(y_out, pd.Series,
                                              f"{name}: y must be Series")
                        self.assertFalse(y_out.empty,
                                         f"{name}: y must not be empty")

                        joined = pd.concat([y_out, tuned_x], axis=1).bfill()
                        self.assertIsInstance(joined, pd.DataFrame)
                        self.assertFalse(joined.empty,
                                         f"{name}: joined DataFrame empty")


if __name__ == "__main__":
    unittest.main()