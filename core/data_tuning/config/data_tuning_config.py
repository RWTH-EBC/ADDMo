from core.util.abstract_config import BaseConfig

class DataTuningFixedConfig(BaseConfig):
    def __init__(self):
        # absolute path to raw data
        self.path_to_raw_data = r"D:\04_GitRepos\addmo-extra\raw_input_data\InputData.xlsx"
        self.name_of_raw_data = "test_raw_data"
        self.name_of_tuning = "test_fixed_reproduction_tuning"
        self.target = "Total active power"  # Output of prediction
        self.features = [
            "Schedule",
            "FreshAir Temperature",
            "FreshAir Temperature___diff",
            "FreshAir Temperature___lag1",
            "FreshAir Temperature___squared",
            "Total active power___lag1",
        ]

