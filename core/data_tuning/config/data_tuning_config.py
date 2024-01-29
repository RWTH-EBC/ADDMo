from core.util.abstract_config import BaseConfig

class DataTuningFixedConfig(BaseConfig):
    def __init__(self):
        self.config_as_dict = None # safe the config as dict for specific use cases

        # yaml blue print starts here
        self.path_to_raw_data = r"D:\04_GitRepos\addmo-extra\raw_input_data\InputData.xlsx"  #
        # absolute path to raw data
        self.name_of_raw_data = "test_raw_data"  # name of raw data
        self.name_of_tuning = "test_fixed_reproduction_tuning" # name of tuning

        # self.period = {
        #     'start': "10.10.2023 00:00:00",  # None or in DD.MM.YYYY HH:mm:ss format
        #     'end': "10.10.2023 23:59:59",  # None or in DD.MM.YYYY HH:mm:ss format
        # }

        self.target = "Total active power"  # Output of prediction
        self.features = [
            "Schedule",
            "FreshAir Temperature",
            "FreshAir Temperature___diff",
            "FreshAir Temperature___lag1",
            "FreshAir Temperature___squared",
            "Total active power___lag1",
        ]

