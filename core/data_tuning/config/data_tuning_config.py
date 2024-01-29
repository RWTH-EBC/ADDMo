from core.util.load_save import load_yaml_to_dict

class DataTuningConfig():
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
    def load_yaml_to_class(self, path_to_yaml:dict):
        '''Loads the dict to a class object. Overwrites existing attributes. Only works for
        flat yaml files. And only for attributes that are already defined in the class.'''

        config_dict = load_yaml_to_dict(path_to_yaml)
        if config_dict is not None:
            for key, value in config_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        else:
            raise ValueError("YAML file is empty or not properly formatted.")

        self.config_as_dict = config_dict # safe the config as dict for specific use cases
