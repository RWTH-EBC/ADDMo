from abc import ABC, abstractmethod
from core.util.load_save import load_yaml_to_dict

class BaseConfig(ABC):
    def load_yaml_to_class(self, path_to_yaml:dict):
        """
        Load data from config.yaml and dynamically assign to the class variables.
        Overwrites existing attributes. Only works for flat yaml files. And only for attributes
        that are already defined in the class."""

        config_dict = load_yaml_to_dict(path_to_yaml)
        if config_dict is not None:
            for key, value in config_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        else:
            raise ValueError("YAML file is empty or not properly formatted.")
