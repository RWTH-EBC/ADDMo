import os.path
from abc import ABC, abstractmethod
from core.util.load_save import load_yaml_to_dict, save_yaml_from_dict, create_or_override_directory


class BaseConfig(ABC):
    def load_yaml_to_class(self, path_to_yaml:str):
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

    def save_class_to_yaml(self, path_to_yaml:str):
        """
        Save the class attributes to a yaml file.
        """
        config_dict = {}
        for key, val in self.__dict__.items():
            if not key.startswith("__") and not callable(getattr(self, key)):
                if isinstance(val, (int, float, str, bool, list, tuple, dict)):
                    config_dict[key] = val

        save_yaml_from_dict(path_to_yaml, config_dict)