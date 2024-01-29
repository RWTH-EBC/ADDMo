import yaml
def load_yaml_to_dict(path_to_yaml):
    """
    Load a yaml file to a dictionary.
    """
    with open(path_to_yaml, "r") as f:
        config = yaml.safe_load(f)
    return config

def save_yaml_from_dict(path_to_yaml, dict_to_save):
    """
    Save a dictionary to a yaml file.
    """
    with open(path_to_yaml, "w") as f:
        yaml.dump(dict_to_save, f)

