import pandas as pd
import wandb
import os
from abc import ABC, abstractmethod


from core.util.pickle_handling import write_pkl, read_pkl



import wandb
import os
from abc import ABC, abstractmethod
from core.util.pickle_handling import write_pkl, read_pkl

class AbstractLogger(ABC):

    @staticmethod
    @abstractmethod
    def start_experiment(config: dict = None, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def finish_experiment():
        pass

    @staticmethod
    @abstractmethod
    def log(log: dict):
        pass

    @staticmethod
    @abstractmethod
    def log_artifact(data, name: str, art_type: str):
        pass

    @staticmethod
    @abstractmethod
    def use_artifact(name: str, alias: str = 'latest'):
        pass



class WandbLogger(AbstractLogger):
    active: bool = False
    project = None # Name of the Weights & Biases project that you created in the browser wandb.ai
    directory = None # Local directory where a backup of the uploaded files is stored

    @staticmethod
    def start_experiment(config=None, **kwargs):
        """"Starts a new experiment and logs the config to wandb."""
        wandb.init(project=WandbLogger.project, config=config, **kwargs, dir=WandbLogger.directory)
        return wandb.config

    @staticmethod
    def finish_experiment():
        """Finishes the current experiment."""
        wandb.finish()

    @staticmethod
    def log(log: dict):
        for key, value in log.items():
            if isinstance(value, pd.DataFrame):
                table = wandb.Table(dataframe=value)
                wandb.log({key: table})


    @staticmethod
    @only_if_active
    def log_artifact(data, name: str, art_type: str):
        artifact = wandb.Artifact(name=name, type=art_type)
        artifact.add_file(_get_path(name))
        wandb.run.log_artifact(artifact)
        artifact.wait()

    @staticmethod
    @only_if_active
    def use_artifact(name: str, alias: str = 'latest'):
        artifact = wandb.use_artifact(f'{name}:{alias}')
        filename = artifact.download() + '\\' + name + '.pkl'
        return read_pkl(filename)

class LocalLogger(AbstractLogger):

    directory = None

    @staticmethod
    @only_if_active
    def start_experiment(config, **kwargs):
        return config

    @staticmethod
    @only_if_active
    def finish_experiment():
        pass

    @staticmethod
    @only_if_active
    def log(log: dict):
        pass

    @staticmethod
    @only_if_active
    def log_artifact(data, name: str, art_type: str):
        if art_type == 'data':
            file_path = os.path.join(LocalLogger.directory, name + '.csv')
            data.to_csv(file_path)

    @staticmethod
    @only_if_active
    def use_artifact(name: str, alias: str = 'latest'):
        filename = name
        return read_pkl(filename, LocalLogger.directory)



class ExperimentLogger(AbstractLogger):
    """Static class to trigger the different loggers. A static class can be used throughout the
    whole code without the need to pass it as an argument."""

    @staticmethod
    def start_experiment(config=None, **kwargs):
        config_wandb = WandbLogger.start_experiment(config, **kwargs)
        config_local = LocalLogger.start_experiment(config, **kwargs)
        return config_wandb or config_local

    @staticmethod
    def finish_experiment():
        WandbLogger.finish_experiment()
        LocalLogger.finish_experiment()

    @staticmethod
    def log(log: dict):
        WandbLogger.log(log)
        LocalLogger.log(log)

    @staticmethod
    def log_artifact(data, name: str, art_type: str):
        WandbLogger.log_artifact(data, name, art_type)
        LocalLogger.log_artifact(data, name, art_type)

    @staticmethod
    def use_artifact(name: str, alias: str = 'latest'):
        data_wandb = WandbLogger.use_artifact(name, alias)
        data_local = LocalLogger.use_artifact(name, alias)
        return data_wandb or data_local
