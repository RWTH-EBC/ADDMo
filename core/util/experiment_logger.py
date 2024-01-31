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

    project = None # Weights & Biases project
    directory = None # Local directory

    @staticmethod
    def start_experiment(config=None, **kwargs):
        wandb.init(project=WandbLogger.project, config=config, **kwargs, dir=WandbLogger.directory)
        return wandb.config

    @staticmethod
    def finish_experiment():
        wandb.finish()

    @staticmethod
    def log(log: dict):
        wandb.log(log)

    @staticmethod
    def log_artifact(data, name: str, art_type: str):
        artifact = wandb.Artifact(name=name, type=art_type)
        artifact.add_file(_get_path(name))
        wandb.run.log_artifact(artifact)
        artifact.wait()

    @staticmethod
    def use_artifact(name: str, alias: str = 'latest'):
        artifact = wandb.use_artifact(f'{name}:{alias}')
        filename = artifact.download() + '\\' + name + '.pkl'
        return read_pkl(filename)

class LocalLogger(AbstractLogger):

    directory = None

    @staticmethod
    def start_experiment(config, **kwargs):
        return config

    @staticmethod
    def finish_experiment():
        pass

    @staticmethod
    def log(log: dict):
        pass

    @staticmethod
    def log_artifact(data, name: str, art_type: str):
        if art_type == 'data':
            file_path = os.path.join(LocalLogger.directory, name + '.csv')
            data.to_csv(file_path)

    @staticmethod
    def use_artifact(name: str, alias: str = 'latest'):
        filename = name
        return read_pkl(filename, LocalLogger.directory)



class ExperimentLogger(AbstractLogger):
    """Static class to trigger the different loggers. A static class can be used throughout the
    whole code without the need to pass it as an argument."""

    wandb_logger = None
    local_logger = None

    @staticmethod
    def start_experiment(config=None, **kwargs):
        config_wandb = None
        config_local = None

        if ExperimentLogger.wandb_logger:
            config_wandb = WandbLogger.start_experiment(config, **kwargs)
        if ExperimentLogger.local_logger:
            config_local = LocalLogger.start_experiment(config, **kwargs)

        return config_wandb or config_local

    @staticmethod
    def finish_experiment():
        if ExperimentLogger.wandb_logger:
            WandbLogger.finish_experiment()
        if ExperimentLogger.local_logger:
            LocalLogger.finish_experiment()

    @staticmethod
    def log(log: dict):
        if ExperimentLogger.wandb_logger:
            WandbLogger.log(log)
        if ExperimentLogger.local_logger:
            LocalLogger.log(log)

    @staticmethod
    def log_artifact(data, name: str, art_type: str):
        if ExperimentLogger.wandb_logger:
            WandbLogger.log_artifact(data, name, art_type)
        if ExperimentLogger.local_logger:
            LocalLogger.log_artifact(data, name, art_type)

    @staticmethod
    def use_artifact(name: str, alias: str = 'latest'):
        data_wandb = None
        data_local = None

        if ExperimentLogger.wandb_logger:
            data_wandb = WandbLogger.use_artifact(name, alias)
        if ExperimentLogger.local_logger:
            data_local = LocalLogger.use_artifact(name, alias)

        return data_wandb or data_local
