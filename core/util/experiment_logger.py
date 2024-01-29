import wandb
import os
from abc import ABC, abstractmethod


from core.util.pickle_handling import write_pkl, read_pkl


class AbstractLogger(ABC):
    def __init__(self, project):
        self.project = project
        self.wandb_logger = None # to trigger them separately in some cases
        self.local_logger = None # to trigger them separately in some cases

    @abstractmethod
    def start_experiment(self, project: str, config:dict = None, **kwargs):
        '''
        Returns a config dict.
        Despite defined as class in decision is taken, as class objects can be refactored easily.
        '''
        pass

    @abstractmethod
    def finish_experiment(self):
        pass

    @abstractmethod
    def log(self, log: dict):
        pass

    @abstractmethod
    def log_artifact(self, data, name: str, art_type: str):
        pass

    @abstractmethod
    def use_artifact(self, name: str, alias: str = 'latest'):
        pass




class WandbLogger(AbstractLogger):
    def __init__(self, project):
        self.project = project

    def start_experiment(self, config=None, **kwargs):
        wandb.init(project=self.project, config=config, **kwargs)
        return wandb.config

    def finish_experiment(self):
        wandb.finish()

    def log(self, log: dict):
        wandb.log(log)

    def log_artifact(self, data, name: str, art_type: str):
        artifact = wandb.Artifact(name=name, type=art_type)
        artifact.add_file(_get_path(name))
        wandb.run.log_artifact(artifact)
        artifact.wait()

    def use_artifact(self, name: str, alias: str = 'latest'):
        artifact = wandb.use_artifact(f'{name}:{alias}')
        filename = artifact.download() + '\\' + name + '.pkl'
        return read_pkl(filename)

class LocalLogger(AbstractLogger):
    def __init__(self, directory):
        self.directory = directory

    def start_experiment(self, config, **kwargs):
        # Local logging may not require specific actions at start
        return config

    def finish_experiment(self):
        # Local logging may not require specific actions at finish
        pass

    def log(self, log: dict):
        # Implement local logging logic (e.g., saving to a file)
        pass

    def log_artifact(self, data, name: str, art_type: str):
        # save the data to .csv
        if art_type == 'data':
            file_path = os.path.join(self.directory, name + '.csv')
            data.to_csv(file_path)

    def use_artifact(self, name: str, alias: str = 'latest'):
        filename = name
        return read_pkl(filename, self.directory)


class ExperimentLogger(AbstractLogger):
    def __init__(self, wandb_logger:WandbLogger=None, local_logger:LocalLogger=None):
        self.wandb_logger = wandb_logger
        self.local_logger = local_logger

    def start_experiment(self, project: str, config=None, **kwargs):
        config_wandb = None
        config_local = None

        if self.wandb_logger:
            config_wandb = self.wandb_logger.start_experiment(project, config, **kwargs)
        if self.local_logger:
            config_local = self.local_logger.start_experiment(project, config, **kwargs)

        return config_wandb or config_local

    def finish_experiment(self):
        if self.wandb_logger:
            self.wandb_logger.finish_experiment()
        if self.local_logger:
            self.local_logger.finish_experiment()

    def log(self, log: dict):
        if self.wandb_logger:
            self.wandb_logger.log(log)
        if self.local_logger:
            self.local_logger.log(log)

    def log_artifact(self, data, name: str, art_type: str):
        if self.wandb_logger:
            self.wandb_logger.log_artifact(data, name, art_type)
        if self.local_logger:
            self.local_logger.log_artifact(data, name, art_type)

    def use_artifact(self, name: str, alias: str = 'latest'):
        data_wandb = None
        data_local = None

        if self.wandb_logger:
            data_wandb = self.wandb_logger.use_artifact(name, alias)
        if self.local_logger:
            data_local = self.local_logger.use_artifact(name, alias)

        return data_wandb or data_local