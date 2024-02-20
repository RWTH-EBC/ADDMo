import os
from abc import ABC, abstractmethod
import pickle

import pandas as pd
import wandb

from core.model_tuning.models.abstract_model import AbstractMLModel


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
    def use_artifact(name: str, alias: str = "latest"):
        pass


class WandbLogger(AbstractLogger):
    active: bool = False
    project = None  # Name of the Weights & Biases project that you created in the browser wandb.ai
    directory = None  # Local directory where a backup of the uploaded files is stored

    @staticmethod
    def start_experiment(config=None, **kwargs):
        """Starts a new experiment and logs the config to wandb."""
        if WandbLogger.active:
            wandb.init(
                project=WandbLogger.project,
                config=config,
                dir=WandbLogger.directory,
                **kwargs,
            )
            wandb.define_metric("*", step_metric="global_step")
            return wandb.config

    @staticmethod
    def finish_experiment():
        """Finishes the current experiment."""
        if WandbLogger.active:
            wandb.finish()

    @staticmethod
    def log(log: dict):
        if WandbLogger.active:
            processed_log = {}
            for name, data in log.items():
                if isinstance(data, (pd.DataFrame, pd.Series)):
                    data = data.reset_index()
                    processed_log[name] = wandb.Table(dataframe=data)
                elif isinstance(data, list):
                    processed_log[name] = wandb.Histogram(data)
                elif isinstance(data, str):
                    processed_log[name] = wandb.Html(data)
                elif isinstance(data, dict):
                    # flatten the dictionary
                    for key, value in data.items():
                        processed_log[f"{name}.{key}"] = value
                else:
                    processed_log[name] = data
            wandb.log(processed_log)

    @staticmethod
    def log_artifact(
        data,
        name: str,
        art_type: str,
        description: str = None,
        metadata: dict = None,
    ):
        if WandbLogger.active:
            # save data to disk first
            filepath = None
            if art_type == "pkl":
                filepath = os.path.join(WandbLogger.directory, name + ".pkl")
                with open(filepath, "wb") as f:
                    pickle.dump(data, f)
            if art_type == "onnx":
                model: AbstractMLModel = data
                filepath = os.path.join(WandbLogger.directory, name + ".onnx")
                model.save_model(filepath)

            # create artifact object
            artifact = wandb.Artifact(
                name=name, type=art_type, description=description, metadata=metadata
            )

            # add saved file to the artifact, you may also add a whole directory to the artifact
            artifact.add_file(filepath)

            # actually log the artifact
            wandb.run.log_artifact(artifact)
            # artifact.wait()

    @staticmethod
    def use_artifact(name: str, alias: str = "latest"):
        if WandbLogger.active:
            artifact = wandb.use_artifact(f"{name}:{alias}")
            filename = artifact.download() + "\\" + name + ".pkl"
            return read_pkl(filename)


class LocalLogger(AbstractLogger):
    active: bool = False  # Activate local logging
    directory = None  # Directory to store artifacts locally
    run_time_storage = {}  # Storage for the current run

    @staticmethod
    def start_experiment(config, **kwargs):
        if LocalLogger.active:
            return config

    @staticmethod
    def finish_experiment():
        if LocalLogger.active:
            # safe run_time_storage to disk
            pass  # Implement finish experiment logic here

    @staticmethod
    def log(log: dict):
        if LocalLogger.active:
            # safe to run_time_storage
            pass  # Implement log logic here

    @staticmethod
    def log_artifact(data, name: str, art_type: str):
        if LocalLogger.active:
            if art_type == "data":
                file_path = os.path.join(LocalLogger.directory, name + ".csv")
                data.to_csv(file_path)

    @staticmethod
    def use_artifact(name: str, alias: str = "latest"):
        if LocalLogger.active:
            filename = name  # Assuming filename logic is handled appropriately
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
    def use_artifact(name: str, alias: str = "latest"):
        data_wandb = WandbLogger.use_artifact(name, alias)
        data_local = LocalLogger.use_artifact(name, alias)
        return data_wandb or data_local
