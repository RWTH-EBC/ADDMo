import os
from abc import ABC, abstractmethod
import pickle
import pandas as pd
from pandas import ExcelWriter
import wandb
from addmo.s3_model_tuning.models.abstract_model import AbstractMLModel
from addmo.util.load_save import save_config_to_json
from addmo.util.load_save_utils import create_or_clean_directory
from addmo.util.load_save_utils import create_path_or_ask_to_override
from addmo.s3_model_tuning.models.model_factory import ModelFactory


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

    @staticmethod
    def _handle_pkl(data,name,art_type, directory):
        filename = f"{name}.{art_type}"
        create_path_or_ask_to_override(filename, directory)
        filepath = os.path.join(directory, filename)
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        return "pkl", [filepath]

    @staticmethod
    def _handle_model(data,name,art_type, directory):
        model: AbstractMLModel = data
        filepath = os.path.join(directory, f"{name}.{art_type}")
        metadata_filepath = os.path.join(directory, f"{name}_metadata.json")
        model.save_regressor(directory, name, art_type)
        return "regressor", [filepath, metadata_filepath]

class WandbLogger(AbstractLogger):
    active: bool = False
    project = None  # Name of the Weights & Biases project that you created in the browser wandb.ai
    directory = None  # Local directory where a backup of the uploaded files is stored

    @staticmethod
    def start_experiment(config=None, **kwargs):
        """
        Starts a new experiment and logs the config to wandb.
        """
        if WandbLogger.active:
            if not os.path.exists(WandbLogger.directory):
                os.makedirs(WandbLogger.directory)
            wandb.init(
                project=WandbLogger.project,
                config=config,
                dir=WandbLogger.directory,
                **kwargs,
            )

            return wandb.config

    @staticmethod
    def finish_experiment():
        """
        Finishes the current experiment.
        """
        if WandbLogger.active:
            wandb.finish()

    @staticmethod
    def log(log: dict):
        """
        Logs run data.
        """
        if WandbLogger.active:
            processed_log = {}
            for name, data in log.items():
                if isinstance(data, (pd.DataFrame, pd.Series)):
                    data = data.reset_index()
                    processed_log[name] = wandb.Table(dataframe=data)
                elif isinstance(data, list):
                    processed_log[name] = str(data)
                elif isinstance(data, tuple):
                    processed_log[name] = str(data)
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
            metadata: dict = None
    ):
        """
        Logs artifact data.
        """
        if not WandbLogger.active:
            return

        type_handlers = {
            "pkl": lambda d: super(WandbLogger, WandbLogger)._handle_pkl(d, name, art_type, WandbLogger.directory),
            "h5": lambda d: super(WandbLogger, WandbLogger)._handle_model(d, name, art_type, WandbLogger.directory),
            "keras": lambda d: super(WandbLogger, WandbLogger)._handle_model(d, name, art_type, WandbLogger.directory),
            "joblib": lambda d: super(WandbLogger, WandbLogger)._handle_model(d, name, art_type, WandbLogger.directory),
            "onnx": lambda d: super(WandbLogger, WandbLogger)._handle_model(d, name, art_type, WandbLogger.directory),
        }

        if art_type not in type_handlers:
            raise ValueError(f"Unsupported artifact type: {art_type}")

        artifact_type, files_to_add = type_handlers[art_type](data)

        artifact = wandb.Artifact(
            name=name, type=artifact_type, description=description, metadata=metadata
        )

        for file in files_to_add:
            artifact.add_file(file)

        wandb.run.log_artifact(artifact)
        artifact.wait()

    @staticmethod
    def use_artifact(name: str, alias: str = "latest"):
        """
        Downloads logged model artifact from wandb.
        """
        if WandbLogger.active:
            artifact = wandb.use_artifact(f"{name}:{alias}")
            artifact_dir = artifact.download()

            # Find the model and metadata files
            for file in os.listdir(artifact_dir):
                if file.endswith(('.joblib', '.onnx', '.h5', '.keras', '.pkl')):
                    model_file = file

            model_path = os.path.join(artifact_dir, model_file)

            if model_file.endswith('.pkl'):
                with open(model_path, "rb") as f:
                    loaded_model = pickle.load(f)
            else:
                loaded_model = ModelFactory().load_model(model_path)

            return loaded_model


class LocalLogger(AbstractLogger): #Todo: evtl. komplett löschen und auf normale speicher funktionen umstellen?
    active: bool = False  # Activate local logging
    directory = None  # Directory to store artifacts locally
    run_time_storage = {}  # Storage for the current run

    @staticmethod
    def start_experiment(config, **kwargs):
        """
        Starts a new experiment and logs the config to LocalLogger.
        """
        if LocalLogger.active:
            create_or_clean_directory(LocalLogger.directory)
            path = os.path.join(LocalLogger.directory, "config.json")
            save_config_to_json(config, path)
            return config

    @staticmethod
    def finish_experiment():
        """
        Finishes the current experiment.
        """
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
        """
        Logs artifact data.
        """
        if LocalLogger.active:
            if art_type == "system_data":
                file_path = os.path.join(LocalLogger.directory, name)
                data.to_csv(os.path.join(file_path + ".csv"))

                return

            # type_handlers = {
            #     "pkl": lambda d: super(LocalLogger, LocalLogger)._handle_pkl(d, name, art_type, LocalLogger.directory),
            #     "h5": lambda d: super(LocalLogger, LocalLogger)._handle_model(d, name, art_type, LocalLogger.directory),
            #     "keras": lambda d: super(LocalLogger, LocalLogger)._handle_model(d, name, art_type, LocalLogger.directory),
            #     "joblib": lambda d: super(LocalLogger, LocalLogger)._handle_model(d, name, art_type, LocalLogger.directory),
            #     "onnx": lambda d: super(LocalLogger, LocalLogger)._handle_model(d, name, art_type, LocalLogger.directory),
            # }
            #
            # if art_type not in type_handlers:
            #     raise ValueError(f"Unsupported artifact type: {art_type}")
            #
            # artifact_type, files_to_add = type_handlers[art_type](data)
            # print(f"Saved {artifact_type} files: {files_to_add}")


    @staticmethod
    def use_artifact(name: str, alias: str = "latest"):
        """
        Downloads logged model artifact.
        """
        if LocalLogger.active:
            filename = name + '.csv'
            file_path = os.path.join(LocalLogger.directory, filename)
            if os.path.exists(file_path):  # Check if the file exists
                return pd.read_csv(file_path)
            else:
                # If the file does not exist, return None silently
                return None


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
        return data_wandb
