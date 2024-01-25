import git
import wandb

from core.util.pickle_handling import write_pkl, _get_path, read_pkl


class ExperimentLogger:
    ##################################################################################################################
    wandb_active = False  # Activate weights & biases
    directory = git.Repo('.', search_parent_directories=True).working_tree_dir + '/storedData'  # Local directory
    project = 'remeasure-test'  # Weights & Biases project

    ##################################################################################################################

    @staticmethod
    def start_experiment(config=None, **kwargs) -> dict:
        if ExperimentLogger.wandb_active:
            wandb.init(project=ExperimentLogger.project, config=config, **kwargs,
                       dir=git.Repo('.', search_parent_directories=True).working_tree_dir + '/wanddb')
            config = wandb.config
        return config

    @staticmethod
    def finish_experiment():
        if ExperimentLogger.wandb_active:
            wandb.finish()

    @staticmethod
    def log(log: dict):
        if ExperimentLogger.wandb_active:
            wandb.log(log)

    @staticmethod
    def log_artifact(data, name: str, art_type: str):
        write_pkl(data, name, ExperimentLogger.directory, override=True)
        if ExperimentLogger.wandb_active:
            artifact = wandb.Artifact(name=name, type=art_type)
            artifact.add_file(_get_path(name, ExperimentLogger.directory))
            wandb.run.log_artifact(artifact)
            artifact.wait()

    @staticmethod
    def use_artifact(name: str, alias: str = 'latest'):
        if ExperimentLogger.wandb_active:
            artifact = wandb.use_artifact(name + ':' + alias)
            filename = artifact.download() + '\\' + name + '.pkl'
            directory = None
        else:
            filename = name
            directory = ExperimentLogger.directory
        data = read_pkl(filename, directory)
        return data
