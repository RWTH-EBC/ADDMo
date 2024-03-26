from pydantic import BaseModel, Field


class ModelTunerConfig(BaseModel):
    models: list[str] = Field(["MLP"], description="List of models to use")

    hyperparameter_tuning_type: str = Field(
        "OptunaTuner",
        description="Type of hyperparameter tuning, e.g., OptunaTuner, GridSearchTuner",
    )
    hyperparameter_tuning_kwargs: dict = Field(
        {"n_trials": 2}, description="Kwargs for the tuner"
    )

    validation_score_mechanism: str = Field(
        "cv", description="Validation score mechanism, e.g., cross validation, holdout"
    )
    validation_score_mechanism_kwargs: dict = Field(
        default=None, description="Kwargs for the validation score mechanism"
    )

    validation_score_splitting: str = Field(
        "KFold", description="Validation score splitting, e.g., KFold, PredefinedSplit"
    )
    validation_score_splitting_kwargs: dict = Field(
        default=None, description="Kwargs for the validation score splitter"
    )

    validation_score_metric: str = Field(
        "r2", description="Validation score metric, e.g., r2, neg_mean_absolute_error"
    )
    validation_score_metric_kwargs: dict = Field(
        default=None, description="Kwargs for the validation score metric"
    )



class ModelTuningExperimentConfig(BaseModel):
    name_of_raw_data: str = Field(
        "test_data", description="Refer to the raw data connected to this"
    )
    name_of_data_tuning_experiment: str = Field(
        "test_data_tuning",
        description="Refer to the data tuning experiment aka the input data for this model tuning experiment",
    )
    name_of_model_tuning_experiment: str = Field(
        "test_model_tuning", description="Set name of the current experiment"
    )
    abs_path_to_data: str = Field(
        r"D:\\04_GitRepos\\addmo-extra\\raw_input_data\\InputData.xlsx",
        description="Path to the file that has the data",
    )
    name_of_target: str = Field(
        "FreshAir Temperature", description="Name of the target variable"
    )

    # Model Tuning Variables
    start_train_val: str = Field(
        "2016-08-01 00:00",
        description="Start date and time for training and validation",
    )
    stop_train_val: str = Field(
        "2016-08-14 23:45", description="Stop date and time for training and validation"
    )
    start_test: str = Field(
        "2016-08-15 00:00", description="Start date and time for testing"
    )
    end_test: str = Field(
        "2016-08-16 23:45", description="End date and time for testing"
    )
    config_model_tuner: ModelTunerConfig = Field(ModelTunerConfig(), description="Model tuner config, set your own config.")
