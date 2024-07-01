from pydantic import BaseModel, Field

from addmo.s3_model_tuning.config.model_tuning_config import ModelTunerConfig

class DataTuningAutoSetup(BaseModel):
    # Global Variables
    name_of_raw_data: str = Field(
        "test_raw_data",
        description="Set name of the folder where the experiments shall be saved.",
    )
    name_of_tuning: str = Field(
        "data_tuning_experiment_auto", description="Set name of the experiments series."
    )
    abs_path_to_data: str = Field(
        r"D:\04_GitRepos\addmo-extra\raw_input_data\InputData.xlsx",
        description="Path to the file that has the system_data.",
    )
    name_of_target: str = Field(
        "FreshAir Temperature", description="Name of the target variable."
    )

    # FeatureConstruction Variables
    create_differences: bool = Field(
        False,
        description="Feature difference creation (building the derivative of the features).",
    )
    create_manual_target_lag: bool = Field(
        True, description="Manual construction of target lags."
    )
    target_lag: list = Field([1, 2], description="Array of lags for the target.")
    create_automatic_timeseries_target_lag: bool = Field(
        False, description="Automatic construction of time series target lags."
    )
    minimum_target_lag: int = Field(
        1, description="Minimal target lag which shall be considered."
    )
    create_manual_feature_lags: bool = Field(
        False, description="Manual construction of feature lags."
    )
    feature_lags: dict = Field(
        {"FreshAir Temperature": [1, 2], "Total active power": [1, 2]},
        description="Feature_lags in format {var_name: [lags]}",
    )
    create_automatic_feature_lags: bool = Field(
        False, description="Automatic construction of feature lags via wrapper."
    )
    minimum_feature_lag: int = Field(1, description="Minimum feature lag.")
    maximum_feature_lag: int = Field(20, description="Maximum feature lag.")

    # FeatureSelection Variables
    manual_feature_selection: bool = Field(
        False, description="Manual selection of Features by their Column number."
    )
    selected_features: list = Field(
        ["FreshAir Temperature", "Total active power"],
        description="Variable names of the features to keep.",
    )
    filter_low_variance: bool = Field(
        True, description="Remove features with low variance."
    )
    low_variance_threshold: float = Field(
        0.1, description="Variance threshold for feature removal."
    )
    filter_ICA: bool = Field(
        False, description="Filter: Independent Component Analysis(ICA)."
    )
    filter_univariate: bool = Field(
        False, description="Filter univariate by scikit-learn."
    )
    univariate_score_function: str = Field(
        "mutual_info_regression",
        description="'mutual_info_regression' or 'f_regression'.",
    )
    univariate_search_mode: str = Field(
        "percentile", description="'percentile' or 'k_best'."
    )
    univariate_filter_params: int = Field(
        50, description="Percent of features to keep or number of top features to keep."
    )
    embedded_model: str = Field(
        "RF", description="Estimator for use in all embedded methods."
    )
    filter_recursive_embedded: bool = Field(
        False, description="Enable recursive feature elimination."
    )
    recursive_embedded_number_features_to_select: int = Field(
        18, description="Number of features to select in recursive feature elimination."
    )
    wrapper_sequential_feature_selection: bool = Field(
        False, description="Enable wrapper sequential feature selection."
    )
    sequential_direction: str = Field(
        "forward",
        description="'forward' or 'backward' direction for sequential feature selection.",
    )

    # Wrapper Model Variables
    config_model_tuning: ModelTunerConfig = Field(
        ModelTunerConfig(), description="Model tuning setup, set your own config."
    )

    min_increase_4_wrapper: float = Field(
        0.005,
        description="Minimum score increase for a feature to be considered worthy in wrapper methods.",
    )
