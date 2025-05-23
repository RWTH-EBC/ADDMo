import os
from pydantic import BaseModel, Field, PrivateAttr
from addmo.util.load_save_utils import root_dir

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
        os.path.join(root_dir(),'addmo_examples','raw_input_data','InputData.xlsx'),
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
    target_lag: list[int] = Field([1, 2], description="Array of lags for the target.")

    minimum_target_lag: int = Field(
        1, description="Minimal target lag which shall be considered."
    )
    create_manual_feature_lags: bool = Field(
        False, description="Manual construction of feature lags."
    )
    feature_lags: dict[str, list[int]] = Field(
        {"FreshAir Temperature": [1, 2], "Total active power": [1, 2]},
        description="Feature_lags in format {var_name: [lags]}",
    )
    # FeatureSelection Variables
    manual_feature_selection: bool = Field(
        False, description="Manual selection of Features by their Column name."
    )
    selected_features: list[str] = Field(
        ["FreshAir Temperature", "Total active power"],
        description="Variable names of the features to keep.",
    )

    sequential_direction: str = Field(
        "forward",
        description="'forward' or 'backward' direction for sequential feature selection.",
    )

    filter_recursive_by_count: bool = Field(
        False,
        description="Enable recursive feature elimination."
    )
    recursive_embedded_number_features_to_select: int = Field(
        18,
        description="Number of features to select in recursive feature elimination."
    )
    filter_recursive_by_score: bool = Field(
        False,
        description="Enable wrapper sequential feature selection."
    )
    min_increase_for_wrapper: float = Field(
        0.01,
        description="Minimum performance gain for wrapper-based feature selection."
    )



