from pydantic import BaseModel, Field


class DetectorConfig(BaseModel):
    detectors: list[str] = Field(["KNN"], description="List of detectors to use")
    detector_experiment_name: str = Field(
        "detector_experiment1", description="Name of the detector experiment"
    )

    # for splitting train / test / validation
    use_test_for_validation: bool = Field(
        True, description="Appends test data to validation data if True"
    )
    use_train_for_validation: bool = Field(
        True, description="Appends train data to validation " "data if True"
    )
    use_remaining_for_validation: bool = Field(
        False, description="Appends remaining data to validation data if True"
    )

    # for tuning
    tuning_bool: bool = Field(True, description="If True, the detector will be tuned")

    # for scoring
    beta_f_score: float = Field(1, description="Beta value for F-score")
