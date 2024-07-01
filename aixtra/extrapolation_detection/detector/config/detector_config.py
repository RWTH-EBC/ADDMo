from pydantic import BaseModel, Field


class DetectorConfig(BaseModel):
    detectors: list[str] = Field(["KNN"], description="List of detectors to use")
    detector_experiment_name: str = Field(
        "detector_experiment1", description="Name of the detector experiment"
    )

    # for splitting train / test / validation
    use_test_for_validation: bool = Field(
        True, description="Appends test system_data to validation system_data if True"
    )
    use_train_for_validation: bool = Field(
        True, description="Appends train system_data to validation " "system_data if True"
    )
    use_remaining_for_validation: bool = Field(
        False, description="Appends remaining system_data to validation system_data if True"
    )

    # for tuning
    tuning_bool: bool = Field(True, description="If True, the detector will be tuned")

    # for scoring
    beta_f_score: float = Field(
        1,
        description="Beta value for F-score. "
                    "Recall is beta-times more important than precision."
                    "Recall is more important if beta > 1, precision if beta < 1.",
    )
