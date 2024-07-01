from pydantic import BaseModel, Field


class DataTuningFixedConfig(BaseModel):
    path_to_raw_data: str = Field(
        r"D:\04_GitRepos\addmo-extra\raw_input_data\InputData.xlsx",
        description="Absolute path to raw system_data",
    )
    name_of_raw_data: str = Field(
        "test_raw_data", description="Name of the raw system_data set"
    )
    name_of_data_tuning: str = Field(
        "test_data_tuning", description="Name of the system_data tuning configuration"
    )
    target: str = Field("Total active power", description="Output of prediction")
    features: list[str] = Field(
        [
            "Schedule",
            "FreshAir Temperature",
            "FreshAir Temperature___diff",
            "FreshAir Temperature___lag1",
            "FreshAir Temperature___squared",
            "Total active power___lag1",
        ],
        description="List of features which the tuning shall result in.",
    )
