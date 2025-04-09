import os
from pydantic import BaseModel, Field
from addmo.util.load_save_utils import root_dir
from pydantic import conlist

class DataTuningFixedConfig(BaseModel):
    abs_path_to_data: str = Field(
        os.path.join(root_dir(),'addmo_examples','raw_input_data','InputData.xlsx'),
        description="Absolute path to raw system_data",
    )
    name_of_raw_data: str = Field(
        "test_raw_data", description="Name of the raw system_data set"
    )
    name_of_tuning: str = Field(
        "data_tuning_experiment_fixed", description="Name of the system_data tuning configuration"
    )
    name_of_target: str = Field("FreshAir Temperature", description="Output of prediction")
    features: list[str] = Field(
        [
            "Schedule",
            "Total active power",
            "FreshAir Temperature___diff",
            "FreshAir Temperature___lag1",
            "FreshAir Temperature___squared",
            "Total active power___lag1",
        ],
        description="List of features which the tuning shall result in.",

    )
