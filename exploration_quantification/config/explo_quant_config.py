from pydantic import BaseModel, Field
from typing import Union

from extrapolation_detection.detector.config.detector_config import DetectorConfig
class ExploQuantConfig(BaseModel):
    detectors: str = Field(
        "ConvexHull", description="Detector to be used for exploration " "detection."
    )
    exploration_bounds: Union[dict, str] = Field(
        "infer",
        description="Bounds for each variable as dict {name_of_var: (min, max)};"
        "or 'infer' to infer the bounds from the min and max of the data.",
    )
    explo_grid_points_per_axis: int = Field(
        100, description="Number of grid points per axis for exploration"
    )

    test: DetectorConfig = DetectorConfig()