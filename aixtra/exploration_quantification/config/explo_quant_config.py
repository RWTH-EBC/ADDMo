from pydantic import BaseModel, Field
from typing import Union


class ExploQuantConfig(BaseModel):
    detectors: list[str] = Field(
        ["ConvexHull"], description="Detector to be used for exploration " "detection."
    )
    exploration_bounds: Union[dict, str] = Field(
        "infer",
        description="Bounds for each variable as dict {name_of_var: (min, max)};"
        "or 'infer' to infer the bounds from the min and max of the system_data.",
    )
    explo_grid_points_per_axis: int = Field(
        20, description="Number of grid points per axis for exploration"
    )