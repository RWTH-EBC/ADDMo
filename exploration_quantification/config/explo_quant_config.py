from core.util.abstract_config import BaseConfig
class ExploQuantConfig(BaseConfig):

    def __init__(self):
        self.detectors: list[str] = ["ConvexHull", "KNN"]

        self.bounds: dict or str = "infer" #{"var1": (0, 1), "var2": (0, 1)} # bounds for each


        # variable or infer to infer from the data
        self.explo_grid_points_per_axis = 100


