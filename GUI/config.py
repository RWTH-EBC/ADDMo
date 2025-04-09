from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union

class ConfigModel(BaseModel):
    name_of_raw_data: str = Field(..., description="Name of the raw data")
    name_of_tuning: str
    abs_path_to_data: str
    name_of_target: str
    create_differences: bool
    create_manual_target_lag: bool
    target_lag: List[int]
    create_automatic_timeseries_target_lag: bool
    minimum_target_lag: int
    create_manual_feature_lags: bool
    feature_lags: Dict[str, List[int]]
    create_automatic_feature_lags: bool
    minimum_feature_lag: int
    maximum_feature_lag: int
    manual_feature_selection: bool
    selected_features: List[str]
    filter_low_variance: bool
    low_variance_threshold: float
    filter_ICA: bool
    filter_univariate: bool
    univariate_score_function: str
    univariate_search_mode: str
    univariate_filter_params: int
    embedded_model: str
    filter_recursive_embedded: bool
    recursive_embedded_number_features_to_select: int
    wrapper_sequential_feature_selection: bool
    sequential_direction: str
    hyperparameter_tuning_type: str
    hyperparameter_tuning_kwargs: Dict[str, Union[int, float]]
    validation_score_mechanism: str
    validation_score_mechanism_kwargs: Optional[Dict[str, Union[int, float]]]
    validation_score_splitting: str
    validation_score_splitting_kwargs: Dict[str, Union[int, float]]
    validation_score_metric: str
    validation_score_metric_kwargs: Optional[Dict[str, Union[int, float]]]
    model: str
    min_increase_4_wrapper: float