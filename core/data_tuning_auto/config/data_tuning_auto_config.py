from core.util.abstract_config import BaseConfig

"""Serves as blueprint for the yaml-configs."""
class DataTuningAutoSetup(BaseConfig):
    """
    Object that stores all the setup and user input information of Data Tuning"""

    def __init__(self):
        # -----------------------Global Variables-------------------------------

        self.name_of_raw_data: str = "test_raw_data"  # Set name of the folder where the experiments
        # shall be saved, e.g. the name of the observed data
        self.name_of_tuning: str = "data_tuning_experiment_1"  # Set name of the experiments series

        self.abs_path_to_data: str = r"D:\04_GitRepos\addmo-extra\raw_input_data\InputData.xlsx"  # Path to the file that has
        # the data
        self.name_of_target:str = "FreshAir Temperature"
        # Name of the target variable

        # -----------------------ImportData Variables-------------------------------

        # -----------------------Preprocessing Variables-------------------------------

        # -----------------------PeriodSelection Variables-------------------------------

        # -----------------------FeatureConstruction Variables-------------------------------

        # self.correlation_plotting = False  # Production of Cross and Autocorrelation Plots in order to find meaningful OwnLags and FeatureLags
        # self.lags_4_plotting = 200  # Number of lags which will be plotted in x-axis

        self.create_differences: bool = False  # Feature difference creation (building the
        # derivative of the features)

        # manual construction of target lags
        self.create_manual_target_lag:bool = False
        self.target_lag: list = [1, 2] # Type in an array of lags for the target

        # automatic construction of time series target lags
        self.create_automatic_timeseries_target_lag: bool = True
        self.minimum_target_lag: int = 1  # minimal target lag which shall be considered

        # manual construction of feature lags
        self.create_manual_feature_lags:bool = False
        self.feature_lags: dict = {"FreshAir Temperature": [1, 2],
                             "Total active power": [1, 2]}
        # feature_lags in format {var_name: [first lag (int), second lag (int)]}

        # automatic construction of feature lags via wrapper, adds the best lag per feature
        self.create_automatic_feature_lags = False
        self.minimum_feature_lag: int = 1
        self.maximum_feature_lag: int = 20

        # -----------------------FeatureSelection Variables-------------------------------

        self.manual_feature_selection:bool= False  # Manual selection of Features by their Column
        # number (After Feature Construction)
        self.selected_features: list = ["FreshAir Temperature", "Total active power"]
        # variable names of the features to keep (incl. created features)

        self.filter_low_variance:bool = True  # Remove features with low variance
        self.low_variance_threshold = 0.1  # Removes all features with a lower variance than the stated threshold; variance is calculated with scaled data (if a scaler was used, regularly only features that are always the same have a small variance)

        self.filter_ICA:bool = False  # Filter: Independent Component Analysis(ICA)

        self.filter_univariate:bool = False  # Filter univariate by scikit-learn
        self.univariate_score_function:str = "mutual_info_regression"
        # 'mutual_info_regression' or 'f_regression'
        self.univariate_search_mode:str = "percentile"
        self.univariate_filter_params: int = 50  # If percentile: percent of features to keep; if
        # Kbest: number of top features to keep

        self.embedded_model: str = "RF"  # Define and set Estimator which shall be used in all
        # embedded
        # Methods(only present in feature selection)

        self.filter_recursive_embedded: bool = False
        self.recursive_embedded_number_features_to_select:int = 18
        # Enter number of features to select. Enter 0 for automatic choice


        self.wrapper_sequential_feature_selection:bool = False  # Wrapper recursive feature
        self.sequential_direction = "forward"  # 'forward' or 'backward'

        # -----------------------Wrapper Model Variables-------------------------------

        self.hyperparameter_tuning_type: str = "OptunaTuner"  # e.g. OptunaTuner, GridSearchTuner, or NoTuningTuner
        self.hyperparameter_tuning_kwargs: dict = {"n_trials": 5}  # kwargs for the tuner

        self.validation_score_mechanism: str = "cv"  # e.g. cross validation, holdout, etc.
        self.validation_score_mechanism_kwargs: dict = None  # kwargs for the mechanism

        self.validation_score_splitting: str = "KFold"  # all custom splitters or scikit-learn
        # splitters, e.g. kfold, timeseriessplit, etc.
        self.validation_score_splitting_kwargs: dict = {"n_splits":3}  # kwargs for the splitter

        self.validation_score_metric: str = "r2"  # all custom metrics or scikit-learn metrics,
        # e.g. r2, neg_mean_absolute_error, d2_pinball_score, etc.
        self.validation_score_metric_kwargs: dict = None  # kwargs for the metric

        self.model: str = "MLP"  # array of the models you want to use

        self.min_increase_4_wrapper: float = 0.005  # Minimum difference between two scores to
        # accept a feature construction or selection as worthy. Only used in wrapper methods.


