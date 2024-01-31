from core.util.abstract_config import BaseConfig

"""Serves as blueprint for the yaml-configs."""
class DataTuningAutoSetup(BaseConfig):
    """
    Object that stores all the setup and user input information of Data Tuning"""

    def __init__(self, **kwargs):
        # -----------------------Global Variables-------------------------------

        self.name_of_raw_data: str = "AHU Data1"  # Set name of the folder where the experiments
        # shall be saved, e.g. the name of the observed data
        self.name_of_data_tuning_experiment: str = "NoOL"  # Set name of the experiments series

        self.abs_path_to_data: str = r"/raw_input_data/InputData.xlsx"  # Path to the file that has
        # the data
        self.name_of_target:str = "Empty"  # Name of the target variable

        self.abs_path_to_result_folder: str = r"Empty"  # Path to the folder where the results will
        # be stored

        # -----------------------Wrapper Variables-------------------------------

        self.wrapper_model: str = "MLP"  # Set the model that shall be used for the wrapper
        self.wrapper_params: str = "default"  # State the parameters that the model should have.
        # Eg. "default" or a dictionary with parameters readable for the model
        self.scoring_split_technique: str = "TimeSeriesSplit"  # Set the split technique that shall
        # be used
        self.scoring_metric: str = "R2"  # Scoring method for the wrapper
        self.min_increase_4_wrapper: float = 0.005  # Minimum difference between two scores to
        # accept a feature constuction or selection as worthy. Only used in wrapper methods.

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
        self.create_automatic_timeseries_target_lag: bool = False
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

        self.filter_low_variance:bool = False  # Remove features with low variance
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
        self.recursive_embedded_scoring = "R2" # choose a scoring function hyperparameter tuning
        self.recursive_embedded_threshold = False
        self.recursive_embedded_threshold_type = "median"



        self.wrapper_recursive_feature_selection = False  # Wrapper recursive feature



