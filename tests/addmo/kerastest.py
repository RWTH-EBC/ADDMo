import pandas as pd
import os
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from addmo.s3_model_tuning.models.keras_models import SciKerasSequential
from addmo.s3_model_tuning.models.model_factory import ModelFactory
from addmo.s3_model_tuning.models.scikit_learn_models import ScikitMLP
from addmo.util.load_save_utils import root_dir
from addmo.s3_model_tuning.config.model_tuning_config import ModelTunerConfig
from addmo.s3_model_tuning.model_tuner import ModelTuner
from aixtra_use_case.config.ed_experiment_config import ExtrapolationExperimentConfig

# from sweeps import config_blueprints
# LocalLogger.directory = r'C:\Users\mre-rpa\PycharmProjects\addmo\addmo-automated-ml-regression\0000_testfiles'
# LocalLogger.active = True
# WandbLogger.active = False
# WandbLogger.project = 'keras'
# WandbLogger.directory = r'C:\Users\mre-rpa\PycharmProjects\addmo\addmo-automated-ml-regression\0000_testfiles'
# config = ExtrapolationExperimentConfig()
# config.simulation_data_name = "testkeras"
# config.experiment_name = "test"
#
# ExperimentLogger.start_experiment(config=config)
    # Load and prepare system_data
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['price'] = pd.Series(data.target)
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df['price'], test_size=0.2, random_state=42)
dir_path = os.path.join(root_dir(), "tests")
#
model2 = ScikitMLP()
model2.fit(X_train, y_train)
model2.save_regressor(dir_path, "regressor1")
# ExperimentLogger.log_artifact(model2, "reg", "keras")
model1= ModelFactory.load_model(r"C:\Users\mre-rpa\PycharmProjects\pythonProject2\addmo-automated-ml-regression\tests\regressor1.joblib")
# model1 = ExperimentLogger.use_artifact("reg")
# print(model1)
pred_loaded= model1.predict(X_test)
y_pred = model2.predict(X_test)
r_loaded= r2_score(y_test, pred_loaded)
r2= r2_score(y_test, y_pred )
print( "trained model r2 score: ", r2, "loaded: ", r_loaded)

# # print(y_pred)
# # print(keras.__version__)
# # print(tensorflow.__version__)
# # print(scikeras.__version__)
# def setup_model_tuning_config():
#     # Configures and returns a ModelTunerConfig instance
#     return ModelTunerConfig(
#         models=["SciKerasSequential"],
#         hyperparameter_tuning_type="OptunaTuner",
#         hyperparameter_tuning_kwargs={"n_trials": 3},
#         validation_score_mechanism="cv",
#         validation_score_splitting="KFold",
#         validation_score_metric="neg_mean_squared_error"
#     )
# config = ExtrapolationExperimentConfig()
#
# config = setup_model_tuning_config()
# tuner = ModelTuner(config)
#
# model_dict = tuner.tune_all_models(X_train, y_train)
# print(model_dict)
# best_model_name = tuner.get_best_model_name(model_dict)
# best_model = tuner.get_model(model_dict, "SciKerasSequential")
# print("best model name is: ", best_model_name)
# print("best model is: ", best_model)
# regressor=best_model
# p= regressor.get_params()
# print(p)
# regressor.regressor.model.summary()
# WandbLogger.active = True
# WandbLogger.project = 'ED_Boptest_TAir_mid_ODE_noise_m0_std0.01'
# WandbLogger.directory = r'C:\Users\mre-rpa\PycharmProjects\addmo\addmo-automated-ml-regression\0000_testfiles'
# config = { }
# ExperimentLogger.start_experiment(config=config)
# regressor: AbstractMLModel = ExperimentLogger.use_artifact("regressor")
# print(regressor.regressor.model.summary())
# model_config = config_blueprints.no_tuning_config(config)
# print(model_config)
# config=model_config
# tuner = ModelTuner(config)
#
# model_dict = tuner.tune_all_models(X_train, y_train)
# print(model_dict)