import numpy as np
from sklearn.externals import joblib
import os
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PostProcessing:
    DEFAULT_PREDICTOR_MODEL_FILE_NAMES = {
        'ANN': 'ann_bayesian_predictor.save',
        'GB': 'gradientboost_bayesian.save',
        'LASSO': 'lasso_bayesian.save',
        'RF': 'rf_predictor.save',
        'SVR': 'svr_bayesian_predictor.save',
    }

    def __init__(
            self,
            dir_base: str = 'TrialInput',
            dir_trial_tuned_data: str = 'TrialTunedData',
            dir_model: str = 'TrialTunedModel'
    ):
        logger.info(f"Initializing PostProcessing ...")
        # Dir paths
        self.dir_base = dir_base
        self.dir_trial_tuned_data = os.path.join(self.dir_base, dir_trial_tuned_data)
        self.dir_model = os.path.join(self.dir_trial_tuned_data, 'Predictions', dir_model)

        # Load all obj files
        self.feature_scaler = self.load_model(os.path.join(self.dir_trial_tuned_data, 'FeatureScaler.save'))
        self.scaler_tracker = self.load_model(os.path.join(self.dir_trial_tuned_data, 'ScalerTracker.save'))
        self.name_of_signal = self.load_model(os.path.join(self.dir_trial_tuned_data, 'NameOfSignal.save'))
        self.best_models = {
            k: self.load_model(os.path.join(self.dir_model, 'BestModels', v))  # Load ML models
            for k, v in self.DEFAULT_PREDICTOR_MODEL_FILE_NAMES.items()
        }

    def model_single_predict(self, model_name: str, features_data: dict):
        if model_name not in self.DEFAULT_PREDICTOR_MODEL_FILE_NAMES.keys():
            raise ValueError(f"No predictor model named '{model_name}',it must be one of "
                             f"{self.DEFAULT_PREDICTOR_MODEL_FILE_NAMES.keys()}")
        else:
            model = self.best_models[model_name]

        if model is None:
            logging.error(f"No predictor model available for '{model_name}'")
            return None

        if not hasattr(model, "predict"):
            logging.error(f"No predict attribute available for '{model_name}'")
            return None

        # Convert features dict into df
        features_data[self.name_of_signal] = np.nan  # Add signal key-value to features+
        df_features = pd.DataFrame([features_data])
        input_data_scaled = self.feature_scaler.transform(df_features)[:, :-1]

        # Make prediction
        try:
            predictions = model.predict(input_data_scaled)
            predictions_scaled = self.scaler_tracker.inverse_transform(predictions.reshape(-1, 1))
            return predictions_scaled[0][0]
        except Exception as e:
            logging.error(f"An error occured while predicting {model_name}: {e}")

    @staticmethod
    def load_model(model_file_path: str):
        """Load a machine learning model (.save file) from the specified file path."""
        if not os.path.isfile(model_file_path):
            logger.error(f"No model file found at {model_file_path}")
            return None

        try:
            # Load the model from the given path
            model = joblib.load(model_file_path)
            logger.info(f"Model loaded successfully from '{model_file_path}'")
            return model

        except Exception as e:
            logger.error(f"An error occurred while loading the model: {e}")
            return None


if __name__ == '__main__':
    WORKSPACE = r"D:\sciebo\JJI_Privat\Diss\040_DynOptimazation\ADDMoResultData"
    test = PostProcessing(
        dir_base=os.path.join(WORKSPACE, 'TrialInput_0.7_Dyn_u.dt'),
        dir_trial_tuned_data='TrialTunedData_0.7_Dyn_u.dt',
        dir_model='TrialTunedModel_0.7_Dyn_u.dt'
    )
    res1 = test.model_single_predict(
        model_name='RF',
        features_data={
            'u_a [m/s]': 3, 'dt_ia [K]': 18, 'h_win [m]': 1.335, 'w_win [m]': 1.065, 'Aprj_win [m2]': 0.0673
        }
    )
    print(res1)
