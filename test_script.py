import os
from addmo.s3_model_tuning.config.model_tuning_config import ModelTunerConfig
from addmo.s3_model_tuning.model_tuner import ModelTuner

# specify the model type (one of ScikitMLP, ScikitLinearReg, ScikitLinearRegNoScaler, ScikitSVR, ScikitMLP_TargetTransformed, SciKerasSequential)
model_type = "SciKerasSequential"  # Example model type

# Define the configuration for model tuning
config = ModelTunerConfig(
    models=[model_type],  # Specify the model to train (e.g., SciKerasSequential)
    trainings_per_model=1,          # Number of training iterations per model
    hyperparameter_tuning_type="OptunaTuner",  # Type of hyperparameter tuning
    hyperparameter_tuning_kwargs={"n_trials": 2},  # Number of trials for hyperparameter tuning
)

# Initialize the ModelTuner
model_tuner = ModelTuner(config=config)

# Define training and validation data (replace with actual data)
# Example: x_train_val and y_train_val should be pandas DataFrames/Series
import pandas as pd
import numpy as np



df = pd.read_csv("D:/sle-fmu/Git/prediction-accuracy/hydronic_heatpump/training/trainings_data_addmo.csv")
df.set_index("Time", inplace=True)
df.index = pd.to_datetime(df.index)
df = df.ffill().bfill()
print(df.head())
df_train = df.drop(columns=["T"])
df_test = df["T"]


# Dummy data for demonstration purposes
x_train_val = pd.DataFrame(np.random.rand(100, 5), columns=[f"feature_{i}" for i in range(5)])
y_train_val = pd.Series(np.random.rand(100), name="target")

# Tune the model
print("Tuning the model...")
best_model = model_tuner.tune_model(model_type, df_train, df_test)

# Export the trained model
output_directory = "./exported_models"  # Directory to save the model
os.makedirs(output_directory, exist_ok=True)  # Ensure the directory exists

model_filename = "best_model"  # Name of the exported model file
file_type = "keras"  # File type for saving the model (e.g., 'keras' or 'onnx')

print("Saving the trained model...")
best_model.save_regressor(output_directory, model_filename, file_type=file_type)

print(f"Model saved successfully to {output_directory}/{model_filename}.{file_type}")