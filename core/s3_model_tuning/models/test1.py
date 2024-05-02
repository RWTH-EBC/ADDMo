import numpy as np
from sklearn.model_selection import train_test_split
from core.s3_model_tuning.models.keras_model import BaseKerasModel
from core.s3_model_tuning.models.model_factory import ModelFactory
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Step 1: Load the iris dataset from scikit-learn
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Species'] = pd.Series(data.target)
print(df.head())
X=df.iloc[:, :-1]

# Step 2: Preprocess the data (optional)
# Here, we'll standardize the features using StandardScaler
# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, df['Species'], test_size=0.2, random_state=42)
# Step 2: Preprocess the data (optional)
# Here, we'll standardize the features using StandardScaler
print(X_train)
input_shape = X_train.shape[1]  # Number of features
output_shape = len(set(y_train))
print(X_train)
print(y_train)
print(input_shape)
print(output_shape)