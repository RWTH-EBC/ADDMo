import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from core.s3_model_tuning.models.keras_model import BaseKerasModel
from core.s3_model_tuning.models.model_factory import ModelFactory
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.preprocessing import StandardScaler

# Step 1: Load the iris dataset from scikit-learn
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Species'] = pd.Series(data.target)
X = df.drop([ 'Species'], axis=1)
Y= df['Species']

# Step 2: Pre-processing
# Keras requires your output feature to be one-hot encoded values.
Y_final = pd.get_dummies(Y, columns=['Species'], prefix= 'Species')
X_train, X_test, y_train, y_test = train_test_split(X, Y_final, test_size=0.2, random_state=42)
input_shape = X_train.shape[1]  # Number of features
output_shape = Y_final.shape[1]   # Number of classes


# Step 4: Create an instance of the BaseKerasModel class
model = BaseKerasModel(input_shape, output_shape)

# Step 5: Compile the model
model.compile_model()

# Step 6: Fit the model on training data
model.fit(X_train, y_train)
model.save_regressor(r'C:\Users\mre-rpa\PycharmProjects\addmo\addmo-automated-ml-regression\0000_testfiles', 'kera1', file_type='keras')

# Step 7: Use the trained model for prediction
model2= ModelFactory().load_model(r"C:\Users\mre-rpa\PycharmProjects\addmo\addmo-automated-ml-regression\0000_testfiles\kera1.keras")
print(model2)
print(model)
y_pred = model2.predict(X_test)

# Print predictions
r_squared = r2_score(y_test, y_pred)
print("R-squared (1):", r_squared)