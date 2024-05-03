import pandas as pd
import numpy as np
from core.s3_model_tuning.models.scikit_learn_models import BaseScikitLearnModel
from core.s3_model_tuning.models.scikit_learn_models import ScikitMLP, LinearReg
from core.s3_model_tuning.models.model_factory import ModelFactory
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from core.s3_model_tuning.models.abstract_model import AbstractMLModel

# Load dataset as numpy array
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
X= data.data
y= data.target
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# Training Scikit Learn Model from BaseScikitLearn class
model= ScikitMLP()
model.fit(X_train, y_train)
model.save_regressor(r'C:\Users\mre-rpa\PycharmProjects\addmo\addmo-automated-ml-regression\0000_testfiles', file_type='onnx')

# Load the serialized model
model1= ModelFactory().load_model(r"C:\Users\mre-rpa\PycharmProjects\addmo\addmo-automated-ml-regression\0000_testfiles\ScikitMLP.onnx")
print(model1)
y_pred=model1.predict(X_test)
r_squared= r2_score(y_test, y_pred)
print(" R-squared:", r_squared)