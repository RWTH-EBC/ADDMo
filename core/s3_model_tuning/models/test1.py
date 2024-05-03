import pandas as pd
from core.s3_model_tuning.models.scikit_learn_models import BaseScikitLearnModel
from core.s3_model_tuning.models.scikit_learn_models import ScikitMLP, LinearReg
from core.s3_model_tuning.models.model_factory import ModelFactory
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from core.s3_model_tuning.models.abstract_model import AbstractMLModel
import numpy as np

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
X= data.data
print(X)
y= data.target
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
#instance of the model
model2= ScikitMLP()

model2.fit(X_train, y_train)
model2.save_regressor(r'D:\PyCharm 2023.3.5\pythonProject\addmo-automated-ml-regression\0000_testfiles', file_type='onnx')

# Load the serialized model
m3= ModelFactory()
m3= m3.load_model(r"D:\PyCharm 2023.3.5\pythonProject\addmo-automated-ml-regression\0000_testfiles\ScikitMLP.onnx")

print(m3)
y_pred1=m3.predict(X_test)
r_squared2= r2_score(y_test, y_pred1)

print(" R-squared (2):", r_squared2)