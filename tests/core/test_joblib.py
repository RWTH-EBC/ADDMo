import pandas as pd
import numpy as np
from core.s3_model_tuning.models.scikit_learn_models import BaseScikitLearnModel
from core.s3_model_tuning.models.scikit_learn_models import ScikitMLP, LinearReg
from core.s3_model_tuning.models.model_factory import ModelFactory
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from core.s3_model_tuning.models.abstract_model import AbstractMLModel

 # Loading dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['price'] = pd.Series(data.target)
X= data.data
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,:-1], df['price'], test_size=0.2, random_state=42)

# Training Scikit Learn Model from BaseScikitLearn class
model1 = LinearReg()
model1.fit(X_train, y_train)
model1.save_regressor(r'C:\Users\mre-rpa\PycharmProjects\addmo\addmo-automated-ml-regression\0000_testfiles', file_type='joblib')
# Load the serialized model
m2= ModelFactory().load_model(r"C:\Users\mre-rpa\PycharmProjects\addmo\addmo-automated-ml-regression\0000_testfiles\LinearReg.joblib")
print(m2)
y_pred= m2.predict(X_test)
r_squared = r2_score(y_test, y_pred)
print("R-squared (1):", r_squared)

#Onnx testing using pandas dataframe

#model2= ScikitMLP()
#model2.fit(X_train, y_train)
#model2.save_regressor(r'C:\Users\mre-rpa\PycharmProjects\addmo\addmo-automated-ml-regression\0000_testfiles', file_type='onnx')
#m3= ModelFactory()
#m3= m3.load_model(r"C:\Users\mre-rpa\PycharmProjects\addmo\addmo-automated-ml-regression\0000_testfiles\ScikitMLP.onnx")
#print(m3)
#y_pred1=m3.predict(X_test)
#print(" R-squared (2):", r_squared2)