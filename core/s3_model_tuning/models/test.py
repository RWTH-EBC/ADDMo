from core.s3_model_tuning.models.scikit_learn_models import BaseScikitLearnModel
from core.s3_model_tuning.models.scikit_learn_models import MLP, LinearReg
from core.s3_model_tuning.models.model_factory import ModelFactory
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

data = fetch_california_housing()
print(data.DESCR )
X, y = data.data, data.target

print(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#instance of the model
model1 = LinearReg()

model1.fit(X_train, y_train)
model1.save_model("linear.joblib")
model1.save_model_onnx("linear.onnx")
# Load the serialized model
m2= ModelFactory()
m2= m2.load_serialized_model("linear.joblib")
m3= ModelFactory()
m3= m3.load_onnx('linear.onnx')
print(m2)
y_pred= m2.predict(X_test)
print(m3)
y_pred1=m3.predict_onnx(X_test)
r_squared = r2_score(y_test, y_pred)
r_squared2= r2_score(y_test, y_pred1)
print("R-squared (1):", r_squared)
print(" R-squared (2):", r_squared2)