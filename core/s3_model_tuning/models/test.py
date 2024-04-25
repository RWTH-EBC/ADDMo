from core.s3_model_tuning.models.scikit_learn_models import BaseScikitLearnModel
from core.s3_model_tuning.models.scikit_learn_models import MLP, LinearReg
from core.s3_model_tuning.models.model_factory import ModelFactory
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from core.s3_model_tuning.models.abstract_model import AbstractMLModel

data = fetch_california_housing()
#print(data.DESCR )
#print(data)
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
print(data.feature_names)
#instance of the model
model1 = MLP()
model2= LinearReg()
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model1.save_model("mlp.joblib")
model1.save_model("mlp.onnx")
model2.save_model("linear.joblib")
model2.save_model("linear.onnx")
# Load the serialized model
model1_joblib= ModelFactory()
model1_onnx= ModelFactory()
model1_joblib= model1_joblib.load_model("mlp.joblib")
model1_onnx= model1_onnx.load_model('mlp.onnx')
model2_joblib= ModelFactory()
model2_onnx= ModelFactory()
model2_joblib= model2_joblib.load_model("linear.joblib")
model2_onnx= model2_onnx.load_model('linear.onnx')
print('object type for MLP joblib model is: ', model1_joblib)
print('object type for MLP onnx model is: ', model1_onnx)
print('object type for Linear onnx model is: ', model2_onnx)
print('object type for Linear joblib model is: ', model2_joblib)

y_pred1= model1_joblib.predict(X_test)
y_pred2=model1_onnx.predict(X_test)
y_pred3=model2_joblib.predict(X_test)
y_pred4= model2_onnx.predict(X_test)
r_squared1 = r2_score(y_test, y_pred1)
r_squared2= r2_score(y_test, y_pred2)
r_squared3= r2_score(y_test, y_pred3)
r_squared4= r2_score(y_test, y_pred4)

print("R-squared coeff for mlp joblib model:", r_squared1)
print(" R-squared coeff for mlp onnx model:", r_squared2)
print(" R-squared coeff for linear joblib model:", r_squared3)
print(" R-squared coeff for linear onnx model:", r_squared4)
