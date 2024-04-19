from core.s3_model_tuning.models.scikit_learn_models import BaseScikitLearnModel
from core.s3_model_tuning.models.scikit_learn_models import MLP, LinearReg
from core.s3_model_tuning.models.model_factory import ModelFactory
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

data = load_boston()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#instance of the model
model1 = LinearReg()

model1.fit(X_train, y_train)
model1.save_model("linear.joblib")

# Load the serialized model
m2= ModelFactory()
m2= m2.load_serialized_model("linear.joblib")

print(m2)
y_pred= m2.predict(X_test)

r_squared = r2_score(y_test, y_pred)
print("R-squared:", r_squared)
