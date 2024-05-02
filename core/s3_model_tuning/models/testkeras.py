import numpy as np
from sklearn.model_selection import train_test_split
from core.s3_model_tuning.models.keras_model import BaseKerasModel
from core.s3_model_tuning.models.model_factory import ModelFactory
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Step 1: Load the iris dataset from scikit-learn
iris = load_iris()
X, y = iris.data, iris.target

# Step 2: Preprocess the data (optional)
# Here, we'll standardize the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 4: Create an instance of the BaseKerasModel class
model = BaseKerasModel()

# Step 5: Compile the model
model.compile_model()

# Step 6: Fit the model on training data
model.fit(X_train, y_train)
model.save_regressor(r'D:\04_GitRepos\addmo-extra\0000_testfiles', 'kera1', file_type='keras')
# Step 7: Use the trained model for prediction
model2= ModelFactory()
model2.load_model(r"D:\04_GitRepos\addmo-extra\0000_testfiles\kera1.keras")
predictions = model2.predict(X_test)

# Print predictions
print("Predictions:", predictions)