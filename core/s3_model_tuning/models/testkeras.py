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
X=df.iloc[:, :-1]

# Step 2: Preprocess the data (optional)
# Here, we'll standardize the features using StandardScaler
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, df['Species'], test_size=0.2, random_state=42)
input_shape = X_train.shape[1]  # Number of features
output_shape = len(set(y_train))  # Number of classes
# Step 4: Create an instance of the BaseKerasModel class
model = BaseKerasModel(input_shape, output_shape)

# Step 5: Compile the model
model.compile_model()

# Step 6: Fit the model on training data
model.fit(X_train, y_train)
model.save_regressor(r'D:\PyCharm 2023.3.5\pythonProject\addmo-automated-ml-regression\0000_testfiles', 'kera1', file_type='keras')
# Step 7: Use the trained model for prediction
model2= ModelFactory()
model2.load_model(r"D:\PyCharm 2023.3.5\pythonProject\addmo-automated-ml-regression\0000_testfiles\kera1.keras")
print(model2)
print(model)
predictions = model2.predict(X_test)

# Print predictions
print("Predictions:", predictions)