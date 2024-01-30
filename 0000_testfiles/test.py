from sklearn.neural_network import MLPClassifier  # or MLPRegressor for regression problems
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Create the pipeline with a scaler and an MLP model
mlp_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Feature scaling
    ('mlp', MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42))  # MLP model
])


from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris  # Example dataset

# Load an example dataset
X, y = load_iris(return_X_y=True)

# Perform cross-validation
# Replace 'accuracy' with an appropriate scoring method for your problem
scores = cross_val_score(mlp_pipeline, X, y, cv=5, scoring='accuracy')

# Output the results
print("Cross-validation scores:", scores)
print("Average score:", scores.mean())
