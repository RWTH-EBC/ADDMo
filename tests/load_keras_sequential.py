from tensorflow import keras

# Load the model
model_path = r"D:\04_GitRepos\addmo-extra\aixtra_use_case\results\Empty\regressors\regressor.keras"
loaded_model = keras.models.load_model(model_path)

# Iterate through the layers and print their activation functions
for layer in loaded_model.layers:
    if hasattr(layer, 'activation'):
        print(f"Layer: {layer.name}")
        print(f"Type: {type(layer).__name__}")
        print(f"Activation: {layer.activation.__name__}")
        print("---")