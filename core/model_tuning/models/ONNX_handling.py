import numpy as np

# Todo: check how scaler should be exported and imported
class ONNXInferenceWrapper:
    def __init__(self, ONNX_Runtime_Model):
        self.session = ONNX_Runtime_Model

    def predict(self, X):
        input_name = self.session.get_inputs()[0].name
        X_onnx = X.astype(np.float32)
        result = self.session.run(None, {input_name: X_onnx})
        return result[0]