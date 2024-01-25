from sklearn.neural_network import MLPRegressor

from core.model_tuning.models.abstract_model import AbstractMLModel


class ScikitLearnBaseModel(AbstractMLModel):
    def __init__(self, model):
        self.model = model  # Create an instance of the scikit-learn model

    def fit(self, x, y):
        self.model.fit(x, y)  # Train the model

    def infer(self, x):
        return self.model.predict(x)  # Make predictions

    def set_hyperparameters(self, hyperparameters):
        self.model.set_params(**hyperparameters)  # Set hyperparameters

    def default_hyperparameter(self):
        return self.model.get_params()

    def save_model(self, path):
        # Implement model saving (e.g., using joblib)
        pass

    def load_model(self, path):
        # Implement model loading
        pass


class MLP(ScikitLearnBaseModel):
    def __init__(self):
        super().__init__(MLPRegressor())

    def optuna_hyperparameter_suggest(self, trial):
        hyperparameters = {}

        # Suggest hyperparameters
        n_layers = trial.suggest_int('n_layers', 1, 5)
        hidden_layer_sizes = tuple(
            trial.suggest_int(f'n_units_l{i}', 10, 100) for i in range(n_layers))

        # Dynamic hidden layer sizes based on the number of layers
        hyperparameters['hidden_layer_sizes'] = hidden_layer_sizes

        # Other hyperparameters
        hyperparameters['activation'] = trial.suggest_categorical('activation',
                                                                  ["identity", "logistic", "tanh",
                                                                   "relu"])
        hyperparameters['solver'] = trial.suggest_categorical('solver', ["lbfgs", "sgd", "adam"])
        hyperparameters['alpha'] = trial.suggest_loguniform('alpha', 1e-5, 1e-1)
        hyperparameters['learning_rate'] = trial.suggest_categorical('learning_rate',
                                                                     ["constant", "invscaling",
                                                                      "adaptive"])
        hyperparameters['max_iter'] = trial.suggest_int('max_iter', 100, 500)

        return hyperparameters

    def grid_search_hyperparameter(self):
        hyperparameter_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant', 'adaptive'],
        }
        return hyperparameter_grid
