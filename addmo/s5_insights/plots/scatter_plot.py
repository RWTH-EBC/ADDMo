import matplotlib.pyplot as plt
from addmo.util import plotting as d


def scatter(train_data, predictions, target_name, rmse):
    plt.scatter(train_data, predictions, color=d.blue, label='Predictions')
    plt.plot(train_data, train_data, color=d.red, linestyle='--', label='Perfect Fit')  # Ideal line

    # Line connecting predictions
    plt.plot(train_data, predictions, color=d.green, label='Model Fit')

    plt.text(0.8, max(predictions), f'RMSE: {rmse:.2f}', fontsize=10, color=d.black , ha='center')
    # Labels and legend
    plt.xlabel("Training Data")
    plt.ylabel("Predicted Values")
    plt.title(target_name)
    plt.legend()
    plt.grid(True)
    plt.show()