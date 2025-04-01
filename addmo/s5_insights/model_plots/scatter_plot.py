import matplotlib.pyplot as plt
from addmo.util import plotting as d


def scatter(train_data, predictions, target_name, rmse):

    plt.figure(figsize= (d.cm2inch(15.5), d.cm2inch(15.5)))
    plt.subplots_adjust(left=0.12, right=0.97, bottom=0.08, top=0.95)
    plt.scatter(train_data, predictions, color=d.blue, label='Predictions')
    min_val = min(train_data.min(), predictions.min())
    max_val = max(train_data.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], color=d.red, linestyle='--', label="Target Value")
    plt.gca().text(0.85, 0.1, f'RMSE: {rmse:.2f}', color=d.black, ha='center', va='center',transform=plt.gca().transAxes, bbox=dict(facecolor='white',alpha=0.7, edgecolor=d.black))
    plt.xlabel("Training Data")
    plt.ylabel("Predicted Values")
    plt.title(target_name)
    plt.legend()
    plt.grid(True)
    plt.axis('tight')
    return plt