import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from exploration_quantification import point_generator
from exploration_quantification.exploration_quantifier import ExplorationQuantifier
from exploration_quantification.exploration_quantifier import GridOccupancy
from exploration_quantification import coverage_plotting

# Define a pandas DataFrame with 6 variables
xy = pd.DataFrame({
    'var1': np.random.rand(100),
    'var2': np.random.rand(100),
    'var3': np.random.rand(100),
    # 'var4': np.random.rand(10),
    # 'var5': np.random.rand(10),
    # 'var6': np.random.rand(10)
})

xy.loc[len(xy)] = [1, 1, 1]


# Define the boundaries for each variable
bounds = {
    'var1': (0, 1),
    'var2': (0, 3),
    'var3': (0, 2),
    # 'var4': (0, 2),
    # 'var5': (-10, 20),
    # 'var6': (0, 2)
}

num_points = 12
# points = point_generator.generate_point_grid(xy, bounds, num_points)

x = xy

# coverage_plotting.plot_dataset_parallel_coordinates_plotly(x, bounds, "hallo")
# coverage_plotting.plot_dataset_parallel_coordinates_plotly(x, bounds, "hallo")
# coverage_plotting.plot_dataset_distribution_kde(x, bounds, "hallo")
# coverage_plotting.plot_dataset_parallel_coordinates(x, "hallo")

grid_occupancy = GridOccupancy(100)
grid_occupancy.train(x, bounds)
# grid_occupancy.calculate_coverage()
# grid_occupancy.plot_grid(x, grid_occupancy.occupancy_grid, "Occupancy Grid")




# quantifier.train_exploration_classifier("ConvexHull")
# quantifier.calc_labels()
# volume_percentage = quantifier.calculate_exploration_percentages()
# quantifier.plot_scatter_extrapolation_share_2D()
#
# quantifier2 = ExplorationQuantifier(x, points, bounds)
# quantifier2.train_exploration_classifier("KNN")
# quantifier2.calc_labels()
# volume_percentage2 = quantifier2.calculate_exploration_percentages()
# quantifier2.plot_scatter_extrapolation_share_2D()

print(volume_percentage)
print(quantifier.points_labeled)