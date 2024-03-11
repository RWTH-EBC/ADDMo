import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from exploration_quantification import point_generator
from exploration_quantification.exploration_quantifier import ExplorationQuantifier

# Define a classifier
classifier = KNeighborsClassifier()

# Define a pandas DataFrame with 6 variables
xy = pd.DataFrame({
    'var1': np.random.rand(10)*10,
    'var2': np.random.rand(10),
    'var3': np.random.rand(10),
    'var4': np.random.rand(10),
    # 'var5': np.random.rand(10),
    # 'var6': np.random.rand(10)
})

# Define the boundaries for each variable
xy_boundaries = {
    'var1': (-10, 20),
    'var2': (-10, 20),
    'var3': (-10, 20),
    'var4': (-10, 20),
    # 'var5': (-10, 20),
    # 'var6': (0, 2)
}

# Fit the classifier to the data
classifier.fit(xy, np.random.randint(0, 2, 10))

num_points = 10

bounds = point_generator.infer_meshgrid_bounds(xy)
points = point_generator.generate_point_grid(xy, xy_boundaries, num_points)

quantifier = ExplorationQuantifier(points, bounds)
quantifier.train_exploration_classifier()
quantifier.calc_labels()
volume_percentage = quantifier.calculate_exploration_percentages(classifier)
quantifier.plot_scatter_extrapolation_share_2D()