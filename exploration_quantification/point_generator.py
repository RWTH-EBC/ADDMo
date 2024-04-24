import warnings

import numpy as np
import pandas as pd

from exploration_quantification import point_generator

'''
This module contains functions to generate artificial samples, e.g. gridded.
'''

def _check_bounds(bounds, df):
    '''Check if the bounds are valid for the dataframe.
    The bounds should be a dictionary with the variable names as keys and
    the tuple of the lower and upper bounds as values.
    '''
    for var in df.columns:
        if var not in bounds:
            warnings.warn(f"Variable {var} is not defined in the bounds.")
        if not isinstance(bounds[var], (tuple, list)):
            raise ValueError(f"Bounds for variable {var} should be a tuple.")
        if len(bounds[var]) != 2:
            raise ValueError(f"Bounds for variable {var} should have length 2.")
        if bounds[var][0] >= bounds[var][1]:
            raise ValueError(f"Lower bound for variable {var} should be less than the upper bound.")

def _infer_meshgrid_bounds(df: pd.DataFrame):
    '''Infer the boundaries of the meshgrid from the dataframe.
    The boundaries are used to generate the meshgrid for the 2D plot.
    Df should contain the variables to be gridded over.
    '''

    # Get bounds of nD plot
    bounds = {}
    for column in df.columns:
        min_val = df[column].min()
        max_val = df[column].max()
        bounds[column] = (min_val, max_val)
    return bounds

def infer_or_forward_bounds(bounds: dict, df: pd.DataFrame)-> dict:
    '''Return bounds or infer bounds if desired. Also check if bounds are valid.'''

    if bounds == "infer":
        bounds = _infer_meshgrid_bounds(df)
    else:
        _check_bounds(bounds, df)
        bounds = bounds

    return bounds


def generate_random_points(df, bounds, num_points_per_variable=100):
    num_points = np.prod(num_points_per_variable)
    random_points = np.array(
        [
            np.random.uniform(
                low=bounds[var][0], high=bounds[var][1],
                size=num_points
            )
            for var in df.columns
        ]
    ).T
    random_points_df = pd.DataFrame(random_points, columns=df.columns)
    return random_points_df

def generate_point_grid(df, bounds, num_points_per_variable=100):
    # Generate a grid of points within the specified boundaries for each variable
    grids = np.meshgrid(*[
        np.linspace(start=bounds[var][0], stop=bounds[var][1],
                    num=num_points_per_variable, dtype=np.float32)
        for var in df.columns
    ])

    # Reshape and stack the grids to get a single array of points
    grid_points = np.vstack([grid.ravel() for grid in grids]).T

    grid_points_df = pd.DataFrame(grid_points, columns=df.columns)
    return grid_points_df
