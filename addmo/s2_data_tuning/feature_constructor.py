import pandas as pd

"""If the name of function is "create_" + "variable suffix", the methods are dynamically used 
through the getattr() function in the DataTunerByConfig class."""

def create_lag(var: pd.Series, lag: int):
    """
    Creates a lagged version of the input series.
    """
    series = var.shift(lag)
    series.name = f"{var.name}___lag{lag}"
    return series

def create_diff(var: pd.Series):
    """
    Creates a differenced version of the input series.
    """
    series = var.diff()
    series.name = f"{var.name}___diff"
    return series

def create_squared(var: pd.Series):
    """
    Creates a squared version of the input series.
    """
    series = var.pow(2)
    series.name = f"{var.name}___squared"
    return series