import pandas as pd

'''If the name of function is "create_" + "variable suffix", the methods are dynamically used 
through the getattr() function in the DataTunerByConfig class.'''

def create_lag(var: pd.Series, lag: int):
    series = var.shift(lag)
    series.name = f"{var.name}___lag{lag}"
    return series

def create_diff(var: pd.Series):
    series = var.diff()
    series.name = f"{var.name}___diff"
    return series

def create_squared(var: pd.Series):
    series = var.pow(2)
    series.name = f"{var.name}___squared"
    return series