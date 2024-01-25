import pandas as pd

'''It is important that the name of variable is equal to the names in DataTunerByConfig'''

def create_lag(var: pd.Series, lag: int):
    series = var.shift(lag)
    series.name = f"{var.name}___lag{lag}"
    return series

def create_difference(var: pd.Series):
    series = var.diff()
    series.name = f"{var.name}___diff"
    return series

def create_squared(var: pd.Series):
    series = var.pow(2)
    series.name = f"{var.name}___squared"
    return series