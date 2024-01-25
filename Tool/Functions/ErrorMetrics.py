# -*- coding: utf-8 -*-
"""
Created on Thu Sep 01 16:14:20 2016

@author: hha
"""
import statistics
import numpy as np
from scipy import stats
import pandas as pd
import statsmodels
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from statsmodels.tools.eval_measures import rmse
        
def mean_absolute_scaled_error(measuredData, predictData, mae):
    # Calculate the mean absolute scaled error, with IN-SAMPLE naive method
    denominator = pd.Series()

    if len(measuredData) == 1: # is just one value is available
        denom = 1
    else:
        for i in range(len(measuredData)-1):
            ind = (i+1)
            denominator.set_value(predictData.index[ind], predictData[ind] - predictData[ind-1])
            denom = np.mean(np.abs(denominator))
    return mae/denom


def mean_absolute_percentage_error(measuredData, predictData):
    predictData = predictData[(measuredData != 0)]
    measuredData = measuredData[(measuredData != 0)]
    return np.mean(np.abs((measuredData - predictData) / measuredData)) * 100

def standarddeviation(measuredData, predictData):
    error = measuredData - predictData
    STD = statistics.pstdev(error)
    return STD

def evaluation(measuredData, predictData):
    #Evaluation
    R2 = r2_score(measuredData, predictData)
    MAE = mean_absolute_error(measuredData, predictData)
    MSE = mean_squared_error(measuredData, predictData)
    RMSE = rmse(measuredData, predictData)
    error = measuredData - predictData
    SSE = sum((error*error))
    #MASE = mean_absolute_scaled_error(measuredData, predictData, MAE)
    MAPE = mean_absolute_percentage_error(measuredData, predictData)
    STD = standarddeviation(measuredData, predictData)
    return (R2, STD, RMSE, MAPE, MAE)





