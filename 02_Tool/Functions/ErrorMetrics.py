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
from math import log
from math import exp
from math import factorial
        
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
def residualsumsquares(measuredData,predictData):  #wert fast gleich wie mse
    rss =1/(len(measuredData)) *sum((measuredData-predictData)**2)
    return rss
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

def calculate_AIC(n,num_params,measuredData,predictData):

    mse = mean_squared_error(measuredData, predictData)

    #if (n/num_params) < 40:
        #aic = calculate_AICc(n,mse,num_params)     # AICc vernachlässigen um Ergebnisse nicht zu verfälschen

    #else:
    aic = n*log(mse) + 2*num_params
    return aic
    """
    else:
        listaic=[]
        for i in range(len(num_params)):
            if (n / num_params[i]) < 40:
                aic = calculate_AICc(n, mse, num_params[i])
            else:
                aic = n * log(mse) + 2 * num_params[i]
            listaic.append(aic)
        return listaic
    """

def calculate_BIC(n,num_params, measuredData, predictData):
    mse = mean_squared_error(measuredData, predictData)
    bic = n*log(mse)+ num_params *log(n)
    return bic
    """
    else:
        listbic=[]
        for i in range(len(num_params)):
            bic = n*log(mse)+ num_params[i] *log(n)
            listbic.append(bic)
        return listbic
    """


def calculate_AICc(n,mse,num_params): # korrigierte AIC
    if n-num_params-1==0:
        aicc = n * log(mse) + 2 * num_params + (2 * num_params * (num_params + 1)) / 1  # damit keine division mit 0
    else:
        aicc = n*log(mse) + 2*num_params +(2*num_params*(num_params+1))/(n-num_params-1)
    return aicc
def mixed_kpi1(score,scoreweight,aic,aicweight):
    kpiscore= scoreweight * exp(score) * aicweight * aic
    return kpiscore
def mixed_kpi2(aic,aicweight,bic,bicweight):
    kpiscore = aicweight * aic + bicweight * bic
    return kpiscore
def mixed_kpi3(score,scoreweight,kpi,kpiweight):
    kpiscore= scoreweight * exp(score) * kpiweight*kpi
    return kpiscore

def calc_AICSVR(n,epsilon, measuredData, predictData,features):
    mse = mean_squared_error(measuredData, predictData)
    #measuredData2 = measuredData.values
    R = empiricalrisk(n,epsilon,measuredData,predictData)

    K = 0
    for i in range(features):
        temp = factorial(features)/(factorial(i)*factorial(features-i))
        K = K +temp
    aic =  -(R + 2*K*mse/(n-K))

    return aic
def calc_BICSVR(n,epsilon,measuredData,predictData,features):
    mse = mean_squared_error(measuredData, predictData)
    #measuredData2 = measuredData.values
    R = empiricalrisk(n, epsilon, measuredData, predictData)

    K = 0
    for i in range(features):
        temp = factorial(features) / (factorial(i) * factorial(features - i))
        K = K + temp
    bic = -(R + log(n) * K * mse / (n - K))

    return bic

def empiricalrisk(n,epsilon,measuredData,predictData):
    tempR = 0
    for i in range(n):
        temp =  max((abs(measuredData[i]-predictData[i])-epsilon),0 )
        tempR = tempR+ temp
    R = tempR/n
    return R