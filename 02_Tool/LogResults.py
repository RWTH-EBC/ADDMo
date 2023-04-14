import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from math import exp
import csv
import os
from Functions.ErrorMetrics import*
from sklearn.model_selection import KFold

import statistics

annweight=[]
def log_annweight(weight):
    global annweight
    annweight.append(weight)

def calculate_paramANN(params,inputlayersize,outputlayersize): #Bei ANN anzahl der verbindungen der Neuronen zwischen den Layers. Unter der Annahme ANN ist vollständig verknüpft

    if type(params) is int:
        numparams = (inputlayersize+1)*params+(params+1)*outputlayersize  #+param= bias
    else:
        lendict = len(params)
        temp1 = 0
        for i in range(lendict-1):
            temp1 = temp1 + (params[i]+1) * params[i+1]  #hidden Layer verbindungen bias
        numparams = (inputlayersize+1)*params[0] + temp1 + (params[lendict-1]+1)*outputlayersize
    return numparams

def calc_paramANNweights(fitted_model):  #Problem negative weights. Betrag nehmen
    weightmatrix= fitted_model.coefs_
    biasmatrix = fitted_model.intercepts_
    tempmatrix=0
    for i in range(len(weightmatrix)):
        tempweight=0
        tempbias=0
        for x in np.nditer(weightmatrix[i]):
            tempweight = tempweight + abs(x)
        for y in np.nditer(biasmatrix[i]):
            tempbias= tempbias+ abs(y)
        tempmatrix=tempmatrix+tempweight +tempbias

    numparam = tempmatrix
    return numparam

def calc_paramRF(estimator):

    temp = 0
    for i in range(len(estimator)):
        treei = estimator[i]
        nodei= calc_leafnodes(treei)
        temp = temp+ nodei
    numparams = temp
    return numparams

def calc_leafnodes(treei):
    n_nodes = treei.tree_.node_count
    children_left=treei.tree_.children_left
    children_right=treei.tree_.children_right
    numleaf = 0
    for i in range(n_nodes):
        if children_left[i] == children_right[i]:       # leaf node falls child_left == child_right, falls False ist es ein split node
            numleaf +=1

    return numleaf
def calc_paramSVRV1(params):        # Nur C Cost beachtet in V1 : großes C-> mehr acc und mehr complexity-> evtl C als Komplexitätsaparameter
    numparams = params
    #numparams = exp(params)

    return numparams
def calc_paramNuSVR(estimator):
    sv = estimator.support_
    numparam = sv.size
    return numparam

def cross_val_ic(Estimator,Features_train,Signal_train, CV, typ):  #X : Features_Train, Y: Signal_train
    #trained_model = Estimator.fit(Feat)
    kf = KFold(n_splits=CV)
    X_train=[]
    X_test=[]
    Y_train=[]
    Y_test=[]
    aicscore=[]
    bicscore=[]
    for train_index,test_index in kf.split(Features_train,Signal_train):
        X_train.append(Features_train[train_index])
        X_test.append(Features_train[test_index])
        Y_train.append(Signal_train[train_index])
        Y_test.append(Signal_train[test_index])

    if typ == "RF":
        for i in range(CV):
            trained_model=Estimator.fit(X_train[i],Y_train[i])
            predicted= trained_model.predict(X_test[i])
            paramestimator = trained_model.estimators_
            numparams = calc_paramRF(paramestimator)
            aicscore.append(calculate_AIC(len(Y_train[i]), numparams, Y_test[i], predicted))
            bicscore.append(calculate_BIC(len(Y_train[i]),numparams,Y_test[i],predicted))
    elif typ =="ANN":
        params = Estimator.hidden_layer_sizes
        numparams = calculate_paramANN(params, X_train[0].shape[1], 1) # traindaten alle gleiche größe
        weights=[]
        for i in range(CV):
            trained_model = Estimator.fit(X_train[i], Y_train[i])
            numparamsweight = calc_paramANNweights(trained_model)
            weights.append(numparamsweight)

            predicted = trained_model.predict(X_test[i])

            aicscore.append(calculate_AIC(len(Y_train[i]), numparamsweight, Y_test[i], predicted))
            bicscore.append(calculate_BIC(len(Y_train[i]), numparamsweight, Y_test[i], predicted))
        meanweight=statistics.mean(weights)
        log_annweight(meanweight)
    elif typ=="SVR":
        for i in range(CV):
            trained_model = Estimator.fit(X_train[i],Y_train[i])
            predicted = trained_model.predict(X_test[i])

            #numparam=calc_paramSVRV1(Estimator.C) #Falls AIC mit Hyperparameter C berechnet werden soll
            #aicscore.append(calculate_AIC(len(Y_train[i]), numparam, Y_test[i], predicted))
            #bicscore.append(calculate_BIC(len(Y_train[i]), numparam, Y_test[i], predicted))

            aicscore.append(calc_AICSVR(len(Y_test[i]), Estimator.epsilon, Y_test[i], predicted, X_train[0].shape[1]))
            bicscore.append(calc_BICSVR(len(Y_test[i]), Estimator.epsilon, Y_test[i], predicted, X_train[0].shape[1]))
    elif typ=="NuSVR":
        for i in range(CV):
            trained_model = Estimator.fit(X_train[i], Y_train[i])
            numparam = calc_paramNuSVR(trained_model)
            predicted = trained_model.predict(X_test[i])
            aicscore.append(calculate_AIC(len(Y_train[i]), numparam, Y_test[i], predicted))
            bicscore.append(calculate_BIC(len(Y_train[i]), numparam, Y_test[i], predicted))
    aic = statistics.mean(aicscore)
    bic = statistics.mean(bicscore)

    return aic, bic

def calc_weights(list):  # biclist löschen da algo gleich

    listmin = min(list)   #Kleinster AIC Wert
    deltalist = []
    listweights=[]
    for i in range(len(list)):       #Differenz der Aic vom kleinsten AIC
        deltalist.append(list[i]-listmin)

    temp1 = 0
    for i in range(len(deltalist)):
        temp1 = temp1 + exp(-deltalist[i]/2)
    listnenner = temp1
    #Berechne Gewichte vom AIC
    for i in range(len(deltalist)):
        listweights.append((exp(-deltalist[i]/2))/listnenner)
    #Kontrolle  sollte 1 ergeben
    #kontrolle1=0
    #for i in range(len(aicweights)):
    #    kontrolle1 = kontrolle1+ aicweights[i]
    return listweights
def plot_data(inputpfad):
    data= pd.read_excel('inputpfad')
    timedata = data.Time
    aicdata = data.AIC

    timeplot = plt.plot()

