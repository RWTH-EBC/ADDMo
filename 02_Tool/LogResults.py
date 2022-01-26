import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from math import exp
import csv
import os


def calculate_paramANN(params,inputlayersize,outputlayersize): #Bei ANN anzahl der verbindungen der Neuronen zwischen den Layers. Unter der Annahme ANN ist vollständig verknüpft

    lendict = len(params)

    if lendict == 1:  # nur ein Eintrag für params
        layerparams = params['hidden_layer_sizes']
        if type(layerparams) is tuple or type(layerparams) is list:  # hidden layer has more than one layer
            anzahllayerparams = len(layerparams)
            temp2 = 0
            for i in range(anzahllayerparams):
                if i == 0:  # inputlayer mit ersten hidden layer multiplizieren
                    temp1 = inputlayersize * layerparams[i]
                else:
                    temp1 = layerparams[(i - 1)] * layerparams[i]
                temp2 = temp2 + temp1
            numparams=temp2 + layerparams[(anzahllayerparams - 1)] * outputlayersize
        elif type(layerparams) is int:  #hidden layer has only one layer
            temp1 = inputlayersize * layerparams
            numparams=temp1 + layerparams * outputlayersize
    """
    else:   #Bestparams
        for key in params:
            templayerparams = params[key]
            layerparams = templayerparams['hidden_layer_sizes']
            if type(layerparams) is tuple or type(layerparams) is list:  # hidden layer has more than one layer
                anzahllayerparams = len(layerparams)
                temp2 = 0
                for i in range(anzahllayerparams):
                    if i == 0:  # inputlayer mit ersten hidden layer multiplizieren
                        temp1 = inputlayersize * layerparams[i]
                    else:
                        temp1 = layerparams[(i - 1)] * layerparams[i]
                    temp2 = temp2 + temp1
                listparam.append(temp2 + layerparams[(anzahllayerparams - 1)] * outputlayersize)
            elif type(layerparams) is int:  # hidden layer has only one layer
                temp1 = inputlayersize * layerparams
                listparam.append(temp1 + layerparams * outputlayersize)
                """


    return numparams

def calc_paramRF(params):
    numparams = params['max_depth'] * params['n_estimators']        #andere Formel?

    return numparams
def calc_paramSVRV1(params):        # Nur C Cost beachtet in V1 : großes C-> mehr acc und mehr complexity-> evtl C als Komplexitätsaparameter
    numparams = params
    #numparams = exp(params)

    return numparams

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
    """
    # Gleiches Prinzip für BIC
    bicmin = min(biclist)
    deltabic = []
    bicweights = []
    for i in range(len(biclist)):
        deltabic.append(biclist[i]-bicmin)
    temp2 = 0
    for i in range(len(deltabic)):
        temp2 = temp2 + exp(-deltabic[i]/2)
    bicnenner = temp2
    for i in range(len(deltabic)):
        bicweights.append((exp(-deltabic[i]/2))/bicnenner)
    kontrolle2 = 0
    for i in range(len(bicweights)):
        kontrolle2 = kontrolle2 + bicweights[i]

    """


    return listweights


def plot_data(inputpfad):
    data= pd.read_excel('inputpfad')
    timedata = data.Time
    aicdata = data.AIC

    timeplot = plt.plot()


"""
def log_ann(trainstep, time, aic,bic):

    trainlist = []
    trainlist.append(trainstep)
    timelist = []
    timelist.append(time)
    aiclist = []
    aiclist.append(aic)
    biclist = []
    biclist.append(bic)
    desier_path =  "D:\\thi-dpo\ADDMo\\addmo-automated-ml-regression\\04_CSV"
    file_path = os.path.join(desier_path, 'log_ann.csv')
    with open(file_path, 'w', newline='') as f:
        fieldnames = ['TrainStep','Time', 'Aic', 'Bic']
        thewriter = csv.DictWriter(f, fieldnames=fieldnames)

        thewriter.writeheader()
        for i in range(0,len(trainlist)):
            thewriter.writerow({'TrainStep' : trainlist[i], 'Time' : timelist[i], 'Aic': aiclist[i], 'Bic' : biclist[i]})



def log_svr():
    continue

def log_rf():
    continue

def save_log(trainlist,timelist,aiclist,biclist):
    desier_path = "D:\\thi-dpo\ADDMo\\addmo-automated-ml-regression\\04_CSV"
    file_path = os.path.join(desier_path, 'log_ann.csv')
    with open(file_path, 'w', newline='') as f:
        fieldnames = ['TrainStep', 'Time', 'Aic', 'Bic']
        thewriter = csv.DictWriter(f, fieldnames=fieldnames)

        thewriter.writeheader()
        for i in range(0, len(trainlist)):
            thewriter.writerow({'TrainStep': trainlist[i], 'Time': timelist[i], 'Aic': aiclist[i], 'Bic': biclist[i]})
"""