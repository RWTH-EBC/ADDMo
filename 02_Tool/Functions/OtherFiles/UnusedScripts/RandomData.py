from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

from Tool.BlackBoxes import *


#Create Features_train
print("Cell Create Features---------------------------------------------------------")
N_SAMPLES_TRAIN = 1000
N_SAMPLES_TEST = 500

Feature1 = np.random.rand(N_SAMPLES_TRAIN, 1)
Feature2 = np.random.rand(N_SAMPLES_TRAIN, 1)
Features_train = np.concatenate([Feature1, Feature2], axis=1)


Feature1_test = np.random.rand(N_SAMPLES_TEST, 1)*2
Feature2_test = np.random.rand(N_SAMPLES_TEST, 1)*2
Features_test = np.concatenate([Feature1_test, Feature2_test], axis=1)


#Create Signal_train
print("Cell Create Signal---------------------------------------------------")
#Define the used Formular in the function "Formular"
def formular(Feature1, Feature2):
    Signal = Feature1 - Feature2
    #Signal = Feature2**3
    return Signal

Signal_train = formular(Feature1, Feature2)
# Add noise to signal
#maximum percentage of noise per sample 0 to 1
MAXNOISE = 0.0
Signal_train += (MAXNOISE * np.random.uniform(-1, 1, (N_SAMPLES_TRAIN, 1)))
Signal_train = Signal_train.ravel()

Signal_test = formular(Feature1_test, Feature2_test)
Signal_test = Signal_test.ravel()

#Predictions
SVR_result_dic = svr_grid_search_predictor(Features_train, Signal_train, Features_test, Signal_test, [{'gamma': [1e4 , 1 , 1e-4, 'auto'],
                                                                                                                    'C': [1e-4, 1, 1e4],
                                                                                                                    'epsilon': [1, 1e-4]}],
                                           5
                                           )
predicted_svr = SVR_result_dic["prediction"]


RF_result_dic = rf_predictor(Features_train, Signal_train, Features_test, Signal_test)
predicted_rf = RF_result_dic["prediction"]


lasso_result_dic = lasso_grid_search_predictor(Features_train, Signal_train, Features_test, Signal_test, [{'alpha':
                                                                                                                       [100000, 10, 1, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]}], 5)
predicted_lasso = lasso_result_dic["prediction"]


ann_result_dic = ann_grid_search_predictor(Features_train, Signal_train, Features_test, Signal_test, [{'hidden_layer_sizes':
                                                                                                                   [[1],[10],[100],[1000],
                                                                                                                    [1, 1],[10, 10], [100, 100],
                                                                                                                    [1,10],[1,100],[10,100],[100,10],[100,1],[10,1],
                                                                                                                    [1, 1, 1],[10, 10, 10],[100,100,100]]}],
                                           5)
predicted_ann = ann_result_dic["prediction"]


print("Cell PlotResults---------------------------------------------------------")
timestart = time.time()

fig = plt.figure()
ax = Axes3D(fig)

ax.scatter(Feature1, Feature2, Signal_train, color='darkorange', label='TrainSet')
ax.scatter(Feature1_test, Feature2_test, Signal_test, color='orange', label='TestSet')
ax.scatter(Feature1_test, Feature2_test, predicted_svr, color='navy', label='predicted SVR')
ax.scatter(Feature1_test, Feature2_test, predicted_rf, color='blue', label='predicted RF')
ax.scatter(Feature1_test, Feature2_test, predicted_lasso, color='aquamarine', label='predicted Lasso')
ax.scatter(Feature1_test, Feature2_test, predicted_ann, color='aqua', label='predicted MLP')

ax.set_xlabel('Feature1')
ax.set_ylabel('Feature2')
ax.set_zlabel('Signal_train')
#ax.set_title('Score SVR: %s, Score RF: %s' %(clf.score(Features_test, Signal_test), rf.score(Features_test, Signal_test)))
ax.legend()
timeend = time.time()
print("Plotting took %s seconds" %(timeend-timestart))
plt.show()