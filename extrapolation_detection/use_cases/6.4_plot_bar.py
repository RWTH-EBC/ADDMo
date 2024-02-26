import os

import pandas as pd
from matplotlib import pyplot as plt

import machine_learning_util.data_handling as dh
from use_cases.plot import plot_multiple, plot_bar_plot

#######################################################################################################################
save_plot = True
#######################################################################################################################

if not os.path.exists('plots'):
    os.mkdir('plots')

data = pd.read_csv('results\\results_paper.csv', delimiter=';', index_col=0)
data_ideal = pd.read_csv('results\\results_paper_ideal.csv', delimiter=';', index_col=0)

data.index = data.index.str.replace('_TiV', '')
data.index = data.index.str.replace('IF', 'Isolation Forest')

plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams['font.size'] = 12

plot_bar_plot(data.to_numpy(), data.index, bars_ideal=data_ideal.to_numpy(), labels=['Carnot', r'$P_\mathrm{el}$',r'$\Delta T_\mathrm{zone}$'], ylabel='Mean F-score')

