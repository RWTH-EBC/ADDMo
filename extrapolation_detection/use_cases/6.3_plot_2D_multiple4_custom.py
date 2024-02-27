import os

from matplotlib import pyplot as plt

from extrapolation_detection.use_cases.plot import plot_multiple_3
from extrapolation_detection.machine_learning_util import data_handling as dh


#######################################################################################################################
save_plot = True
names = [

    # 'Carnot_uncertain_mid_16Neurons',
    'Carnot_mid_oldANN_4_EnergyAI_32Neurons',
    'Carnot_uncertain_mid_32Neurons',
    'Carnot_mid_newANN_4_buildsys_8Neurons',

    # 'Carnot_uncertain_short',
    # 'Carnot_uncertain_long',
    ]
save_name = 'Multiple3_' + names[1]
clf_names = [
    # 'kNN',
    # 'KDE',
    # 'GPR',
    # 'OCSVM',
    # 'IF',
    # 'kNN_0.75',
    # 'KDE_0.75',
    # 'GPR_0.75',
    # 'OCSVM_0.75',
    # 'IF_0.75',
    'kNN_TiV',
    # 'KDE_TiV',
    # 'GPR_TiV',
    # 'OCSVM_TiV',
    # 'IF_TiV',
    # 'kNN_TiV_0.75',
    # 'KDE_TiV_0.75',
    # 'GPR_TiV_0.75',
    # 'OCSVM_TiV_0.75',
    # 'IF_TiV_0.75',
    # 'kNN_untuned_test',
    # 'KDE_untuned_test',
    # 'GPR_untuned_test',
    # 'OCSVM_untuned_test',
    # 'IF_untuned',
    # 'kNN_untuned',
    # 'KDE_untuned',
    # 'GPR_untuned',
    # 'OCSVM_untuned',
    # 'IF_untuned',
    # 'kNN_ideal',
    # 'KDE_ideal',
    # 'GPR_ideal',
    # 'OCSVM_ideal',
    # 'IF_ideal',
    ]
#######################################################################################################################

if not os.path.exists('plots'):
    os.mkdir('plots')

data = list()
errors = list()
data_error = list()
validity_domain = list()
score_2D_dct = list()
validity_domain_dct = list()
novelty_2D_dct = list()
threshold = list()
title = list()

for clf_name in clf_names:
    for name in names:

        # Read data
        data.append(dh.read_pkl('data', name))
        errors.append(dh.read_pkl('errors', name))
        data_error.append(dh.read_pkl('data_error', name))
        validity_domain.append(dh.read_pkl('true_validity_classified_train_test_val', name))
        score_2D_dct.append(dh.read_pkl('errors_2D', name))
        validity_domain_dct.append(dh.read_pkl('true_validity_classified_train_test_val', name))
        novelty_2D_dct.append(dh.read_pkl('errors_2D_' + clf_name, name))
        threshold.append(dh.read_pkl(clf_name + '_threshold', name))
        # title.append(clf_name.replace('_TiV', '').replace('IF', 'Isolation Forest'))

title=["32 Neurons", "32 Neurons + Uncertainty", "8 Neurons"]

plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# Plot data
plot_multiple_3(data, errors, data_error, score_2D_dct, validity_domain_dct, novelty_2D_dct, threshold,
              save_name + '.pdf', save_plot, title)
