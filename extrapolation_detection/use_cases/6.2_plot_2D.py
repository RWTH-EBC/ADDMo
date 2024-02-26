import os
import machine_learning_util.data_handling as dh
from use_cases.plot import plot_single

#######################################################################################################################
save_plot = True
names = [
    # 'Carnot_short',
    'Carnot_mid',
    # 'Carnot_long',
    # 'Carnot_uncertain_short',
    'Carnot_uncertain_mid',
    # 'Carnot_uncertain_long',
    ]
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
    # 'kNN_untuned',
    # 'KDE_untuned',
    # 'GPR_untuned',
    # 'OCSVM_untuned',
    # 'IF_untuned',
    # 'kNN_untuned_test',
    # 'KDE_untuned_test',
    # 'GPR_untuned_test',
    # 'OCSVM_untuned_test',
    ]
#######################################################################################################################

if not os.path.exists('plots'):
    os.mkdir('plots')

for clf_name in clf_names:
    for name in names:

        # Read data
        data = dh.read_pkl('data', name)
        errors = dh.read_pkl('errors', name)
        data_error = dh.read_pkl('data_error', name)
        validity_domain = dh.read_pkl('validity_domain', name)
        score_2D_dct = dh.read_pkl('errors_2D', name)
        validity_domain_dct = dh.read_pkl('validity_domain', name)
        novelty_2D_dct = dh.read_pkl('errors_2D_' + clf_name, name)
        threshold = dh.read_pkl(clf_name + '_threshold', name)

        # Plot data
        plot_single(data, errors, data_error, score_2D_dct, validity_domain_dct, novelty_2D_dct, threshold, name + '_' +
                    clf_name + '.pdf', save_plot)
