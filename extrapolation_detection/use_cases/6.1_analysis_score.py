import os

import pandas

import extrapolation_detection.machine_learning_util.data_handling as dh


# creates csv with already available scores
#######################################################################################################################
# TODO: specifiy parameters here
score = 'recall' #f, fbeta, precision, recall
use_cases = [
    'Carnot_mid_oldANN_4_EnergyAI_32Neurons',
    'Carnot_uncertain_mid_32Neurons',
    'Carnot_mid_newANN_4_buildsys_8Neurons',
    # 'Carnot_short',
    # 'Carnot_mid',
    # 'Carnot_long',
    # 'Carnot_uncertain_short',
    # 'Carnot_uncertain_mid',
    # 'Carnot_uncertain_long',
    # 'Boptest_Pel_short',
    # 'Boptest_Pel_mid',
    # 'Boptest_Pel_long',
    # 'Boptest_TAir_short',
    # 'Boptest_TAir_mid',
    # 'Boptest_TAir_long',
]
clfs = [
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
    # # 'KDE_TiV_0.75',
    # 'KDE',
    # 'GPR_TiV_0.75',
    # 'OCSVM_TiV_0.75',
    # 'IF_TiV_0.75',
    # 'kNN_ideal',
    # 'KDE_ideal',
    # 'GPR_ideal',
    # 'OCSVM_ideal',
    # 'IF_ideal',
    # 'kNN_untuned',
    # 'KDE_untuned',
    # 'GPR_untuned',
    # 'OCSVM_untuned',
    # 'IF_untuned',
]
#######################################################################################################################

df = pandas.DataFrame(columns=use_cases, index=clfs)

# Load use cases and classifiers
for use_case in use_cases:
    for clf in clfs:
        data = dh.read_pkl('evaluation_' + clf, use_case)
        df.loc[clf, use_case] = data[score]

# Save data
if not os.path.exists('results'):
    os.mkdir('results')
df.to_csv('results\\' + score + '.csv')

print(f"Data saved to: results\{score}.csv")
