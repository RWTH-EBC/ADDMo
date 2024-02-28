from extrapolation_detection.detector.detector_tuners import Hyper_KNN, Hyper_KDE, Hyper_GP, Hyper_OCSVM, Hyper_IF
from extrapolation_detection.use_cases.train_clf import train_clf

#######################################################################################################################
# TODO: specifiy parameters here
beta = 1
outlier_fraction = 0.05 # irrelevant but necessary to state
use_train_for_validation = True

clfs = {
    'kNN_TiV': Hyper_KNN,
    # 'KDE': Hyper_KDE,
    # 'GPR_TiV': Hyper_GP,
    # 'OCSVM_TiV': Hyper_OCSVM,
    # 'IF_TiV': Hyper_IF,
}
#######################################################################################################################
# name = 'Carnot_uncertain_short'
#
# for clf_name, clf_callback in clfs.items():
#     train_clf(name, clf_name, clf_callback, outlier_fraction, beta=beta,
#               use_train_for_validation=use_train_for_validation)
#######################################################################################################################
# name = 'Carnot_uncertain_mid'
#
# for clf_name, clf_callback in clfs.items():
#     train_clf(name, clf_name, clf_callback, outlier_fraction, beta=beta,
#               use_train_for_validation=use_train_for_validation)
#######################################################################################################################
name = 'Carnot_uncertain_mid_32Neurons'

for clf_name, clf_callback in clfs.items():
    train_clf(name, clf_name, clf_callback, outlier_fraction, beta=beta,
              use_train_for_validation=use_train_for_validation)
# #######################################################################################################################
# # name = 'Carnot_uncertain_long'
# #
# # for clf_name, clf_callback in clfs.items():
# #     train_clf(name, clf_name, clf_callback, outlier_fraction, beta=beta,
# #               use_train_for_validation=use_train_for_validation)
# #######################################################################################################################
# name = 'Boptest_Pel_short'
#
# for clf_name, clf_callback in clfs.items():
#     train_clf(name, clf_name, clf_callback, outlier_fraction, beta=beta,
#               use_train_for_validation=use_train_for_validation)
# #######################################################################################################################
# name = 'Boptest_Pel_mid'
#
# for clf_name, clf_callback in clfs.items():
#     train_clf(name, clf_name, clf_callback, outlier_fraction, beta=beta,
#               use_train_for_validation=use_train_for_validation)
# #######################################################################################################################
# name = 'Boptest_Pel_long'
#
# for clf_name, clf_callback in clfs.items():
#     train_clf(name, clf_name, clf_callback, outlier_fraction, beta=beta,
#               use_train_for_validation=use_train_for_validation)
# #######################################################################################################################
# name = 'Boptest_TAir_short'
#
# for clf_name, clf_callback in clfs.items():
#     train_clf(name, clf_name, clf_callback, outlier_fraction, beta=beta,
#               use_train_for_validation=use_train_for_validation)
# #######################################################################################################################
# name = 'Boptest_TAir_mid'
#
# for clf_name, clf_callback in clfs.items():
#     train_clf(name, clf_name, clf_callback, outlier_fraction, beta=beta,
#               use_train_for_validation=use_train_for_validation)
# #######################################################################################################################
# name = 'Boptest_TAir_long'
#
# for clf_name, clf_callback in clfs.items():
#     train_clf(name, clf_name, clf_callback, outlier_fraction, beta=beta,
#               use_train_for_validation=use_train_for_validation)
# #######################################################################################################################
# name = 'Carnot_short'
#
# for clf_name, clf_callback in clfs.items():
#     train_clf(name, clf_name, clf_callback, outlier_fraction, beta=beta,
#               use_train_for_validation=use_train_for_validation)
#######################################################################################################################
# name = 'Carnot_mid'
#
# for clf_name, clf_callback in clfs.items():
#     train_clf(name, clf_name, clf_callback, outlier_fraction, beta=beta,
#               use_train_for_validation=use_train_for_validation)
#######################################################################################################################
# name = 'Carnot_long'
#
# for clf_name, clf_callback in clfs.items():
#     train_clf(name, clf_name, clf_callback, outlier_fraction, beta=beta,
#               use_train_for_validation=use_train_for_validation)
