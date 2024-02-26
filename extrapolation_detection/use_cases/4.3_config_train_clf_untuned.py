from sklearn.gaussian_process.kernels import RBF

from detector.detector import D_KNN, D_GP, D_ParzenWindow, D_OCSVM, D_IF
from use_cases.train_clf_untuned import train_clf_untuned

#######################################################################################################################
# TODO: specifiy parameters here
outlier_fraction = 0.05

clfs = {
    # 'kNN_untuned': D_KNN(contamination=outlier_fraction, n_neighbors=5, method='largest', p=2),
    # 'KDE_untuned': D_ParzenWindow(contamination=outlier_fraction, kernel='gaussian', bandwith=1.0),
    # 'GPR_untuned': D_GP(contamination=outlier_fraction, kernel=None),
    # 'OCSVM_untuned': D_OCSVM(contamination=outlier_fraction, nu=0.5, kernel='rbf', gamma='auto'),
    # 'IF_untuned': D_IF(contamination=outlier_fraction, random_state=0),
    'kNN_untuned_test': D_KNN(contamination=outlier_fraction, n_neighbors=20, method='mean', p=1),
    'KDE_untuned_test': D_ParzenWindow(contamination=outlier_fraction, kernel='gaussian', bandwith=0.1),
    'GPR_untuned_test': D_GP(contamination=outlier_fraction, kernel=1.0 * RBF(length_scale=100, length_scale_bounds='fixed')),
    'OCSVM_untuned_test': D_OCSVM(contamination=outlier_fraction, nu=0.05, kernel='rbf', gamma=10),
    # 'IF_untuned': D_IF(contamination=outlier_fraction, random_state=0),
}
# #######################################################################################################################
# name = 'Carnot_uncertain_short'
#
# for clf_name, clf_callback in clfs.items():
#     train_clf_untuned(name, clf_name, clf_callback, outlier_fraction)
#######################################################################################################################
name = 'Carnot_uncertain_mid'

for clf_name, clf_callback in clfs.items():
    train_clf_untuned(name, clf_name, clf_callback, outlier_fraction)
# #######################################################################################################################
# name = 'Carnot_uncertain_long'
#
# for clf_name, clf_callback in clfs.items():
#     train_clf_untuned(name, clf_name, clf_callback, outlier_fraction)
# #######################################################################################################################
# name = 'Boptest_Pel_short'
#
# for clf_name, clf_callback in clfs.items():
#     train_clf_untuned(name, clf_name, clf_callback, outlier_fraction)
# #######################################################################################################################
# name = 'Boptest_Pel_mid'
#
# for clf_name, clf_callback in clfs.items():
#     train_clf_untuned(name, clf_name, clf_callback, outlier_fraction)
# #######################################################################################################################
# name = 'Boptest_Pel_long'
#
# for clf_name, clf_callback in clfs.items():
#     train_clf_untuned(name, clf_name, clf_callback, outlier_fraction)
# #######################################################################################################################
# name = 'Boptest_TAir_short'
#
# for clf_name, clf_callback in clfs.items():
#     train_clf_untuned(name, clf_name, clf_callback, outlier_fraction)
# #######################################################################################################################
# name = 'Boptest_TAir_mid'
#
# for clf_name, clf_callback in clfs.items():
#     train_clf_untuned(name, clf_name, clf_callback, outlier_fraction)
# #######################################################################################################################
# name = 'Boptest_TAir_long'
#
# for clf_name, clf_callback in clfs.items():
#     train_clf_untuned(name, clf_name, clf_callback, outlier_fraction)
# #######################################################################################################################
# name = 'Carnot_short'
#
# for clf_name, clf_callback in clfs.items():
#     train_clf_untuned(name, clf_name, clf_callback, outlier_fraction,)
# #######################################################################################################################
# name = 'Carnot_mid'
#
# for clf_name, clf_callback in clfs.items():
#     train_clf_untuned(name, clf_name, clf_callback, outlier_fraction)
# #######################################################################################################################
# name = 'Carnot_long'
#
# for clf_name, clf_callback in clfs.items():
#     train_clf_untuned(name, clf_name, clf_callback, outlier_fraction)
