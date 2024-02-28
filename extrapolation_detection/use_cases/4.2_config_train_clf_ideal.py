from extrapolation_detection.detector.detector_tuners import Hyper_KNN, Hyper_KDE, Hyper_GP, Hyper_OCSVM, Hyper_IF
from extrapolation_detection.use_cases.train_clf import train_clf_ideal

#######################################################################################################################
# TODO: specifiy parameters here
beta = 1
outlier_fraction = 0.05

clfs = {
    'kNN': Hyper_KNN,
    'KDE': Hyper_KDE,
    'GPR': Hyper_GP,
    'OCSVM': Hyper_OCSVM,
    'IF': Hyper_IF,
}
#######################################################################################################################
name = 'Carnot_uncertain_short'

for clf_name, clf_callback in clfs.items():
    train_clf_ideal(name, clf_name, clf_callback, outlier_fraction, beta=beta)
#######################################################################################################################
name = 'Carnot_uncertain_mid'

for clf_name, clf_callback in clfs.items():
    train_clf_ideal(name, clf_name, clf_callback, outlier_fraction, beta=beta)
#######################################################################################################################
name = 'Carnot_uncertain_long'

for clf_name, clf_callback in clfs.items():
    train_clf_ideal(name, clf_name, clf_callback, outlier_fraction, beta=beta)
#######################################################################################################################
name = 'Boptest_Pel_short'

for clf_name, clf_callback in clfs.items():
    train_clf_ideal(name, clf_name, clf_callback, outlier_fraction, beta=beta)
#######################################################################################################################
name = 'Boptest_Pel_mid'

for clf_name, clf_callback in clfs.items():
    train_clf_ideal(name, clf_name, clf_callback, outlier_fraction, beta=beta)
#######################################################################################################################
name = 'Boptest_Pel_long'

for clf_name, clf_callback in clfs.items():
    train_clf_ideal(name, clf_name, clf_callback, outlier_fraction, beta=beta)
#######################################################################################################################
name = 'Boptest_TAir_short'

for clf_name, clf_callback in clfs.items():
    train_clf_ideal(name, clf_name, clf_callback, outlier_fraction, beta=beta)
#######################################################################################################################
name = 'Boptest_TAir_mid'

for clf_name, clf_callback in clfs.items():
    train_clf_ideal(name, clf_name, clf_callback, outlier_fraction, beta=beta)
#######################################################################################################################
name = 'Boptest_TAir_long'

for clf_name, clf_callback in clfs.items():
    train_clf_ideal(name, clf_name, clf_callback, outlier_fraction, beta=beta)
#######################################################################################################################
name = 'Carnot_short'

for clf_name, clf_callback in clfs.items():
    train_clf_ideal(name, clf_name, clf_callback, outlier_fraction, beta=beta)
#######################################################################################################################
name = 'Carnot_mid'

for clf_name, clf_callback in clfs.items():
    train_clf_ideal(name, clf_name, clf_callback, outlier_fraction, beta=beta)
#######################################################################################################################
name = 'Carnot_long'

for clf_name, clf_callback in clfs.items():
    train_clf_ideal(name, clf_name, clf_callback, outlier_fraction, beta=beta)
