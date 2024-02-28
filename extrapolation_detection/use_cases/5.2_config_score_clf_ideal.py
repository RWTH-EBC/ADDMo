from extrapolation_detection.detector.detector_tuners import Hyper_KNN, Hyper_KDE, Hyper_GP, Hyper_OCSVM, Hyper_IF
from extrapolation_detection.use_cases.score_clf import score_clf_ideal, evaluate_ideal, score_clf_ideal_2D

#######################################################################################################################
# TODO: specifiy parameters here
beta = 1
contour_detail_error = 100

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
    score_clf_ideal(name, clf_name)
    score_clf_ideal_2D(name, clf_name, contour_detail_error)
    evaluate_ideal(name, clf_name, beta=beta)
#######################################################################################################################
name = 'Carnot_uncertain_mid'

for clf_name, clf_callback in clfs.items():
    score_clf_ideal(name, clf_name)
    score_clf_ideal_2D(name, clf_name, contour_detail_error)
    evaluate_ideal(name, clf_name, beta=beta)
#######################################################################################################################
name = 'Carnot_uncertain_long'

for clf_name, clf_callback in clfs.items():
    score_clf_ideal(name, clf_name)
    score_clf_ideal_2D(name, clf_name, contour_detail_error)
    evaluate_ideal(name, clf_name, beta=beta)
#######################################################################################################################
name = 'Boptest_Pel_short'

for clf_name, clf_callback in clfs.items():
    score_clf_ideal(name, clf_name)
    evaluate_ideal(name, clf_name, beta=beta)
#######################################################################################################################
name = 'Boptest_Pel_mid'

for clf_name, clf_callback in clfs.items():
    score_clf_ideal(name, clf_name)
    evaluate_ideal(name, clf_name, beta=beta)
#######################################################################################################################
name = 'Boptest_Pel_long'

for clf_name, clf_callback in clfs.items():
    score_clf_ideal(name, clf_name)
    evaluate_ideal(name, clf_name, beta=beta)
#######################################################################################################################
name = 'Boptest_TAir_short'

for clf_name, clf_callback in clfs.items():
    score_clf_ideal(name, clf_name)
    evaluate_ideal(name, clf_name, beta=beta)
#######################################################################################################################
name = 'Boptest_TAir_mid'

for clf_name, clf_callback in clfs.items():
    score_clf_ideal(name, clf_name)
    evaluate_ideal(name, clf_name, beta=beta)
#######################################################################################################################
name = 'Boptest_TAir_long'

for clf_name, clf_callback in clfs.items():
    score_clf_ideal(name, clf_name)
    evaluate_ideal(name, clf_name, beta=beta)
#######################################################################################################################
name = 'Carnot_short'

for clf_name, clf_callback in clfs.items():
    score_clf_ideal(name, clf_name)
    score_clf_ideal_2D(name, clf_name, contour_detail_error)
    evaluate_ideal(name, clf_name, beta=beta)
#######################################################################################################################
name = 'Carnot_mid'

for clf_name, clf_callback in clfs.items():
    score_clf_ideal(name, clf_name)
    score_clf_ideal_2D(name, clf_name, contour_detail_error)
    evaluate_ideal(name, clf_name, beta=beta)
#######################################################################################################################
name = 'Carnot_long'

for clf_name, clf_callback in clfs.items():
    score_clf_ideal(name, clf_name)
    score_clf_ideal_2D(name, clf_name, contour_detail_error)
    evaluate_ideal(name, clf_name, beta=beta)
