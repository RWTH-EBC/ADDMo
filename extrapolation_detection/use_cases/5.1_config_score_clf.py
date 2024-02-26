from use_cases.score_clf import score_clf, score_clf_2D, evaluate

#######################################################################################################################
# TODO: specifiy parameters here
beta = 1
contour_detail_error = 100

clfs = [
    'kNN_TiV',
    # 'KDE_untuned_test',
    # 'GPR_untuned_test',
    # 'OCSVM_untuned_test',
    # # 'IF',
]
# #######################################################################################################################
# name = 'Carnot_uncertain_short'
#
# for clf_name in clfs:
#     score_clf(name, clf_name)
#     score_clf_2D(name, clf_name, contour_detail_error)
#     evaluate(name, clf_name, beta=beta)
#######################################################################################################################
# name = 'Carnot_uncertain_mid'
#
# for clf_name in clfs:
#     score_clf(name, clf_name) # scores available data points
#     score_clf_2D(name, clf_name, contour_detail_error) # scores data points for extrapolation boundary
#     evaluate(name, clf_name, beta=beta) # calculates f-score
# #######################################################################################################################
name = 'Carnot_uncertain_mid_32Neurons'

for clf_name in clfs:
    score_clf(name, clf_name) # scores available data points
    score_clf_2D(name, clf_name, contour_detail_error) # scores data points for extrapolation boundary
    evaluate(name, clf_name, beta=beta) # calculates f-score
# #######################################################################################################################
# name = 'Carnot_uncertain_long'
#
# for clf_name in clfs:
#     score_clf(name, clf_name)
#     score_clf_2D(name, clf_name, contour_detail_error)
#     evaluate(name, clf_name, beta=beta)
# #######################################################################################################################
# name = 'Boptest_Pel_short'
#
# for clf_name in clfs:
#     score_clf(name, clf_name)
#     evaluate(name, clf_name, beta=beta)
# #######################################################################################################################
# name = 'Boptest_Pel_mid'
#
# for clf_name in clfs:
#     score_clf(name, clf_name)
#     evaluate(name, clf_name, beta=beta)
# #######################################################################################################################
# name = 'Boptest_Pel_long'
#
# for clf_name in clfs:
#     score_clf(name, clf_name)
#     evaluate(name, clf_name, beta=beta)
# #######################################################################################################################
# name = 'Boptest_TAir_short'
#
# for clf_name in clfs:
#     score_clf(name, clf_name)
#     evaluate(name, clf_name, beta=beta)
# #######################################################################################################################
# name = 'Boptest_TAir_mid'
#
# for clf_name in clfs:
#     score_clf(name, clf_name)
#     evaluate(name, clf_name, beta=beta)
# #######################################################################################################################
# name = 'Boptest_TAir_long'
#
# for clf_name in clfs:
#     score_clf(name, clf_name)
#     evaluate(name, clf_name, beta=beta)
# ######################################################################################################################
# name = 'Carnot_short'
#
# for clf_name in clfs:
#     score_clf(name, clf_name)
#     score_clf_2D(name, clf_name, contour_detail_error)
#     evaluate(name, clf_name, beta=beta)
# #######################################################################################################################
# name = 'Carnot_mid'
#
# for clf_name in clfs:
#     score_clf(name, clf_name)
#     score_clf_2D(name, clf_name, contour_detail_error)
#     evaluate(name, clf_name, beta=beta)
# #######################################################################################################################
# name = 'Carnot_long'
#
# for clf_name in clfs:
#     score_clf(name, clf_name)
#     score_clf_2D(name, clf_name, contour_detail_error)
#     evaluate(name, clf_name, beta=beta)
