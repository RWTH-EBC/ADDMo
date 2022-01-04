AvailablePredictors = ["svr_bayesian_predictor", "rf_predictor", "ann_bayesian_predictor",
                       "gradientboost_bayesian", "lasso_bayesian", "svr_grid_search_predictor",
                       "gradientboost_gridsearch", "lasso_grid_search_predictor",
                       "ann_grid_search_predictor"]

AvailablePredictors1 = ["svr_bayesian_predictor", "rf_predictor", "ann_bayesian_predictor",
                        "gradientboost_bayesian", "lasso_bayesian", "svr_grid_search_predictor",
                        "gradientboost_gridsearch", "lasso_grid_search_predictor",
                        "ann_grid_search_predictor"]


def func():
    x1 = 1
    x2 = 2
    x3 = 3
    return x1, x2, x3

a = func()
x1, x2, x3 = a
print("the function returns", x1, x2, x3)
