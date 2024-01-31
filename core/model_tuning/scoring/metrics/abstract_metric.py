"""When creating custom metrics, comply to scikit-learn API.
Here a metric is what scikit-learn calls a scorer.
Use the following to turn a plain metric into a scorer:
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html.

Since some metrics, like AIC, also need the estimator to be calculated, I decided to call a metric
an object that takes the estimator and produces a score. Equivalent to scikit-learn's scorer.
"""

class AbstractMetric:
    pass
