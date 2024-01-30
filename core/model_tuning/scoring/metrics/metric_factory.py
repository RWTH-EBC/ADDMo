from sklearn import metrics
from core.model_tuning.scoring.metrics.abstract_metric import AbstractMetric

class MetricFactory:
    """
    Factory for creating metric instances.
    """

    @staticmethod
    def metric_factory(metric_name: str) -> AbstractMetric:
        """
        Creates a metric instance based on the specified name.
        :param metric_name: Name of the metric to create.
        :return: Metric function or scorer object.
        """
        # If metric is from scikit-learn
        if metric_name in metrics.get_scorer_names():
            # Possible metrics:
            # explained_variance, max_error, neg_mean_absolute_error, neg_mean_squared_error,
            # neg_root_mean_squared_error, neg_mean_squared_log_error,
            # neg_root_mean_squared_log_error, neg_median_absolute_error,
            # r2, neg_mean_poisson_deviance, neg_mean_gamma_deviance,
            # neg_mean_absolute_percentage_error, d2_absolute_error_score,
            # d2_pinball_score, d2_tweedie_score

            metric = metrics.get_scorer(metric_name)
        # If metric is custom
        else:
            # This is where you can integrate custom metrics
            # For example, metric = custom_metric_factory(metric_name)
            raise NotImplementedError(f"Custom metric '{metric_name}' is not implemented yet.")

        return metric
