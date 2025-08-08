import inspect

from sklearn import metrics as sk_metrics

from addmo.s3_model_tuning.scoring.metrics import custom_metrics
from addmo.s3_model_tuning.scoring.metrics.abstract_metric import AbstractMetric


class MetricFactory:
    """
    Factory for creating metric instances.
    """

    @staticmethod
    def metric_factory(metric_name, metric_kwargs: dict = None):
        """Get the custom splitter instance dynamically or use scikit-learn splitters."""

        # If metric is custom
        if hasattr(custom_metrics, metric_name):
            custom_metric_class = getattr(custom_metrics, metric_name)
            return custom_metric_class(metric_kwargs)

        # If metric is from scikit-learn
        elif metric_name in sk_metrics.get_scorer_names():
            # Possible regression metrics are:
            # explained_variance, max_error, neg_mean_absolute_error, neg_mean_squared_error,
            # neg_root_mean_squared_error, neg_mean_squared_log_error,
            # neg_root_mean_squared_log_error, neg_median_absolute_error,
            # r2, neg_mean_poisson_deviance, neg_mean_gamma_deviance,
            # neg_mean_absolute_percentage_error, d2_absolute_error_score,
            # d2_pinball_score, d2_tweedie_score

            sk_metric = sk_metrics.get_scorer(metric_name)

            # Customize metric with additional kwargs if provided
            if metric_kwargs is not None:
                sk_metric = sk_metrics.make_scorer(
                    sk_metric._score_func, **metric_kwargs
                )
            return sk_metric

        # If metric is not found
        else:
            # get the names of all custom metrics for error message
            custom_metric_names = [
                name
                for name, obj in inspect.getmembers(custom_metrics)
                if inspect.isclass(obj)
                and issubclass(obj, AbstractMetric)
                and not inspect.isabstract(obj)
            ]

            raise ValueError(
                f"Unknown metric type: {metric_name}. "
                f"Available custom metrics are:"
                f" {', '.join(custom_metric_names)}. "
                f"You can also use any metric from scikit-learn, "
                f"like r2, neg_mean_absolute_error, d2_pinball_score, etc."
            )
