"""The module contains auxiliary functions to perform validation."""

from __future__ import annotations

from typing import Any, Callable, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import typing as npt
from sklearn.ensemble import BaseEnsemble
from sklearn.linear_model._base import LinearModel
from sklearn.metrics import r2_score

from hw_test import utils, visualisation

Estimator = Union[LinearModel, BaseEnsemble]


def evaluate_models(
    estimators: list[tuple[str, Estimator]],
    data: pd.DataFrame,
    target_columns: list[str],
    **kwargs: Any,
) -> list[Estimator]:
    fitted_estimators = []

    for model_name, estimator in estimators:
        print("=" * 50)
        print(model_name)
        print("=" * 50)
        validate_model(estimator=estimator, data=data, target_columns=target_columns, **kwargs)
        fitted_estimators.append(estimator)
        print("\n")

    return fitted_estimators


def validate_model(
    estimator: Estimator,
    data: pd.DataFrame,
    target_columns: list[str],
    show_plots: bool = False,
) -> None:
    x_train, x_test, y_train, y_test = utils.make_split(data, target_columns=target_columns)
    estimator.fit(x_train, y_train)

    y_pred_train = estimator.predict(x_train)
    y_pred_test = estimator.predict(x_test)

    train_score = r2_score(y_train, y_pred_train)
    test_score = r2_score(y_test, y_pred_test)

    scores = create_bootstrap_metrics(y_test, estimator.predict(x_test), r2_score)
    ci = calculate_confidence_interval(scores)
    is_metric_inside = bool(ci[0] < test_score < ci[1])

    print(
        f"Train-score: {round(train_score, 3)}\n"
        f"Test-score: {round(test_score, 3)}\n"
        f"CI: ({round(ci[0], 3)}, {round(ci[1], 3)}), {is_metric_inside}"
    )

    if show_plots:
        _ = visualisation.plot_results(
            x_train,
            y_train,
            y_pred_train,
            title="Train sample",
            col_headers=data.drop(target_columns, axis=1).columns,
            row_headers=target_columns,
        )
        __ = visualisation.plot_results(
            x_test,
            y_test,
            y_pred_test,
            title="Test sample",
            col_headers=data.drop(target_columns, axis=1).columns,
            row_headers=target_columns,
        )
        plt.show()


def create_bootstrap_samples(
    data: npt.NDArray[np.float_],
    n_samples: int = 1000,
) -> npt.NDArray[np.int_]:
    """Creates bootstrap samples.

    Args:
        data: Numpy array containing data for bootstrapping.
        n_samples: How many samples should be returned.

    Returns:
        Numpy array containing chosen indices. Shape of the returned value will be
            ('n_samples', 'len(data)')

    """
    bootstrap_idx = np.random.randint(low=0, high=len(data), size=(n_samples, len(data)))
    return bootstrap_idx


def create_bootstrap_metrics(
    y_true: npt.NDArray[np.float_],
    y_pred: npt.NDArray[np.float_],
    metric: Callable[..., Union[np.float_, npt.NDArray[np.float_]]],
    additional_metric_params: dict[str, Any] | None = None,
    n_samples: int = 1000,
) -> Sequence[Union[np.floating[Any], npt.NDArray[np.float_]]]:
    """Calculation of bootstrap metrics.

    Args:
        y_true: Ground truth (correct) target values.
        y_pred: Estimated target values.
        metric: Function to calculate score.
        additional_metric_params: Parameters that will be passed to 'metric' as kwargs.
        n_samples: How many samples should be used for calculation.

    Returns:
        The sequence of scores. Shape of the returned value will be (`n_samples`,)

    """
    additional_metric_params = additional_metric_params if additional_metric_params else {}
    scores = []

    bootstrap_idx = create_bootstrap_samples(data=y_true, n_samples=n_samples)
    for idx in bootstrap_idx:
        y_true_bootstrap = y_true[idx]
        y_pred_bootstrap = y_pred[idx]

        score = metric(y_true=y_true_bootstrap, y_pred=y_pred_bootstrap, **additional_metric_params)
        scores.append(score)

    return scores


def calculate_confidence_interval(
    scores: Sequence[Union[np.floating[Any], npt.NDArray[np.float_]]], conf_interval: float = 0.95
) -> Tuple[np.floating[Any], np.floating[Any]]:
    """Calculates confidence interval.

    Args:
        scores: List of scores calculated based on bootstrapped metrics.
        conf_interval: Level of confidence in percents.

    Returns:
        Tuple containing lower and upper bounds of the confidence interval as floats.s

    """
    left_bound = np.percentile(scores, ((1 - conf_interval) / 2) * 100)
    right_bound = np.percentile(scores, (conf_interval + ((1 - conf_interval) / 2)) * 100)

    return left_bound, right_bound
