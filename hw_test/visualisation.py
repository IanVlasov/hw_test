"""The module contains auxiliary functions to create graphs."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy import typing as npt


def add_headers(
    fig: plt.Figure,
    *,
    row_headers: list[str] | None = None,
    col_headers: list[str] | None = None,
    row_pad: int = 1,
    col_pad: int = 5,
    rotate_row_headers: bool = True,
    **text_kwargs: Any,
) -> None:
    """The function adds headers to the subplots.

    Args:
        fig: Figure where headers should be added.
        row_headers: List containing headers for each row. Defaults to None.
        col_headers: List containing headers for each column. Defaults to None.
        row_pad: Integer value to adjust row padding.
        col_pad:Integer value to adjust column padding.
        rotate_row_headers: True if row headers should be rotated on 90 degrees, False otherwise.
            Defaults to False.
        **text_kwargs:

    """
    # Based on https://stackoverflow.com/a/25814386

    axes = fig.get_axes()

    for ax in axes:
        sbs = ax.get_subplotspec()

        # Putting headers on cols
        if (col_headers is not None) and sbs.is_first_row():
            ax.annotate(
                col_headers[sbs.colspan.start],
                xy=(0.5, 1),
                xytext=(0, col_pad),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="center",
                va="baseline",
                **text_kwargs,
            )

        # Putting headers on rows
        if (row_headers is not None) and sbs.is_first_col():
            ax.annotate(
                row_headers[sbs.rowspan.start],
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - row_pad, 0),
                xycoords=ax.yaxis.label,
                textcoords="offset points",
                ha="right",
                va="center",
                rotation=rotate_row_headers * 90,
                **text_kwargs,
            )


def plot_results(
    data: npt.NDArray[np.float_],
    y_true: npt.NDArray[np.float_],
    y_pred: npt.NDArray[np.float_],
    row_headers: list[str],
    col_headers: list[str],
    title: str,
) -> plt.Figure:
    """Plots the results of regression on subplots. Each graph contains one of the feature against
    one of the objective function.

    Args:
        data: Numpy array containing samples to plot.
        y_true: Ground truth (correct) target values.
        y_pred: Estimated target values.
        row_headers: List containing headers for each row.
        col_headers: List containing headers for each column.
        title: The main title for the graph.

    Returns:

    """
    nrows = y_true.shape[1]
    ncols = data.shape[1]

    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 10), sharey="row")
    fig.suptitle(title, fontsize=15)

    add_headers(
        fig,
        col_headers=col_headers,
        row_headers=row_headers,
        fontfamily="monospace",
        fontweight="bold",
        fontsize="large",
    )

    for i in range(nrows):
        for j in range(ncols):
            axes[i, j].scatter(data[:, j], y_true[:, i])
            axes[i, j].scatter(data[:, j], y_pred[:, i])
            axes[i, j].grid()
            axes[i, j].legend(["True", "Predicted"])

    return fig
