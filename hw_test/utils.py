"""
The module contains various auxiliary functions which are not related to any particular section.
"""

import numpy as np
import pandas as pd
from numpy import typing as npt
from sklearn.model_selection import train_test_split


def make_split(
    data: pd.DataFrame,
    target_columns: list[str],
    train_size: float = 0.7,
    shuffle: bool = True,
    random_state: int = 1,
) -> tuple[
    npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_]
]:
    x_train, x_test, y_train, y_test = train_test_split(
        data.drop(target_columns, axis=1).values,
        data[target_columns].values,
        train_size=train_size,
        shuffle=shuffle,
        random_state=random_state,
    )
    return x_train, x_test, y_train, y_test
