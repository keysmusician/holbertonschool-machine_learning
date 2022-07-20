#!/usr/bin/env python3
""" Defines `from_numpy`. """
import pandas as pd
from string import ascii_uppercase


columns = [char for char in ascii_uppercase]

def from_numpy(array):
    """
    Creates a `pd.DataFrame` from a `np.ndarray`.

    array: The `np.ndarray` from which you should create the `pd.DataFrame`.

    Returns: The newly created `pd.DataFrame`.
    """
    return pd.DataFrame.from_records(array, columns=columns[:array.shape[1]])
