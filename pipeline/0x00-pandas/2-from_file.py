#!/usr/bin/env python3
""" Defines `from_file`. """
import pandas as pd


def from_file(filename, delimiter):
    """
    Loads data from a CSV file as a `pd.DataFrame`.

    filename: The CSV file to load from.
    delimiter: The column separator.

    Returns: The loaded `pd.DataFrame`.
    """
    return pd.read_csv(filename, delimiter=delimiter)
