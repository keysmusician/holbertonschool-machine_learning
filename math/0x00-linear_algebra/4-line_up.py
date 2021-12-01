#!/usr/bin/env python3
"""Defines `add_arrays`."""
from itertools import zip_longest


def add_arrays(arr1, arr2):
    """Adds two arrays element-wise."""
    try:
        return [sum(tuple) for tuple in zip_longest(arr1, arr2)]
    except TypeError:
        return None
