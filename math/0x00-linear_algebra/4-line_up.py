#!/usr/bin/env python3
"""Defines `add_arrays`."""


def add_arrays(arr1, arr2):
    """Adds two arrays element-wise."""
    if len(arr1) != len(arr2):
        return None
    return [sum(tuple) for tuple in zip(arr1, arr2)]
