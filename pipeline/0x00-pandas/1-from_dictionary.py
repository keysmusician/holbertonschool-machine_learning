#!/usr/bin/env python3
""" Creates a `pd.DataFrame` from a dictionary. """
import pandas as pd


df = pd.DataFrame.from_dict({
    'First': [0.0, 0.5, 1.0, 1.5],
    'Second': ['one', 'two', 'three', 'four']
})
df.index = ('A', 'B', 'C', 'D')
