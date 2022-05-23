#!/usr/bin/env python3
""" Defines `positional_encoding`. """
import numpy as np


def positional_encoding(max_seq_len, dm):
    """ Calculates the positional encoding for a transformer. """
    return np.zeros((max_seq_len, dm))
