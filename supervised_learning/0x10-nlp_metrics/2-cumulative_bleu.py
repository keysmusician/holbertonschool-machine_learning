#!/usr/bin/env python3
""" Defines `cumulative_bleu`. """
import numpy as np
ngram_bleu = __import__('1-ngram_bleu').ngram_bleu


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU score for a sentence.

    references: A list of reference translations.
    sentence: A list containing the model proposed sentence.
    n: The size of the n-gram to use for evaluation.

    Returns: The cumulative n-gram BLEU score.
    """
    scores = [ngram_bleu(references, sentence, m) for m in range(1, n + 1)]

    return np.product(scores) ** (1 / n)
