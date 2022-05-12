#!/usr/bin/env python3
""" Defines `ngram_bleu`. """
import numpy as np


def n_gramify(word_list, n):
    """
    Converts a list of words to a generator of n-grams.

    Yields: The next n-gram in the sequence.
    """
    start = 0
    while n <= len(word_list):
        yield ' '.join(word_list[start:n])
        start += 1
        n += 1


def ngram_bleu(references, sentence, n):
    """
    Calculates the n-gram BLEU score for a sentence.

    references: A list of reference translations.
    sentence: A list containing the model proposed sentence.
    n: The size of the n-gram to use for evaluation.

    Returns: The n-gram BLEU score.
    """
    minimum_reference_length = min(len(reference) for reference in references)
    sentence_length = len(sentence)
    brevity_penalty = 1 if sentence_length > minimum_reference_length\
        else np.exp(1 - (minimum_reference_length / sentence_length))

    sentence = list(n_gramify(sentence, n))
    references = [list(n_gramify(reference, n)) for reference in references]
    n_gram_max_counts = {}
    for n_gram in sentence:
        for reference in references:
            reference_n_gram_count = reference.count(n_gram)
            if reference_n_gram_count > n_gram_max_counts.get(n_gram, 0):
                n_gram_max_counts[n_gram] = reference_n_gram_count

    return sum(n_gram_max_counts.values()) * brevity_penalty\
        / len(sentence)
