#!/usr/bin/env python3
""" Defines `uni_bleu`. """
import numpy as np


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a sentence.

    references: A list of reference translations.
    sentence: A list containing the model proposed sentence.

    Returns: The unigram BLEU score.
    """
    word_max_counts = {}
    for word in set(sentence):
        for reference in references:
            reference_word_count = reference.count(word)
            if reference_word_count > word_max_counts.get(word, 0):
                word_max_counts[word] = reference_word_count

    minimum_reference_length = min(len(ref) for ref in references)
    sentence_length = len(sentence)
    brevity_penalty = 1 if sentence_length > minimum_reference_length\
        else np.exp(1 - (minimum_reference_length / sentence_length))

    return sum(word_max_counts.values()) * brevity_penalty / sentence_length
