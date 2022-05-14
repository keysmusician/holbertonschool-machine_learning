#!/usr/bin/env python3
""" Defines `fasttext_model`. """
from gensim.models import FastText


def fasttext_model(
        sentences, size=100, min_count=5, negative=5, window=5, cbow=True,
        iterations=5, seed=0, workers=1):
    """
    Creates and trains a Gensim fastText model.

    sentences: A list of sentences to be trained on.
    size: The dimensionality of the embedding layer.
    min_count: The minimum number of occurrences of a word for use in training.
    negative: The size of negative sampling.
    window: The maximum distance between the current and predicted word within
        a sentence.
    cbow: A boolean to determine the training type; True is for CBOW; False is
        for Skip-gram.
    iterations: The number of iterations to train over.
    seed: The seed for the random number generator.
    workers: The number of worker threads to train the model.

    Returns: The trained model.
    """
    model = FastText(
        sentences=sentences,
        size=size,
        min_count=min_count,
        window=window,
        negative=negative,
        iter=iterations,
        seed=seed,
        workers=workers,
        sg=(not cbow)
        )

    return model
