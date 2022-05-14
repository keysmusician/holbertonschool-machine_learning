#!/usr/bin/env python3
""" Defines `gensim_to_keras`. """


def gensim_to_keras(model):
    """
    Converts a Gensim word2vec model to a keras Embedding layer.

    model: A trained Gensim word2vec model.

    Returns: The trainable Keras Embedding.
    """
    return model.wv.get_keras_embedding()
