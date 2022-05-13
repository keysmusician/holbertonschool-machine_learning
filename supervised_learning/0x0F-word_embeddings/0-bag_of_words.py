#!/usr/bin/env python3
""" Defines `bag_of_words`. """
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag-of-words embedding matrix.

    sentences: A list of sentences to analyze.
    vocab: A list of the vocabulary words to use for the analysis. If None, all
        words within `sentences` will be used.

    Returns: (embeddings, features):
        embeddings: A numpy.ndarray of shape (s, f) containing the embeddings.
            s: The number of sentences in sentences.
            f: The number of features analyzed.
        features: A list of the features used for embeddings.
    """
    vectorizer = CountVectorizer(vocabulary=vocab)
    embedding = vectorizer.fit_transform(sentences).toarray()
    features = vectorizer.get_feature_names_out()

    return (embedding, features)
