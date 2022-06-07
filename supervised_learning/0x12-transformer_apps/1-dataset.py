#!/usr/bin/env python3
""" Defines `Dataset`. """
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """ Loads and preps a dataset for machine translation. """

    def __init__(self):
        """ Initializes a Dataset for NLP. """
        dataset = 'ted_hrlr_translate/pt_to_en'
        self.data_train = tfds.load(dataset, split='train', as_supervised=True)
        self.data_valid = tfds.load(
            dataset, split='validation', as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for this dataset.

        The maximum vocabulary size is 2^15.

        data: A `tf.data.Dataset` whose examples are formatted as a tuple
            (pt, en):
            pt: The `tf.Tensor` containing the Portuguese sentence.
            en: The `tf.Tensor` containing the corresponding English sentence.

        Returns: (tokenizer_pt, tokenizer_en)
            tokenizer_pt: The Portuguese tokenizer.
            tokenizer_en: The English tokenizer.
        """
        MAX_VOCAB_SIZE = 2 ** 15
        Tokenizer = tfds.deprecated.text.SubwordTextEncoder
        tokenizer_pt = Tokenizer.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=MAX_VOCAB_SIZE)
        tokenizer_en = Tokenizer.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=MAX_VOCAB_SIZE)

        return (tokenizer_pt, tokenizer_en)

    def encode(self, pt, en):
        """
        Encodes a translation into token IDs/indexes.

        The tokenized sentences include the start and end of sentence token
        IDs. The start token is indexed as `vocab_size`. The end token is
        indexed as `vocab_size + 1`.

        pt: The tf.Tensor containing the Portuguese sentence.
        en: The tf.Tensor containing the corresponding English sentence.

        Returns: (pt_tokens, en_tokens)
            pt_tokens: A `np.ndarray` containing the Portuguese token IDs.
            en_tokens: A `np.ndarray` containing the English token IDs.
        """
        token_IDs = []
        for language_tensor, tokenizer in [
            (pt, self.tokenizer_pt),
            (en, self.tokenizer_en)
        ]:
            SENTENCE_START = tokenizer.vocab_size
            SENTENCE_END = SENTENCE_START + 1
            token_IDs.append(
                [SENTENCE_START] +
                tokenizer.encode(language_tensor.numpy()) +
                [SENTENCE_END]
            )

        return tuple(token_IDs)
