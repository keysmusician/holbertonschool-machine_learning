#!/usr/bin/env python3
""" Defines `train_transformer` """
import tensorflow.compat.v2 as tf
Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """
    Creates and trains a transformer model for machine translation of
    Portuguese to English.

    N: The number of blocks in the encoder and decoder.
    dm: The dimensionality of the model.
    h: The number of heads.
    hidden: The number of hidden units in the fully connected layers.
    max_len: The maximum number of tokens per sequence.
    batch_size: The batch size for training.
    epochs: The number of epochs to train for.

    Returns: The trained model.
    """
    # Create the dataset
    dataset = Dataset(batch_size, max_len)

    # Instantiate a Transformer model
    transformer = Transformer(
        N,
        dm,
        h,
        hidden,
        dataset.tokenizer_pt.vocab_size + 2,
        dataset.tokenizer_en.vocab_size + 2,
        max_len,
        max_len,
    )

    # Custom optimizations
    class TransformerLRS(tf.keras.optimizers.schedules.LearningRateSchedule):
        """ Custom learning rate schedule """

        def __init__(self, warmup_steps=4000):
            """ Initializes the TransformerLRS """
            self.warmup_steps = warmup_steps

        def __call__(self, step):
            """ Calculates the learning rate at `step`. """
            learning_rate = (
                dm ** -0.5 *
                tf.math.minimum(step ** -0.5, step * self.warmup_steps ** -1.5)
            )

            return learning_rate

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=TransformerLRS(),
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9,
    )

    # Define the loss function
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    def loss_function(real, pred):
        """ Calculates the loss of a prediction. """
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    def accuracy_function(real, pred):
        """ Calculates the accuracy of the model. """
        accuracies = tf.equal(real, tf.argmax(pred, axis=2))

        mask = tf.math.logical_not(tf.math.equal(real, 0))
        accuracies = tf.math.logical_and(mask, accuracies)

        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    # Custom training procedure
    def train_step(inputs, targets):
        """ Trains the model on a single batch. """
        tar_inp = targets[:, :-1]
        tar_real = targets[:, 1:]

        encoder_mask, look_ahead_mask, decoder_mask = create_masks(
            inputs, tar_inp)

        with tf.GradientTape() as tape:
            predictions = transformer(
                inputs,
                tar_inp,
                True,
                encoder_mask,
                look_ahead_mask,
                decoder_mask,
            )
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, transformer.trainable_variables))
        train_loss(loss)
        train_accuracy(accuracy_function(tar_real, predictions))

    # Train
    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()

        for (batch_number, (inputs, targets)) in enumerate(dataset.data_train):
            train_step(inputs, targets)

            if batch_number % 50 == 0:
                print(
                    'Epoch {}, batch {}: loss {} accuracy {}'.format(
                        epoch, batch_number, train_loss.result(),
                        train_accuracy.result()
                    )
                )

        print(
            'Epoch {}: loss {} accuracy {}'.format(
                epoch, train_loss.result(), train_accuracy.result())
        )

    return transformer
