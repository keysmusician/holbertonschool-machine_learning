#!/usr/bin/env python3
"""Defines `train_mini_batch`."""
import tensorflow.compat.v1 as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    Trains a loaded neural network model using mini-batch gradient descent.

    Args:
        X_train: A numpy.ndarray of shape (m, 784) containing the training
            data, where 'm' is the number of data points and 784 is the number
            of input features.
        Y_train: A one-hot numpy.ndarray of shape (m, 10) containing the
            training labels, where 10 is the number of classes the model should
            classify.
        X_valid: A numpy.ndarray of shape (m, 784) containing the validation
            data.
        Y_valid: A one-hot numpy.ndarray of shape (m, 10) containing the
            validation labels.
        batch_size: The number of data points in a batch
        epochs: The number of times the training should pass through the whole
            dataset.
        load_path: The path from which to load the model.
        save_path: The path to where the model should be saved after training.

        Returns: The path where the model was saved
    """
    with tf.Session() as session:
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(session, load_path)

        # Restore variables
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        training_data = {x: X_train, y: Y_train}
        validation_data = {x: X_valid, y: Y_valid}

        for epoch in range(epochs + 1):

            # Calculate and print metrics
            metrics = (loss, accuracy)
            t_cost, t_accuracy = session.run(metrics, training_data)
            v_cost, v_accuracy = session.run(metrics, validation_data)
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(t_cost))
            print("\tTraining Accuracy: {}".format(t_accuracy))
            print("\tValidation Cost: {}".format(v_cost))
            print("\tValidation Accuracy: {}".format(v_accuracy))

            # Shuffle data
            X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)

            batch_start, batch_end, step = 0, batch_size, 1
            while epoch < epochs:
                X_batch = X_shuffled[batch_start:batch_end]
                Y_batch = Y_shuffled[batch_start:batch_end]

                mini_batch = {x: X_batch, y: Y_batch}
                session.run(train_op, mini_batch)

                # Print metrics
                if step % 100 == 0:
                    s_cost, s_accuracy = session.run(metrics, mini_batch)
                    print("\tStep {}:".format(step))
                    print("\t\tCost: {}:".format(s_cost))
                    print("\t\tAccuracy: {}:".format(s_accuracy))

                if batch_end >= len(X_shuffled):
                    break
                batch_start += batch_size
                batch_end += batch_size
                step += 1

        return saver.save(session, save_path)
