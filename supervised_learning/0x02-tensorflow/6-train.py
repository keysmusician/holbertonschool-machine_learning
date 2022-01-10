#!/usr/bin/env python3
"""Defines `train`."""
import tensorflow.compat.v1 as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha,
          iterations, save_path="/tmp/model.ckpt"):
    """
    Builds, trains, and saves a neural network classifier.

    Args:
        X_train: A numpy.ndarray containing the training input data
        Y_train: A numpy.ndarray containing the training labels
        X_valid: A numpy.ndarray containing the validation input data
        Y_valid: A numpy.ndarray containing the validation labels
        layer_sizes: A list containing the number of nodes in each layer of the
            network
        activations: A list containing the activation functions for each layer
            of the network
        alpha: The learning rate
        iterations: The number of iterations to train over
        save_path: Designates where to save the model

    Returns: The path of where the model was saved.
    """
    input_feature_count = X_train.shape[1]
    class_count = Y_train.shape[1]
    x, y = create_placeholders(input_feature_count, class_count)
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train = create_train_op(loss, alpha)

    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    with sess.as_default():
        for i in range(iterations + 1):
            if i % 100 == 0 or i == 0 or i == iterations:
                costt = loss.eval(feed_dict={x: X_train, y: Y_train})
                acct = accuracy.eval(feed_dict={x: X_train, y: Y_train})
                costv = loss.eval(feed_dict={x: X_valid, y: Y_valid})
                accv = accuracy.eval(feed_dict={x: X_valid, y: Y_valid})
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(costt))
                print("\tTraining Accuracy: {}".format(acct))
                print("\tValidation Cost: {}".format(costv))
                print("\tValidation Accuracy: {}".format(accv))
            if i == iterations:
                break
            sess.run(train, {x: X_train, y: Y_train})
        saver = tf.train.Saver()
        return saver.save(sess, save_path)
