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

    for item in (x, y, y_pred, loss, accuracy, train):
        tf.add_to_collection("model", item)

    training_data = {x: X_train, y: Y_train}
    validation_data = {x: X_valid, y: Y_valid}
    saver = tf.train.Saver()
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    metrics = (accuracy, loss)
    for iteration in range(iterations + 1):
        training_accuracy, training_cost = session.run(metrics, training_data)
        validation_accuracy, validation_cost = \
            session.run(metrics, validation_data)

        if iteration % 100 == 0:
            print(f'After {iteration} iterations:')
            print(f'\tTraining Cost: {training_cost}')
            print(f'\tTraining Accuracy: {training_accuracy}')
            print(f'\tValidation Cost: {validation_cost}')
            print(f'\tValidation Accuracy: {validation_accuracy}')

        session.run(train, training_data)

    print(f'After {iteration} iterations:')
    print(f'\tTraining Cost: {training_cost}')
    print(f'\tTraining Accuracy: {training_accuracy}')
    print(f'\tValidation Cost: {validation_cost}')
    print(f'\tValidation Accuracy: {validation_accuracy}')

    path = saver.save(session, save_path)
    session.close()

    return path
