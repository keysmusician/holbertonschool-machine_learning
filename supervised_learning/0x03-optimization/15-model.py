#!/usr/bin/env python3
"""Trains a neural network with optimizations."""
import tensorflow.compat.v1 as tf
import numpy as np


# Task 14: Batch normalization layer
def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in tensorflow.

    prev: The activated output of the previous layer.
    n: The number of nodes in the layer to be created.
    activation: The activation function that should be used on the output of
        the layer.

    Returns: a tensor of the activated output for the layer
    """
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(n, kernel_initializer=init)
    new = layer(prev)

    mean, var = tf.nn.moments(new, axes=[0])
    beta = tf.Variable(tf.constant(0.0, shape=[n]), trainable=True,
                       name='beta')
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), trainable=True,
                        name='gamma')
    epsilon = 1e-8
    batch_normalization = tf.nn.batch_normalization(new, mean, var, beta,
                                                    gamma, epsilon)

    return activation(batch_normalization)


def forward_prop(input, layers, activations, epsilon):
    """
    Constructs the graph for forward propagation.

    prev: The input tensor to the graph.
    layers: A list containing the number of nodes in each layer of the network.
    activations: A list containing the activation functions used for each layer
        of the network.
    epsilon: A small number used to avoid division by zero.

    Return: The ouput tensor of the graph.
    """
    # init = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    # all layers get batch_normalization but the last one, that stays without
    # any activation or normalization
    previous = input
    for neuron_count, activation in zip(layers[:-1], activations[:-1]):
        previous = create_batch_norm_layer(previous, neuron_count, activation)

    return tf.layers.Dense(
        layers[-1], activations[-1])(previous)  # init?


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way.
    Args:
        X: A numpy.ndarray of shape (m, nx) to shuffle, where 'm' is the number
            of data points and 'nx' is the number of features in X.
        Y: A numpy.ndarray of shape (m, ny) to shuffle, where 'm' is the same
            number of data points as in X and 'ny' is the number of features in
            Y.
    Returns: The shuffled X and Y matrices.
    """
    if (X.shape[0] == Y.shape[0]):
        random_order = np.random.permutation(X.shape[0])
        return X[random_order], Y[random_order]


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """
    Builds, trains, and saves a neural network model in TensorFlow using Adam
    optimization, mini-batch gradient descent, learning rate decay, and batch
    normalization.

    Data_train: A tuple containing the training inputs and training labels,
        respectively.
    Data_valid: A tuple containing the validation inputs and validation labels,
        respectively.
    layers: A list containing the number of nodes in each layer of the network.
    activations: A list containing the activation functions used for each layer
        of the network.
    alpha: The learning rate.
    beta1: The weight for the first moment of Adam Optimization.
    beta2: The weight for the second moment of Adam Optimization.
    epsilon: A small number used to avoid division by zero.
    decay_rate: The decay rate for inverse time decay of the learning rate
        (the corresponding decay step should be 1).
    batch_size: The number of data points that should be in a mini-batch.
    epochs: The number of times the training should pass through the whole
        dataset.
    save_path: The path where the model should be saved to.

    Returns: The path where the model was saved.
    """
    # get X_train, Y_train, X_valid, and Y_valid from Data_train and Data_valid
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid
    x_last_dim = X_train.shape[-1]
    y_last_dim = Y_train.shape[-1]

    # initialize x, y and add them to collection
    x = tf.placeholder(name='x', dtype=tf.float32, shape=(None, x_last_dim))
    y = tf.placeholder(name='y', dtype=tf.float32, shape=(None, y_last_dim))
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    # initialize y_pred and add it to collection
    y_pred = forward_prop(x, layers, activations, epsilon)
    tf.add_to_collection('y_pred', y_pred)

    # intialize loss and add it to collection
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    tf.add_to_collection('loss', loss)

    # intialize accuracy and add it to collection
    y_argmax = tf.argmax(y, 1)
    y_pred_argmax = tf.argmax(y_pred, 1)
    equality = tf.math.equal(y_pred_argmax, y_argmax)
    accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))
    tf.add_to_collection('accuracy', accuracy)

    # intialize global_step variable
    # hint: not trainable
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # compute decay_steps
    decay_step = X_train.shape[0] / batch_size

    # create "alpha" the learning rate decay operation in tensorflow
    alpha = tf.train.inverse_time_decay(
        alpha, global_step, decay_step, decay_rate, True)

    # initizalize train_op and add it to collection
    # hint: don't forget to add global_step parameter in optimizer().minimize()
    adam = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    train_op = adam.minimize(loss, global_step)
    tf.add_to_collection('train_op', train_op)

    training_data = {x: X_train, y: Y_train}
    validation_data = {x: X_valid, y: Y_valid}

    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)

        for epoch in range(epochs + 1):
            # print training and validation cost and accuracy
            metrics = (loss, accuracy)
            t_cost, t_accuracy = session.run(metrics, training_data)
            v_cost, v_accuracy = session.run(metrics, validation_data)
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(t_cost))
            print("\tTraining Accuracy: {}".format(t_accuracy))
            print("\tValidation Cost: {}".format(v_cost))
            print("\tValidation Accuracy: {}".format(v_accuracy))

            # shuffle data
            X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)

            step = 0
            for batch_start in range(0, X_train.shape[0], batch_size):
                # get X_batch and Y_batch from X_train shuffled and Y_train
                # shuffled
                batch_end = batch_start + batch_size
                X_batch = X_shuffled[batch_start:batch_end]
                Y_batch = Y_shuffled[batch_start:batch_end]
                # run training operation
                mini_batch = {x: X_batch, y: Y_batch}
                session.run(train_op, mini_batch)

                if step and step % 100 == 0:
                    # print batch cost and accuracy
                    s_cost, s_accuracy = session.run(metrics, mini_batch)
                    print("\tStep {}:".format(step))
                    print("\t\tCost: {}".format(s_cost))
                    print("\t\tAccuracy: {}".format(s_accuracy))
                step += 1

        # print training and validation cost and accuracy again
        t_cost, t_accuracy = session.run(metrics, training_data)
        v_cost, v_accuracy = session.run(metrics, validation_data)
        print("After {} epochs:".format(epoch))
        print("\tTraining Cost: {}".format(t_cost))
        print("\tTraining Accuracy: {}".format(t_accuracy))
        print("\tValidation Cost: {}".format(v_cost))
        print("\tValidation Accuracy: {}".format(v_accuracy))

        # save and return the path to where the model was saved
        saver = tf.train.Saver()
        return saver.save(session, save_path)
