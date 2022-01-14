#!/usr/bin/env python3
"""Trains a neural network with optimizations."""
import tensorflow.compat.v1 as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def calculate_accuracy(y, y_pred):
    """ calcs accuracy of prediction """
    y1 = tf.argmax(y, 1)
    yp1 = tf.argmax(y_pred, 1)
    equality = tf.math.equal(yp1, y1)
    acc = tf.math.reduce_mean(tf.cast(equality, tf.float32))
    return acc


def create_placeholders(nx, classes):
    """ return two nn placeholders """
    x = tf.placeholder("float", (None, nx), "x")
    y = tf.placeholder("float", (None, classes), "y")
    return x, y


def create_layer(prev, n, activation):
    """ create layer for nn, prev output, n nodes, activation f'n """
    kernel_initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(n, activation=activation,
                            kernel_initializer=kernel_initializer,
                            name="layer")
    return layer(prev)


def forward_prop(x, layer_sizes=[], activations=[]):
    """ creeates forward prop graph for nn, x input data, layers sizes nodes
        activations is f'ns for each layer. Returns prediction of net as tensor
    """
    prev = x
    for i in range(len(layer_sizes) - 1):
        prev = create_batch_norm_layer(prev, layer_sizes[i], activations[i])
    prev = create_layer(prev, layer_sizes[len(layer_sizes) - 1],
                        activations[len(layer_sizes) - 1])
    return prev


def calculate_loss(y, y_pred):
    """calcs softmax entropy loss """
    return tf.losses.softmax_cross_entropy(y, y_pred)


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """ creates Adam train operation """
    return tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon).minimize(loss)


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """ creates learning rate decay operation. starts at alpha, inv time decays
    """
    rate = tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                       decay_rate, staircase=True)
    return rate


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


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """ builds trains saves nn
        Data_train: tuple with training inputs and labels respectively
        Data valid: tuple with validate inputs and labels respectively
        layers: list containing number of nodes in each layer
        activation: list containg activation function for each layer
        alpha: learning rate
        beta1: weight for first moment of adam opt
        beta2: weight for second moment of adam opt
        epsilon: small number to avoid division by zero
        decay_rate: decay rate for inverse time decay of learning rate
        batch_size: number of data points in mini batch
        epochs: number of times training should go through whole set
        save_path: where to save to
    """
    with tf.Session() as sess:
        nx = Data_train[0].shape[1]
        classes = Data_train[1].shape[1]
        x, y = create_placeholders(nx, classes)
        y_pred = forward_prop(x, layers, activations)
        loss = calculate_loss(y, y_pred)
        accuracy = calculate_accuracy(y, y_pred)
        global_step = tf.Variable(0, trainable=False)
        # calculate decay step
        decay_step = 1
        decalpha = tf.train.inverse_time_decay(
            alpha, global_step, decay_step, decay_rate, True)
        train_op = create_Adam_op(loss, decalpha, beta1, beta2, epsilon)

        tf.add_to_collection("y", y)
        tf.add_to_collection("x", x)
        tf.add_to_collection("y_pred", y_pred)
        tf.add_to_collection("loss", loss)
        tf.add_to_collection("accuracy", accuracy)
        tf.add_to_collection("train_op", train_op)

        saver = tf.train.Saver()
        feed_dict_train = {x: Data_train[0], y: Data_train[1]}
        feed_dict_valid = {x: Data_valid[0], y: Data_valid[1]}
        sess.run(tf.global_variables_initializer())

        for i in range(epochs + 1):
            lt = sess.run(loss, feed_dict_train)
            at = sess.run(accuracy, feed_dict_train)
            lv = sess.run(loss, feed_dict_valid)
            av = sess.run(accuracy, feed_dict_valid)

            print("after {} epochs:".format(i))
            print("\tTraining Cost: {}".format(lt))
            print("\tTraining Accuracy: {}".format(at))
            print("\tValidation Cost: {}".format(lv))
            print("\tValidation Accuracy: {}".format(av))

            if i == epochs:
                continue
            shuf_x, shuf_y = shuffle_data(Data_train[0], Data_train[1])

            for j in range(0, int(Data_train[0].shape[0] / batch_size + 1)):
                lower = j * batch_size
                upper = lower + batch_size
                if j == int(Data_train[0].shape[0] / batch_size + 1):
                    upper = lower + Data_train[0].shape[0] % batch_size
                mini_batch = {x: shuf_x[lower: upper],
                              y: shuf_y[lower: upper]}
                sess.run(train_op, mini_batch)
                if j != 0 and (j + 1) % 100 == 0:
                    ltm = sess.run(loss, feed_dict=mini_batch)
                    atm = sess.run(accuracy, feed_dict=mini_batch)
                    print("\tStep {}:".format(j + 1))
                    print("\t\tCost: {}".format(ltm))
                    print("\t\tAccuracy: {}".format(atm))
            sess.run(tf.assign(global_step, i + 1))
        saved = saver.save(sess, save_path)
        return saved
