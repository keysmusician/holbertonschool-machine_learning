#!/usr/bin/env python3
"""Trains an artificial neural network."""
import tensorflow.compat.v1 as tf


def forward_prop(prev, layers, activations, epsilon):
    #all layers get batch_normalization but the last one, that stays without any activation or normalization

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

def shuffle_data(X, Y):
    """Randomly shuffles data."""
    random_order = np.random.permutation(X.shape[0])
    return X[random_order], Y[random_order]


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """Trains a neural network."""
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    # initialize x, y and add them to collection
    x = tf.placeholder(name='x', dtype=tf.float32, shape=(None, len(X_train)))
    y = tf.placeholder(name='y', dtype=tf.float32, shape=(None, len(Y_train)))
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    # initialize y_pred and add it to collection
    y_pred = forward_prop(x, layers, activations, epsilon)
    tf.add_to_collection('y_pred', y_pred)
    # intialize loss and add it to collection
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    tf.add_to_collection('loss', loss)
    # intialize accuracy and add it to collection
    y1 = tf.argmax(y, 1)
    yp1 = tf.argmax(y_pred, 1)
    equality = tf.math.equal(yp1, y1)
    accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))
    tf.add_to_collection('accuracy', accuracy)

    # intialize global_step variable
    # hint: not trainable
    global_step = tf.Variable(
        tf.constant(0), name='global_step', trainable=False)

    # compute decay_steps

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

        for epoch in range(epochs):
            # Print metrics
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
                # Slice a mini-batch
                X_batch = X_shuffled[batch_start:batch_end]
                Y_batch = Y_shuffled[batch_start:batch_end]

                # Train on the current mini-batch
                mini_batch = {x: X_batch, y: Y_batch}
                session.run(train_op, mini_batch)

                # print batch cost and accuracy
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

        # save and return the path to where the model was saved
        saver = tf.Saver()
        return saver.save(session, save_path)
