#!/usr/bin/env python3
"""
This module contains a simple function
"""

import tensorflow.compat.v1 as tf
create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_train_op = __import__('5-create_train_op').create_train_op


def train(X_train, Y_train, X_valid, Y_valid,
          layer_sizes, activations, alpha,
          iterations, save_path="/tmp/model.ckpt"):
    """builds, trains, and saves a neural network classifier"""
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_train_op(loss, alpha)

    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)

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
            sess.run(train_op, {x: X_train, y: Y_train})
        saver = tf.train.Saver()
        return saver.save(sess, save_path)
