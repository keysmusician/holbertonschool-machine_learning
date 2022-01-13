def forward_prop(prev, layers, activations, epsilon):
    #all layers get batch_normalization but the last one, that stays without any activation or normalization


def shuffle_data(X, Y):
    # fill the function


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    # get X_train, Y_train, X_valid, and Y_valid from Data_train and Data_valid

    # initialize x, y and add them to collection

    # initialize y_pred and add it to collection

    # intialize loss and add it to collection

    # intialize accuracy and add it to collection

    # intialize global_step variable
    # hint: not trainable

    # compute decay_steps

    # create "alpha" the learning rate decay operation in tensorflow

    # initizalize train_op and add it to collection
    # hint: don't forget to add global_step parameter in optimizer().minimize()

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for i in range(epochs):
            # print training and validation cost and accuracy

            # shuffle data

            for j in range(0, X_train.shape[0], batch_size):
                # get X_batch and Y_batch from X_train shuffled and Y_train shuffled

                # run training operation

                                # print batch cost and accuracy

        # print training and validation cost and accuracy again

        # save and return the path to where the model was saved
