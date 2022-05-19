#!/usr/bin/env python3
"""
A model which uses the past 24 hours of Bitcoin data to predict the value of
BTC at the close of the following hour.
"""
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf


# Load the data
bitstamp_hourly_dataframe = pd.read_csv('preprocessed_data.csv')

# Slice the data into training, validation, and test sets:
n = len(bitstamp_hourly_dataframe)
training_data = bitstamp_hourly_dataframe[0:int(n * 0.7)]
validation_data = bitstamp_hourly_dataframe[int(n * 0.7):int(n * 0.9)]
test_data = bitstamp_hourly_dataframe[int(n * 0.9):]

# Normalize features
train_mean = training_data.mean()
train_std = training_data.std()

training_data = (training_data - train_mean) / train_std
validation_data = (validation_data - train_mean) / train_std
test_data = (test_data - train_mean) / train_std

# Convert to TF Datasets
training_ds = tf.keras.preprocessing.timeseries_dataset_from_array(
    data=training_data[:-25],
    targets=training_data['Close'][25:],
    sequence_length=25,
    sequence_stride=1,
    shuffle=False,
    batch_size=32
)

validation_ds = tf.keras.preprocessing.timeseries_dataset_from_array(
    data=validation_data[:-25],
    targets=validation_data['Close'][25:],
    sequence_length=25,
    sequence_stride=1,
    shuffle=False,
    batch_size=32
)

# Design the model
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.Dense(units=1)
])

model.compile(loss=tf.losses.MeanSquaredError(),
              optimizer=tf.optimizers.Adam(),
              metrics=[tf.metrics.MeanAbsoluteError()])

# Training procedure
MAX_EPOCHS = 20
PATIENCE = 2

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=PATIENCE,
                                                  mode='min')

history = model.fit(training_ds, epochs=MAX_EPOCHS,
                    validation_data=training_ds,
                    callbacks=[early_stopping])

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.show()
