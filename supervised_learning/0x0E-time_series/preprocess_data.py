#!/usr/bin/env python3
""" Preprocesses the data for "forecast_btc.py". """
import pandas as pd


# Load the raw data into a Pandas DataFrame
bitstamp_raw_dataframe = pd.read_csv(
    "./data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv")
coinbase_raw_dataframe = pd.read_csv(
    "./data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv")

# Take every on-the-hour entry, ignoring records with missing values
bitstamp_raw_dataframe['Timestamp'] = pd.to_datetime(
    bitstamp_raw_dataframe['Timestamp'], unit='s')
coinbase_raw_dataframe['Timestamp'] = pd.to_datetime(
    coinbase_raw_dataframe['Timestamp'], unit='s')

# Resample to the hour, fill missing data from the Coinbase dataset
# Drop unneccessary columns
bitstamp_hourly_dataframe = bitstamp_raw_dataframe\
    .set_index('Timestamp')\
    .drop(
        columns=["Open", "High", "Low", "Volume_(BTC)", "Volume_(Currency)"]
    )\
    .fillna(coinbase_raw_dataframe.set_index('Timestamp'))\
    .drop(range(1585513, 1585521))[8::60]\
    .dropna()

bitstamp_hourly_dataframe.to_csv('preprocessed_data.csv', index=False)
