#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = df.drop(columns='Weighted_Price')\
    .rename(columns={'Timestamp': 'Date'})
df['Date'] = pd.to_datetime(df['Date'], unit='s')
df = df.set_index('Date')
df['Close'] = df['Close'].fillna(method='pad')
df = df.fillna({
    'High': df['Close'],
    'Low': df['Close'],
    'Open': df['Close'],
    'Volume_(BTC)': 0,
    'Volume_(Currency)': 0
})

df.loc[pd.to_datetime('1-1-2017'):]\
    .resample('1440 min')\
    .aggregate({
        'High': 'max',
        'Low': 'min',
        'Open': 'mean',
        'Close': 'mean',
        'Volume_(BTC)': 'sum',
        'Volume_(Currency)': 'sum',

    })\
    .plot()

plt.show(block=True)

