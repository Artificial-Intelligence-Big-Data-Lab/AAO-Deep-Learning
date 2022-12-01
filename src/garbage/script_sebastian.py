import os
import sys
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
from datetime import timedelta
import matplotlib.pyplot as plt
from pyts.image import GADF
from classes.Market import Market
size = 20

def group(df, freq):
	grouped = df.drop(['date', 'time'], axis=1).groupby(pd.Grouper(key='date_time', freq=freq), sort=True).agg({
		'open': 'first',
		'close': 'last',
		'high': 'max',
		'low': 'min',
		'up': 'sum',
		'down': 'sum',
		'volume': 'sum'
	})

	grouped['delta'] = grouped['close'] - grouped['open']
	
	return grouped.tail(size) # Last 40 rows

def calculate_gaf(df):
	df = df.dropna().head(20)
	print(df.shape)
	input()
	#df = df['delta'] # Uses 'delta' only
	
	# Scaling
	min_ = np.amin(df)
	max_ = np.amax(df)
	scaled_serie = (2 * df - max_ - min_) / (max_ - min_)
	
	# Floating point inaccuracy
	scaled_serie = np.where(scaled_serie >= 1., 1., scaled_serie)
	scaled_serie = np.where(scaled_serie <= -1., -1., scaled_serie)

    # Polar coords
	phi = np.arccos(scaled_serie)
	print(phi.shape)
	input()
	gadf = GADF(size)

	return gadf.fit_transform(phi.transpose())


# MAIN FLOW
df = pd.read_csv('../datasets/sp500_cet.csv').set_index('id')


df = df.dropna().reset_index()
df['date_time'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y-%m-%d %H:%M:%S')

# Generates and saves subset dataset
mask = (df['date_time'] >= '2013-01-15') & (df['date_time'] < '2013-02-13') # Original code uses <=
df = df.loc[mask]
df.to_csv('sp500_subset.csv')

# Grouping
one_h = group(df, freq='1h')
four_h = group(df, freq='4h')
eight_h = group(df, freq='8h')
one_d = group(df, freq='1d')

# Single GAFs
first = calculate_gaf(one_h)
second = calculate_gaf(four_h)
third = calculate_gaf(eight_h)
fourth = calculate_gaf(one_d)

#matplotlib.image.imsave('image.png', first, cmap='rainbow')