import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.utils import shuffle

from datetime import datetime
from nolearn.lasagne import BatchIterator
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet as nn
import theano
from sklearn.grid_search import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
import sys
from time import time
import matplotlib.pyplot as plt

class AdjustVariable(object):
	def __init__(self, name, start=0.03, stop=0.001):
		self.name = name
		self.start = start
		self.stop = stop
		self.ls = None

	def __call__(self, nn, train_history):
		if self.ls is None:
			self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

		epoch = train_history[-1]['epoch']
		new_value = float32(self.ls[epoch-1])
		getattr(nn, self.name).set_value(new_value)


def float32(k):
	return np.cast['float32'](k)

def load(test=False, cols=None):
	fname = "test.csv" if test else "train.csv"
	df = pd.read_csv(fname)

	df['Image'] = df['Image'].apply(lambda x: np.fromstring(x, sep=' '))

	if cols:
		df = df[list(cols) + ['Image']]

	print(df.count())

	df = df.dropna()

	X = np.vstack(df['Image'].values) / 255
	X = X.astype(np.float32)

	if not test:
		y = df[df.columns[:-1]].values
		y = (y-48)/48
		X, y = shuffle(X, y, random_state=22)
		y = y.astype(np.float32)
	else:
		y = None

	return X, y