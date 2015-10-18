import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.utils import shuffle

from datetime import datetime
from nolearn.lasagne import BatchIterator
import lasagne
from lasagne import layers
from lasagne.nonlinearities import softmax, leaky_rectify
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet as nn
import theano
import theano.tensor as T
from lasagne.objectives import categorical_crossentropy
from nolearn.lasagne import visualize
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.grid_search import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
import sys
from time import time
import matplotlib.pyplot as plt
import cPickle as pickle
import os
from urllib import urlretrieve
import gzip
from skimage import transform as tf


class EarlyStopping(object):
	def __init__(self, patience=100):
	    self.patience = patience
	    self.best_valid = np.inf
	    self.best_valid_epoch = 0
	    self.best_weights = None

	def __call__(self, nn, train_history):
	    current_valid = train_history[-1]['valid_loss']
	    current_epoch = train_history[-1]['epoch']
	    if current_valid < self.best_valid:
	        self.best_valid = current_valid
	        self.best_valid_epoch = current_epoch
	        self.best_weights = nn.get_all_params_values()
	    elif self.best_valid_epoch + self.patience < current_epoch:
	        print("Early stopping.")
	        print("Best valid loss was {:.6f} at epoch {}.".format(
	            self.best_valid, self.best_valid_epoch))
	        nn.load_params_from(self.best_weights)
	        raise StopIteration()

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

def load_data_shared(filename="./mnist_expanded.pkl"):
    f = open(filename, 'rb')
    training_data, p, q = pickle.load(f)
    p = None
    q = None
    f.close()
    train = np.asarray(training_data[0]).astype(np.float32)
    test = np.asarray(training_data[1]).astype(np.int32)
    # shared_x = theano.shared(train, borrow=True)
    # shared_y = theano.shared(test, borrow=True)
    return train, test

def load(test=False):
	fname = "test.csv" if test else "newest_train.csv"
	
	X=None
	y=None
	
	if test:
		df = pd.read_csv(fname)
		X = np.asarray(df[df.columns[:]].values)
		X = np.vstack(X.astype(np.float32)) / 255
	else:
		X, y = load_data_shared()
		
	# if cols:
	# 	df = df[list(cols) + ['Image']]

	# X = np.vstack(X.astype(np.float32)) / 255
	return X, y

def get_shear_sample(X, y, n):
	subset = np.random.random_integers(0, X.shape[0]-1, n)
	X_sub = X[subset]
	for index, img in enumerate(X_sub):
		afine_tf = tf.AffineTransform(shear=np.random.uniform(0.1,0.25))
		if index%500==0:
			print "Processed Shear {}".format(index)
		img = tf.warp(img, afine_tf)
		img = img.reshape(-1, 1, 28, 28)
		X = np.vstack( (X, img) )
		y = np.append(y,y[subset[index]])
	return X, y

def get_rotated_sample(X, y, n):
	subset = np.random.random_integers(0, X.shape[0]-1, n)
	X_sub = X[subset]
	for index, img in enumerate(X_sub):
		if index%500==0:
			print "Processed Rotated {}".format(index)
		img1 = tf.rotate(img, np.random.uniform(5,15))
		img1 = img1.reshape(-1, 1, 28, 28)
		img2 = tf.rotate(img, -np.random.uniform(5,15))
		img2 = img2.reshape(-1, 1, 28, 28)
		X = np.vstack( (X, img1) )
		X = np.vstack( (X, img2) )
		y = np.append(y,y[subset[index]])
		y = np.append(y,y[subset[index]])
	return X, y

def load2d(test=False):
	X, y = load(test=test)
	X = X.reshape(-1, 1, 28, 28)
	# if not test:
	# 	X, y = get_rotated_sample(X, y, 5000)
	# 	X, y = get_shear_sample(X, y, 5000)
	# 	X = np.vstack(X) * 255
	# 	X = X.astype(np.uint8)
	# 	X = X.reshape(-1,784)
	# 	# d = np.hstack( (y, X) )
	# 	print X.shape, y.shape
	# 	df = DataFrame()
	# 	df['label'] = y
	# 	for index in xrange(X.shape[1]):
	# 		df['pixel' + str(index)] = X[:,index]
	# 	df.to_csv('new_train.csv', index=False)
	# X = X.astype(np.float32)
	return X, y
	
def predict(net):
	X = load2d(test=True)[0]
	y = net.predict(X)
	# print y
	df = DataFrame()
	df['ImageId'] = np.array([i+1 for i in xrange(y.shape[0])])
	df['Label'] = y
	df.to_csv('nn_out.csv', index=False)

def plot_loss(net1):
	train_loss = np.array([i["train_loss"] for i in net1.train_history_])
	valid_loss = np.array([i["valid_loss"] for i in net1.train_history_])
	plt.plot(train_loss, linewidth=2, label="train")
	plt.plot(valid_loss, linewidth=3, label="valid")
	plt.grid()
	plt.legend()
	plt.xlabel("epoch")
	plt.ylabel("loss")
	# plt.yscale("log")
	plt.show()


net1 = nn(
	layers = [
		('input', layers.InputLayer),
		('conv1', layers.Conv2DLayer),
		('pool1', layers.MaxPool2DLayer),
		# ('dropout1', layers.DropoutLayer),
		('conv2', layers.Conv2DLayer),
		('pool2', layers.MaxPool2DLayer),
		('dropout2', layers.DropoutLayer),
		('hidden3', layers.DenseLayer),
		('dropout3', layers.DropoutLayer),
		('hidden4', layers.DenseLayer),
		('dropout4', layers.DropoutLayer),
		('output', layers.DenseLayer),
	],

	input_shape=(None, 1, 28, 28),
	conv1_num_filters = 128, conv1_filter_size=(5,5), pool1_pool_size=(2,2),
	conv1_W=lasagne.init.GlorotUniform(),  
	conv1_nonlinearity=leaky_rectify,
	# dropout1_p=0.3,
	conv2_num_filters = 64, conv2_filter_size=(3,3), pool2_pool_size=(2,2),
	conv2_W=lasagne.init.GlorotUniform(),
	conv2_nonlinearity=leaky_rectify,
	dropout2_p=0.4,
	hidden3_num_units = 2000,  hidden4_num_units=2000,
	hidden3_nonlinearity=leaky_rectify,
	hidden4_nonlinearity=leaky_rectify,
	dropout3_p=0.5,
	dropout4_p=0.5,
	
	batch_iterator_train=BatchIterator(batch_size=256),
	output_num_units=10,
	output_nonlinearity=softmax,
	# objective_loss_function=categorical_crossentropy,
	regression=False,
	use_label_encoder=True,
	max_epochs=30,

	update=nesterov_momentum,
	update_learning_rate = theano.shared(float32(0.1)),
	update_momentum = theano.shared(float32(0.9)),

	on_epoch_finished = [
		AdjustVariable('update_learning_rate', start=0.1, stop=0.001),
		AdjustVariable('update_momentum', start=0.9, stop=0.9999),
		EarlyStopping(patience=6),
	],

	verbose=1
)

# X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
# net1.fit(X_train, y_train)

X, y = load2d()
# print X.shape, y.shape
net1.fit(X, y)

# with open('net1.pkl', 'rb') as f:
# 	net1 = pickle.load(f)

with open('net1.pkl', 'wb') as f:
	pickle.dump(net1, f, -1)

predict(net1)
# visualize.plot_conv_weights(net1.layers_['conv1'])
# plt.show()
plot_loss(net1)