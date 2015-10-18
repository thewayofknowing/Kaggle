import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.utils import shuffle

from datetime import datetime
from nolearn.lasagne import BatchIterator
from nolearn.lasagne.base import TrainSplit
import lasagne
from lasagne import layers
from lasagne.nonlinearities import softmax, LeakyRectify
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
from skimage import io


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

def load(test=False):
	X=None
	y=None
	
	if test:
		X = np.zeros((300000,3,32,32))
		for index in xrange(300000):
			if index%10000==0:
				print "Processed {} test images".format(index)
			fname = './test/' + str(index+1) + '.png'
			im = np.array(io.imread(fname)).astype(np.float32) / 255
			im = np.rollaxis(im, 2)
			X[index] = im
	else:
		df = pd.read_csv('./trainLabels.csv')
		y = df['label']
		# X = pd.read_csv('./new_train.csv')
		# X = np.reshape(np.array(X), (5000,3,32,32))
		with open('train.pkl','rb') as g:
			X = pickle.load(g)

	# plt.imshow(X[0])
	# plt.show()
	# print X.shape, y.shape
	return X.astype(np.float32), y
	# if cols:
	# 	df = df[list(cols) + ['Image']]

	# X = np.vstack(X.astype(np.float32)) / 255

# def get_shear_sample(X, y, n):
# 	subset = np.random.random_integers(0, X.shape[0]-1, n)
# 	X_sub = X[subset]
# 	for index, img in enumerate(X_sub):
# 		afine_tf = tf.AffineTransform(shear=np.random.uniform(0.1,0.25))
# 		if index%500==0:
# 			print "Processed Shear {}".format(index)
# 		img = tf.warp(img, afine_tf)
# 		img = img.reshape(-1, 1, 28, 28)
# 		X = np.vstack( (X, img) )
# 		y = np.append(y,y[subset[index]])
# 	return X, y

# def get_rotated_sample(X, y, n):
# 	subset = np.random.random_integers(0, X.shape[0]-1, n)
# 	X_sub = X[subset]
# 	for index, img in enumerate(X_sub):
# 		if index%500==0:
# 			print "Processed Rotated {}".format(index)
# 		img1 = tf.rotate(img, np.random.uniform(5,15))
# 		img1 = img1.reshape(-1, 1, 28, 28)
# 		img2 = tf.rotate(img, -np.random.uniform(5,15))
# 		img2 = img2.reshape(-1, 1, 28, 28)
# 		X = np.vstack( (X, img1) )
# 		X = np.vstack( (X, img2) )
# 		y = np.append(y,y[subset[index]])
# 		y = np.append(y,y[subset[index]])
# 	return X, y
	
def predict(net, X):
	y = net.predict(X)
	# print y
	df = DataFrame()
	df['id'] = np.array([i+1 for i in xrange(y.shape[0])])
	df['label'] = y
	print "Writing to Submission File..."
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


net = nn(
	layers = [
		('input', layers.InputLayer),
		('conv1', layers.Conv2DLayer),
		('pool1', layers.MaxPool2DLayer),
		('dropout1', layers.DropoutLayer),
		('conv2', layers.Conv2DLayer),
		('pool2', layers.MaxPool2DLayer),
		('dropout2', layers.DropoutLayer),
		# ('conv3', layers.Conv2DLayer),
		# ('pool3', layers.MaxPool2DLayer),
		# ('dropout3', layers.DropoutLayer),
		# ('conv4', layers.Conv2DLayer),
		# ('pool4', layers.MaxPool2DLayer),
		# ('dropout4', layers.DropoutLayer),
		('hidden5', layers.DenseLayer),
		('dropout5', layers.DropoutLayer),
		('hidden6', layers.DenseLayer),
		('dropout6', layers.DropoutLayer),
		# ('hidden7', layers.DenseLayer),
		# ('dropout7', layers.DropoutLayer),
		('output', layers.DenseLayer),
	],

	input_shape=(None, 3, 32, 32),
	conv1_num_filters = 256, conv1_filter_size=(5,5), pool1_pool_size=(2,2),
	conv1_W=lasagne.init.GlorotUniform(),  
	conv1_nonlinearity=LeakyRectify(leakiness=.03),
	dropout1_p=0.2,
	conv2_num_filters = 128, conv2_filter_size=(2,2), pool2_pool_size=(2,2),
	conv2_W=lasagne.init.GlorotUniform(),
	conv2_nonlinearity=LeakyRectify(leakiness=.03),
	dropout2_p=0.3,
	# conv3_num_filters = 128, conv3_filter_size=(2,2), pool3_pool_size=(2,2),
	# conv3_W=lasagne.init.GlorotUniform(),
	# conv3_nonlinearity=LeakyRectify,
	# dropout3_p=0.4,
	# conv4_num_filters = 256, conv4_filter_size=(2,2), pool4_pool_size=(2,2),
	# conv4_W=lasagne.init.GlorotUniform(),
	# conv4_nonlinearity=LeakyRectify,
	# dropout4_p=0.4,
	hidden5_num_units=4096,
	hidden5_nonlinearity=LeakyRectify(leakiness=.03),
	dropout5_p=0.5,
	hidden6_num_units = 4096,
	hidden6_nonlinearity=LeakyRectify(leakiness=.03),
	dropout6_p=0.5,
	# hidden7_num_units = 1000,
	# hidden7_nonlinearity=LeakyRectify,
	# dropout7_p=0.2,

	batch_iterator_train=BatchIterator(batch_size=128),
	train_split=TrainSplit(eval_size=0.08),
	output_num_units=10,
	output_nonlinearity=softmax,
	# objective_loss_function=categorical_crossentropy,
	regression=False,
	use_label_encoder=True,
	max_epochs=10000,

	update=nesterov_momentum,
	update_learning_rate = theano.shared(float32(0.03)),
	update_momentum = theano.shared(float32(0.9)),

	on_epoch_finished = [
		AdjustVariable('update_learning_rate', start=0.01, stop=0.0001),
		AdjustVariable('update_momentum', start=0.9, stop=0.9999),
		EarlyStopping(patience=200),
	],

	verbose=1
)

X, y = load()
# print X.shape, y.shape
net.fit(X, y)

# with open('net1_copy.pkl', 'rb') as f:
# 	net = pickle.load(f)

with open('net1.pkl', 'wb') as f:
	pickle.dump(net, f, -1)

# predict(net, X)
# visualize.plot_conv_weights(net.layers_['input'])
# plt.show()
plot_loss(net)