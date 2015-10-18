import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.utils import shuffle

from datetime import datetime
from nolearn.lasagne import BatchIterator
from lasagne import layers
import lasagne
from lasagne.updates import nesterov_momentum
from lasagne.nonlinearities import softmax, leaky_rectify
from nolearn.lasagne import NeuralNet as nn
import theano
from sklearn.grid_search import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
import sys
from time import time
import matplotlib.pyplot as plt

from collections import OrderedDict as od
from sklearn.base import clone
import cPickle as pickle

sys.setrecursionlimit(10000)

SPECIALIST_SETTINGS = [
	dict(
	    columns=(
	        'left_eye_center_x', 'left_eye_center_y',
	        'right_eye_center_x', 'right_eye_center_y',
	        ),
	    flip_indices=((0, 2), (1, 3)),
	    ),

	dict(
	    columns=(
	        'nose_tip_x', 'nose_tip_y',
	        ),
	    flip_indices=(),
	    ),

	dict(
	    columns=(
	        'mouth_left_corner_x', 'mouth_left_corner_y',
	        'mouth_right_corner_x', 'mouth_right_corner_y',
	        'mouth_center_top_lip_x', 'mouth_center_top_lip_y',
	        ),
	    flip_indices=((0, 2), (1, 3)),
	    ),

	dict(
	    columns=(
	        'mouth_center_bottom_lip_x',
	        'mouth_center_bottom_lip_y',
	        ),
	    flip_indices=(),
	    ),

	dict(
	    columns=(
	        'left_eye_inner_corner_x', 'left_eye_inner_corner_y',
	        'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
	        'left_eye_outer_corner_x', 'left_eye_outer_corner_y',
	        'right_eye_outer_corner_x', 'right_eye_outer_corner_y',
	        ),
	    flip_indices=((0, 2), (1, 3), (4, 6), (5, 7)),
	    ),

	dict(
	    columns=(
	        'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',
	        'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y',
	        'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
	        'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y',
	        ),
	    flip_indices=((0, 2), (1, 3), (4, 6), (5, 7)),
	    ),
	]

class FlipBatchIterator(BatchIterator):
	flip_indices = [
	    (0, 2), (1, 3),
	    (4, 8), (5, 9), (6, 10), (7, 11),
	    (12, 16), (13, 17), (14, 18), (15, 19),
	    (22, 24), (23, 25),
	]
	
	def transform(self, Xb, yb):
		Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)
		bs = Xb.shape[0]
		indices = np.random.choice(bs, bs/2, replace=False)
		Xb[indices] = Xb[indices, :, :, ::-1]

		if yb is not None:
			yb[indices, ::2] = yb[indices, ::2] * -1
			for a,b in self.flip_indices:
				yb[indices, a], yb[indices, b] = (
						yb[indices, b] , yb[indices, a]
					)

		return Xb, yb


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

def load(test=False, cols=None):
	if test:
		df = pd.read_csv('test.csv')
	else:
		with open('train.pkl', 'rb') as f:
			df = pickle.load(f)

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

	# print X.shape, y.shape
	return X, y

def load2d(test=False, cols=None):
	X, y = load(test=test, cols=cols)
	X = X.reshape(-1, 1, 96, 96)
	return X, y

def plot_loss(net1):
	train_loss = np.array([i["train_loss"] for i in net1.train_history_])
	valid_loss = np.array([i["valid_loss"] for i in net1.train_history_])
	plt.plot(train_loss, linewidth=2, label="train")
	plt.plot(valid_loss, linewidth=3, label="valid")
	plt.grid()
	plt.legend()
	plt.xlabel("epoch")
	plt.ylabel("loss")
	plt.ylim(5e-4, 1e-2)
	# plt.yscale("log")
	plt.show()

def plot_sample(X, y, axis):
	img = X.reshape(96,96)
	axis.imshow(img, cmap='gray')
	axis.scatter(y[0::2]*48 + 48, y[1::2]*48 + 48, marker='x', s=10)

def plot_samples(net1):
	X, _ = load(test=True)
	y_pred = net1.predict(X)

	fig = plt.figure(figsize=(8,8))
	fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

	for i in xrange(16):
		ax = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
		plot_sample(X[i], y_pred[i], ax)

	plt.show()

# Train network for individual parts
def fit_specialists(net):
	# g = open('net-specialists.pkl','rb')
	# specialists = od()

	for index, setting in enumerate(SPECIALIST_SETTINGS):
		if index not in [1,3,5]:
			continue
		cols = setting['columns']
		X, y = load2d(cols=cols)

		model = clone(net)
		model.output_num_units = y.shape[1]
		model.batch_iterator_train.flip_indices = setting['flip_indices']
		model.max_epochs = int(5e6 / y.shape[0])
		if 'kwargs' in setting:
			vars(model).update(setting['kwargs'])

		with open('net-specialists-' + str(index) + '.pkl', 'rb') as f:
			model.load_params_from(pickle.load(f))

		print("Training model from columns {} for {} epochs").format(
				cols, model.max_epochs 
			)
		model.fit(X, y)
		# specialists[cols] = model

		with open('net-specialists-' + str(index) + '.pkl', 'wb') as f:
			pickle.dump(model, f, -1)
		del model

def predict():
	X = load2d(test=True)[0]
	y_pred = np.empty((X.shape[0], 0))

	# with open(fname_specialists, 'rb') as f:
	# 	specialists = pickle.load(f)

	# for model in specialists.values():
	# 	y_pred1 = model.predict(X)
	# 	y_pred = np.hstack([y_pred, y_pred1])

	columns = ()
	for index, setting in enumerate(SPECIALIST_SETTINGS):
		with open('net-specialists-' + str(index) + '.pkl','rb') as f:
			net = pickle.load(f)
		y_pred_ = net.predict(X)
		y_pred = np.hstack([y_pred, y_pred_])
		columns += setting['columns']
			
	# del specialists
	
	# with open('net-specialists1.pkl', 'rb') as f:
	# 	specialists = pickle.load(f)

	# for model in specialists.values():
	# 	y_pred1 = model.predict(X)
	# 	y_pred = np.hstack([y_pred, y_pred1])

	# for cols in specialists.keys():
	# 	columns += cols

	# with open('net11.pkl','rb') as f:
	# 	net = pickle.load(f)

	# y_pred = net.predict(X)

	y_pred2 = y_pred*48 + 48
	y_pred2 = y_pred2.clip(0,96)

	df = DataFrame(y_pred2, columns=columns)

	lookup_table = pd.read_csv("lookup.csv")
	values = []

	for index, row in lookup_table.iterrows():
		values.append(
			(
				row['RowId'],
				df.ix[row.ImageId - 1][row.FeatureName]
			)
		)

	now_str = datetime.now().isoformat().replace(':','-')
	submission = DataFrame(values, columns=('RowId', 'Location'))
	filename = 'submission-{}.csv'.format(now_str)
	submission.to_csv(filename, index=False)
	print("Wrote {}".format(filename))

def rebin( a, newshape ):
    from numpy import mgrid
    assert len(a.shape) == len(newshape)

    slices = [ slice(0,old, float(old)/new) for old,new in zip(a.shape,newshape) ]
    coordinates = mgrid[slices]
    indices = coordinates.astype('i')   #choose the biggest smaller integer index
    return a[tuple(indices)]

def plot_learning_curves():
	for i in xrange(6):
		with open('net-specialists-' + str(i) + '.pkl', 'rb') as f:
			model = pickle.load(f)
			print model.train_history_[-1]['valid_loss']

	# fig = plt.figure(figsize=(10, 6))
	# ax = fig.add_subplot(1,1,1)
	# ax.set_color_cycle(['c','c','m','m','y','y','k','k','g','g','b','b'])

	# valid_losses = []
	# train_losses = []

	# for model_number, (cg, model) in enumerate(models.items(), 1):
	# 	valid_loss = np.array([i['valid_loss'] for i in model.train_history_])
	# 	valid_loss = np.sqrt(valid_loss) * 48
	# 	train_loss = np.array([i['train_loss'] for i in model.train_history_])
	# 	train_loss = np.sqrt(train_loss) * 48

	# 	valid_loss = rebin(valid_loss, (100, ))
	# 	train_loss = rebin(train_loss, (100, ))

	# 	valid_losses.append(valid_loss)
	# 	train_losses.append(train_loss)

	# 	ax.plot(valid_loss, label='{} ({})'.format(cg[0], len(cg)), linewidth=3)
	# 	ax.plot(train_loss, linestyle='--', linewidth=3, alpha=0.6)
	# 	ax.set_xticks([])

	# weights = np.array([m.output_num_units for m in models.values()], dtype=float)
	# weights /= weights.sum()
	# mean_valid_loss = ()
	# ax.legend()
	# ax.set_ylim((1.0, 4.0))
	# ax.grid()
	# plt.ylabel('RMSE')
	# plt.show()

# net1 = nn(
# 		layers = [
# 			('input', layers.InputLayer),
# 			('hiddens', layers.DenseLayer),
# 			('output', layers.DenseLayer),
# 		],
# 		#params
# 		input_shape=(None, 9216),
# 		hiddens_num_units=100,
# 		output_nonlinearity=None,
# 		output_num_units=30,
# 		#optimization
# 		update=nesterov_momentum,
# 		update_learning_rate=0.01,
# 		update_momentum=0.9,

# 		regression=True,
# 		max_epochs=400,
# 		verbose=1
# )

# param_dist = {
#               "dropout1_p": uniform(0.1,0.2),
#               "dropout2_p": uniform(0.2,0.3),
#               "dropout3_p": uniform(0.3,0.4),
#               "dropout5_p": uniform(0.4,0.55),
#               "conv1_num_filters": [32,48,64],
#               "conv2_num_filters": [64,80,96],
#               "conv3_num_filters": [128,156,196],
#               "hidden5_num_units": [1000,1500],
#               "hidden6_num_units": [1000,1500],
#               "output_nonlinearity": ["None"]}

net9 = nn(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('dropout1', layers.DropoutLayer),  # !
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('dropout2', layers.DropoutLayer),  # !
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('dropout3', layers.DropoutLayer),  # !
        ('hidden4', layers.DenseLayer),
        ('dropout4', layers.DropoutLayer),  # !
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, 96, 96),
    conv1_num_filters=64, conv1_filter_size=(5, 5), pool1_pool_size=(2, 2),
    conv1_nonlinearity=leaky_rectify,
    dropout1_p=0.1,  # !
    conv2_num_filters=128, conv2_filter_size=(3, 3), pool2_pool_size=(2, 2),
    conv2_nonlinearity=leaky_rectify,
    dropout2_p=0.2,  # !
    conv3_num_filters=256, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    conv3_nonlinearity=leaky_rectify,
    dropout3_p=0.3,  # !
    hidden4_num_units=2048,
    hidden4_nonlinearity=leaky_rectify,
    dropout4_p=0.5,  # !
    hidden5_num_units=1000,
    hidden5_nonlinearity=leaky_rectify,
    output_num_units=30, output_nonlinearity=None,

    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),

    regression=True,
    batch_iterator_train=FlipBatchIterator(batch_size=128),
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.01, stop=0.0007),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        EarlyStopping(patience=20),
        ],
    max_epochs=5000,
    verbose=1,
)

# plot_learning_curves()
# predict()

# with open('net11.pkl', 'rb') as f:
# 	net = pickle.load(f)
	# net9.load_params_from(net)

fit_specialists(net9)

# X, y = load2d()
# random_search = RandomizedSearchCV(net7, n_jobs=7, param_distributions=param_dist, verbose=True)

# with open('net-specialists.pkl','rb') as f:
# 	model = pickle.load(f)
# 	(cg, model) = model.items()[0]
# 	for i in model.train_history_:
# 		print i['valid_loss'] 

# string = random_search.best_params_
# g = open('params.txt','w')
# g.write(string)
# g.close()
# print string

# net9.fit(X, y)

# with open('net11.pkl','wb') as f:
# 	pickle.dump(net9, f, -1)

# plot_loss(net9)
