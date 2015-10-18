import pandas as pd
import numpy as np
from skimage import io
import cPickle as pickle
from time import time
import sys

sys.setrecursionlimit(10000)

start = time()
for i in xrange(3):
	X = np.zeros((100000,3,32,32))
	for index in xrange(100000):
		if index%5000==0:
			print "Processed {} images, took {} seconds".format(index, time()-start)
			start = time()
		fname = './test/' + str(i*100000 + index+1) + '.png'
		im = np.array(io.imread(fname)).astype(np.float32) / 255
		im = np.rollaxis(im, 2)
		X[index] = im
	print X.shape
	with open('test' + str(i) + '.pkl','wb') as g:
		pickle.dump(X, g)
		g.close()
	

# print df.shape
# df.to_csv('new_train.csv', index=False)
