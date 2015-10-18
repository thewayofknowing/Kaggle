import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import transform
import math

# def rotated(point,angle):
# 	centerPoint = [48,48]
# 	angle = math.radians(angle)
# 	temp_point = point[0]-centerPoint[0] , point[1]-centerPoint[1]
# 	temp_point = ( temp_point[0]*math.cos(angle)-temp_point[1]*math.sin(angle) , temp_point[0]*math.sin(angle)+temp_point[1]*math.cos(angle))
# 	temp_point = temp_point[0]+centerPoint[0] , temp_point[1]+centerPoint[1]
# 	return temp_point

# def plot_sample(X, X_, y):
# 	img = (X*255).reshape(96,96)
# 	img_ = (X_*255).reshape(96,96)
# 	fig = plt.figure()
# 	fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
# 	ax = fig.add_subplot(4, 4, 0, xticks=[], yticks=[])
# 	ax.imshow(img_, cmap='gray')
# 	for i in xrange(0, len(y), 2):
# 		ax.scatter(y[i]*48+48, y[i+1]*48+48, marker='x', s=10)
		# point = rotated((y[i]*48+48, y[i+1]*48+48), 10)
		# ax.scatter(point[0], point[1], marker='x', s=10)
	# ax2 = fig.add_subplot(4, 4, 1, xticks=[], yticks=[])
	# ax2.imshow(img_, cmap='gray')
	# for i in xrange(0, len(y), 2):
	# 	point = rotated((y[i]*48+48, y[i+1]*48+48), 10)
	# 	ax2.scatter(point[0], point[1], marker='x', s=10)
	# plt.show()

df = pd.read_csv('train.csv', na_values=np.nan)
# df = df.dropna()

X = df['Image'].apply(lambda x: np.fromstring(x, sep=' '))
Y = df[df.columns[:-1]]
df2 = pd.DataFrame(columns=df.columns)

nY = []
nX = []

for j in xrange(len(X)):
	if j%200==0:
		print "Processed {}".format(j)
	x = X[j]
	x = x.reshape(96,96)
	y = Y.ix[j]
	for d, axis, index_position, index in [
	                (1,  0, "first", 0),
	                (-1, 0, "first", 27),
	                (1,  1, "last",  0),
	                (-1, 1, "last",  27)
	                ]:
	            X_ = np.roll(x, d, axis)
	            y_ = []
	            if index_position == "first": 
	            	for i in xrange(len(y)):
	            		if y[i] == np.nan:
	            			y_.append(np.nan)
	            		elif i%2==0:
	            			y_.append(y[i])
	            		else:
	            			y_.append(y[i]+d)
	                X_[index, :] = np.zeros(96)
	            else: 
	            	for i in xrange(len(y)):
	            		if y[i] == np.nan:
	            			y_.append(np.nan)
	            		elif i%2==1:
	            			y_.append(y[i])
	            		else:
	            			y_.append(y[i]+d)
	                X_[:, index] = np.zeros(96)
	            nY.append(y_)
	            nX.append(' '.join(map(str,np.ravel(X_))))
df2['Image'] = nX
df2[df2.columns[:-1]] = nY 

# X_ = transform.warp(X_.reshape(96,96), afine_transform)
print df.shape
df = df.append(df2)
print df.shape
# df.to_csv('new_train.csv', index=False)
with open('train.pkl','wb') as f:
	import pickle
	pickle.dump(df, f, -1)