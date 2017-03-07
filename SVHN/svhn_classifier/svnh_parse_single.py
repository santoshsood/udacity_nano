from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
import urllib
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import cPickle as pickle
import gzip
import pylab
import tensorflow as tf
from six.moves import range
import time
import sys
import tensorflow as tf
import scipy.io as sio
from sklearn.cross_validation import train_test_split


train_location = './train_32x32.mat'
test_location = './test_32x32.mat'

def load_train_data():
    train_dict = sio.loadmat(train_location)
    X = np.asarray(train_dict['X'])
    X_train = []
    for i in xrange(X.shape[3]):
      pix = np.asarray(X[:,:,:,i])
      #norm_pix=((255-pix)*1.0)/255
      #norm_pix -= np.mean(norm_pix, axis=0)
      norm_pix=pix*1.0/128-1
      X_train.append(norm_pix)
    X_train = np.asarray(X_train)
    Y_train = train_dict['y']
    for i in xrange(len(Y_train)):
        if Y_train[i]%10 == 0:
            Y_train[i] = 0
    return (X_train,Y_train)


def load_test_data():
    test_dict = sio.loadmat(test_location)
    X = np.asarray(test_dict['X'])
    X_test = []
    for i in xrange(X.shape[3]):
      pix = np.asarray(X[:,:,:,i])
      norm_pix=pix*1.0/128-1
      #norm_pix=((255-pix)*1.0)/255
      #norm_pix -= np.mean(norm_pix, axis=0)
      X_test.append(norm_pix)
    X_test = np.asarray(X_test)
    Y_test = test_dict['y']
    for i in xrange(len(Y_test)):
        if Y_test[i]%10 == 0:
            Y_test[i] = 0
    return (X_test,Y_test)


train_data ,train_labels = load_train_data() 
test_data, test_labels = load_test_data()
train_dataset, validation_dataset, train_labels, validation_labels = train_test_split(train_data, train_labels, test_size=0.1, random_state = 42)


print (train_data.shape )
print (train_labels.shape) 
print (validation_dataset.shape )
print (validation_labels.shape) 
print (test_data.shape) 
print (test_labels.shape)

print ("Processing individual digits :Pickle ")
pickle_file = './svnh_digit_train.pickle'
try:
  f = open(pickle_file, 'wb')
  save = {
    'X': train_dataset,
    'y': train_labels
  }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print ('Unable to save data to', pickle_file, ':', e)
  raise

pickle_file = './svnh_digit_valid.pickle'
try:
  f = open(pickle_file, 'wb')
  save = {
    'X': validation_dataset,
    'y': validation_labels
  }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print ('Unable to save data to', pickle_file, ':', e)
  raise


pickle_file = './svnh_digit_test.pickle'
try:
  f = open(pickle_file, 'wb')
  save = {
    'X': test_data,
    'y': test_labels
  }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print ('Unable to save data to', pickle_file, ':', e)
  raise


