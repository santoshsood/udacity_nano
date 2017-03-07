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

url = 'http://yann.lecun.com/exdb/mnist/'

last_percent_reported = None
PIXEL_DEPTH=255
IMAGE_NUM=None
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_CHAR = 5
def download_progress_hook(count, blockSize, totalSize):
  """A hook to report the progress of a download. This is mostly intended for users with
  slow internet connections. Reports every 5% change in download progress.
  """
  global last_percent_reported
  percent = int(count * blockSize * 100 / totalSize)
  if last_percent_reported != percent:
    if percent % 5 == 0:
      sys.stdout.write("%s%%" % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write(".")
      sys.stdout.flush()
    last_percent_reported = percent

def data_download(filename):
  if not os.path.exists(filename):
  	filename,_=urlretrieve(url+filename,filename , reporthook=download_progress_hook)
  return filename	

train_data_filename = data_download('train-images-idx3-ubyte.gz')
train_labels_filename = data_download('train-labels-idx1-ubyte.gz')
test_data_filename = data_download('t10k-images-idx3-ubyte.gz')
test_labels_filename = data_download('t10k-labels-idx1-ubyte.gz')

#Extract Data for 4-D tensor from training images 
#[images,image_size,image_size,num_channels]
def extract_training(filename,num_images):
 print("Extarxting files & populating dataset")
 with gzip.open(filename) as bytestream:
   bytestream.read(2)     
   MAGIC_NO=np.frombuffer(bytestream.read(2),dtype=np.int16).byteswap()
   #print "MAGIC_NUM",MAGIC_NO
   IMAGE_NUM=np.frombuffer(bytestream.read(4),dtype=np.int32).byteswap()
   #print "IMAGE_NUM",IMAGE_NUM
   IMAGE_SIZE=np.frombuffer(bytestream.read(4),dtype=np.int32).byteswap()
   #print "IMAGE_SIZE",IMAGE_SIZE
   bytestream.read(4)  # read header bytes from file
   
   buf=bytestream.read((IMAGE_SIZE*IMAGE_SIZE*num_images)[0]) 
   data=np.frombuffer(buf,dtype=np.uint8).astype(np.float32)
   data=(data - (PIXEL_DEPTH/2))/PIXEL_DEPTH  # Normalize data
   data=data.reshape((IMAGE_NUM[0],IMAGE_SIZE[0],IMAGE_SIZE[0],NUM_CHANNELS))
   return data

def extract_lables(filename,num_images):
 global IMAGE_SIZE
 global NUM_CHANNELS
 print("Extracting Labels from dataset")
 with gzip.open(filename) as bytestream:
  bytestream.read(4)
  IMAGE_NUM=bytestream.read(4)
  buf=bytestream.read(IMAGE_NUM) 
  data=np.frombuffer(buf,dtype=np.uint8)
  data=data.reshape(1*num_images)
  return data


train_data = extract_training(train_data_filename,60000) #[60000,28,28,1]
train_labels = extract_lables(train_labels_filename,60000) #[60000,1]
test_data = extract_training(test_data_filename,10000)
test_labels = extract_lables(test_labels_filename,10000)


print (train_data.shape )
print (train_labels.shape) 
print (test_data.shape) 
print (test_labels.shape)
print ("MNIST data sucessfully processed")



print ("Processing individual digits :Pickle ")
pickle_file = './digit_train.pickle'
try:
  f = open(pickle_file, 'wb')
  save = {
    'X': train_data[:50000,],
    'y': train_labels[:50000]
  }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print ('Unable to save data to', pickle_file, ':', e)
  raise

pickle_file = './digit_valid.pickle'
try:
  f = open(pickle_file, 'wb')
  save = {
    'X': train_data[50000:],
    'y': train_labels[50000:]
  }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print ('Unable to save data to', pickle_file, ':', e)
  raise


pickle_file = './digit_test.pickle'
try:
  f = open(pickle_file, 'wb')
  save = {
    'X': test_data[:10000],
    'y': test_labels[:10000]
  }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print ('Unable to save data to', pickle_file, ':', e)
  raise


print ("Processing digits sequence")
#Define blank digit
blank = np.random.rand(28,28) - 0.5

#Generate random sequences of 5 or less digit numbers from mnist data
#Blank digt is amrked with label -1
# each sequence of 5 digts is associated with 6 labels, first beign the count of the digits
#followed by 0-5 digits & 5-0 blank spaces
# each sequence ic created by concatinating 5 images 
cont_train_data=np.ndarray(shape=(60000,IMAGE_SIZE,IMAGE_SIZE*NUM_CHAR),dtype=np.float32)
cont_train_labels=np.ndarray(shape=(60000,NUM_CHAR+1),dtype=np.int32)

for i in range(60000):
 num_blank=np.random.randint(0,NUM_CHAR)  
 nums=np.random.choice(60000,size=(NUM_CHAR))
 cont_img=np.ndarray(shape=(NUM_CHAR,IMAGE_SIZE,IMAGE_SIZE))
 cont_label=np.ndarray(shape=(1+NUM_CHAR))

 k=0  
 cont_label[0]=NUM_CHAR-num_blank;      
 for j in range(1,NUM_CHAR-num_blank+1,1):
  cont_img[j-1]=train_data[nums[k]].reshape(28,28)
  cont_label[j]=train_labels[nums[k]]
  k=k+1;

 for l in range(j+1,j+num_blank+1,1):
   cont_img[l-1]=blank
   cont_label[l]=-1

 cont_train_data[i,:,:]=np.hstack(cont_img)
 cont_train_labels[i]=cont_label
 #pylab.imshow(cont_train_data[i].reshape(28,140))
 #plt.show()
 #print(cont_train_labels[i])


pickle_file = './digit_seq_train.pickle'
try:
  f = open(pickle_file, 'wb')
  save = {
    'X': cont_train_data[:50000],
    'y': cont_train_labels[:50000]
  }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print ('Unable to save data to', pickle_file, ':', e)
  raise

#valid dataset with 10000 images
pickle_file = './digit_seq_valid.pickle'
try:
  f = open(pickle_file, 'wb')
  save = {
    'X': cont_train_data[50000:],
    'y': cont_train_labels[50000:]
  }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print ('Unable to save data to', pickle_file, ':', e)
  raise

cont_test_data=np.ndarray(shape=(10000,IMAGE_SIZE,IMAGE_SIZE*NUM_CHAR),dtype=np.float32)
cont_test_labels=np.ndarray(shape=(10000,NUM_CHAR+1),dtype=np.int32)

for i in range(10000):
 #num_blank=1
 num_blank=np.random.randint(0,NUM_CHAR)  
 nums=np.random.choice(10000,size=(NUM_CHAR))
 cont_img=np.ndarray(shape=(NUM_CHAR,IMAGE_SIZE,IMAGE_SIZE))
 cont_label=np.ndarray(shape=(1+NUM_CHAR))
 k=0  
 cont_label[0]=NUM_CHAR-num_blank;      
 for j in range(1,NUM_CHAR-num_blank+1,1):
  cont_img[j-1]=test_data[nums[k]].reshape(28,28)
  cont_label[j]=test_labels[nums[k]]
  k=k+1;
 for l in range(j+1,j+num_blank+1,1):
   cont_img[l-1]=blank
   cont_label[l]=-1
 cont_test_data[i,:,:]=np.hstack(cont_img)
 cont_test_labels[i]=cont_label

#valid dataset with 10000 images
pickle_file = './digit_seq_test.pickle'
try:
  f = open(pickle_file, 'wb')
  save = {
    'X': cont_test_data[:10000],
    'y': cont_test_labels[:10000]
  }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print ('Unable to save data to', pickle_file, ':', e)
  raise





