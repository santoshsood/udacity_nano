
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
from svnh_model_single import lenet 
from sklearn import preprocessing


DEBUG=False
IMAGE_SIZE=32
def accuracy(prediction,labels):
  error = np.mean(np.argmax(prediction, axis=2) == np.argmax(labels,1))
  return error*100

def accuracy_valid(prediction,labels,data,step):
  error = np.mean(np.argmax(prediction, axis=1) == np.argmax(labels,1))
  pred_m=np.argmax(prediction, axis=1)
  label_m= np.argmax(labels,1)
  for i in range(1000):
    if ((pred_m[i]!=label_m[i]) and (step > 700)): 
	print ("i, Label, pred", i, pred_m[i],label_m[i]  )
	plt.imshow(data[i].reshape(32,32,3))
        plt.show()
        
  return error*100

lb = preprocessing.LabelBinarizer()
NUM_CHANNELS=3

with open('../svnh_digit_train.pickle', 'rb') as f:
    save = pickle.load(f)
    X = save['X']
    y = save['y']
    lb.fit(y)
    y=lb.transform(y)
    X = X.reshape(-1, X.shape[1], X.shape[2], NUM_CHANNELS)   
    print ('train: X => ', X.shape, 'y => ', y.shape)

with open('../svnh_digit_valid.pickle', 'rb') as f:
    save = pickle.load(f)
    X_valid = save['X']
    y_valid = save['y']
    lb.fit(y_valid)
    y_valid=lb.transform(y_valid)
    X_valid = X_valid.reshape(-1, X_valid.shape[1], X_valid.shape[2], NUM_CHANNELS)   
    print ('Valid: X => ', X_valid.shape, 'y_valid => ', y_valid.shape)

with open('../svnh_digit_test.pickle', 'rb') as f:
    save = pickle.load(f)
    X_test = save['X']
    y_test = save['y']
    lb.fit(y_valid)
    y_test=lb.transform(y_valid)
    X_test = X_test.reshape(-1, X_test.shape[1], X_test.shape[2], NUM_CHANNELS)   
    print ('Test: X => ', X_test.shape, 'y_test => ', y_test.shape)

    
BATCH_SIZE=256
EVAL_BATCH_SIZE=1000
TEST_BATCH_SIZE=1000
NUM_CHANNELS=3
NUM_LABELS=10
SAVE_FREQUENCY = 500
EVAL_FREQUENCY = 100
NUM_EPOCHS = 100

def train(X,y,X_valid,y_valid,restore=False):
 train_size = X.shape[0]
 global_step = tf.Variable(0, trainable=False)
 with tf.name_scope('input'):
  train_data_node = tf.placeholder(tf.float32,
                    shape=[BATCH_SIZE,IMAGE_SIZE ,IMAGE_SIZE ,NUM_CHANNELS])
  train_labels_node = tf.placeholder(tf.int32,shape=[BATCH_SIZE, NUM_LABELS])
 
 with tf.name_scope('eval_input'):
  eval_data_node = tf.placeholder(tf.float32,
                    shape=(EVAL_BATCH_SIZE,IMAGE_SIZE ,IMAGE_SIZE ,NUM_CHANNELS))

 with tf.name_scope('test_input'):
  test_data_node = tf.placeholder(tf.float32,
                    shape=(TEST_BATCH_SIZE,IMAGE_SIZE ,IMAGE_SIZE ,NUM_CHANNELS))

 #with tf.name_scope('input'):
 # tf.summary.image('train_data_',train_data_node , 10)
 
 with tf.variable_scope('logits') as scope:
  logits=lenet(train_data_node,True)
  scope.reuse_variables()
  train_prediction=tf.nn.softmax(lenet(train_data_node,False))
  eval_prediction=tf.nn.softmax(lenet(eval_data_node,False))


 with tf.name_scope('loss'):
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                  logits=logits, labels=train_labels_node))
  tf.summary.scalar('loss',loss)
 
 learning_rate = tf.train.exponential_decay(
      0.02,              		 # Base learning rate.
      BATCH_SIZE*global_step, 		 # Current index into the dataset.
      train_size,             		 # Decay step.
      0.95,              		 # Decay rate.
      staircase=True)

 with tf.name_scope('optimizer'):
  optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(
        loss, global_step=global_step)

 merged_summary_op = tf.summary.merge_all()
 saver = tf.train.Saver()
  
 def eval_in_batches(data,sess):
   size=data.shape[0]
   print ("DTAA",data.shape)

   if size < EVAL_BATCH_SIZE :
    raise ValueError("Batch size for eval is larger than dataset")
   predictions=np.ndarray(shape=(size,10), dtype=np.float32)
   print ("eval_in_batches")
   for begin in xrange(0, size, EVAL_BATCH_SIZE):
      
      end = begin + EVAL_BATCH_SIZE
      if (end <= size):
        predictions[begin:end, :] = sess.run(
            eval_prediction,
            feed_dict={eval_data_node: data[begin:end,]})
        #print("begin,end", begin, end)
      else:
        batch_predictions = sess.run(
            eval_prediction,
            feed_dict={eval_data_node: data[(size-EVAL_BATCH_SIZE):size,]})
        predictions[(size-EVAL_BATCH_SIZE):size, :] = batch_predictions
        #print("begin,end",size-EVAL_BATCH_SIZE ,size )
   return predictions

 start_time = time.time()
 with tf.Session() as sess:
   writer = tf.summary.FileWriter("./log_trial_1",graph=tf.get_default_graph())  # for 0.8
   tf.initialize_all_variables().run()
    
   if restore:
     saver.restore(sess,"./svhn_single.ckpt")
     print("using checkpoint for initila model")  
   #reader = tf.train.NewCheckpointReader("./svhn_single.ckpt")
   #reader.get_variable_to_shape_map()
   
   for step in xrange(int(NUM_EPOCHS * train_size) // BATCH_SIZE):
      offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
      batch_data = X[offset:(offset + BATCH_SIZE), ]
      batch_labels = y[offset:(offset + BATCH_SIZE),]
        
      # This dictionary maps the batch data (as a numpy array) to the
      # node in the graph is should be fed to.
      feed_dict = {train_data_node: batch_data,
                   eval_data_node : X_valid[:EVAL_BATCH_SIZE,],
                   train_labels_node: batch_labels}
      # Run the graph and fetch some of the nodes.
      _, l, lr, predictions,summary = sess.run(
          [optimizer, loss, learning_rate, train_prediction, merged_summary_op ],
          feed_dict=feed_dict)
      writer.add_summary(summary,step)

      if step % EVAL_FREQUENCY == 0:
        elapsed_time = time.time() - start_time
        start_time = time.time()
        print('Step %d (epoch %.2f), %.1f ms' %
              (step, float(step) * BATCH_SIZE / train_size,
               1000 * elapsed_time / EVAL_FREQUENCY))
        print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
        print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
        print('Validation accuracy: %.1f%%' % accuracy_valid(
            eval_in_batches(X_valid, sess), y_valid, X_valid,step))
      if step % SAVE_FREQUENCY == 0:
       save_path = saver.save(sess, "svhn_single.ckpt")
       print('Model saved in file: {}'.format(save_path))
 
   sys.stdout.flush()
   # Finally print the result!
   test_error = accuracy_valid(eval_in_batches(X_test, sess), y_test,X_test,step)
   print('Test accuracy: %.1f%%' % test_error)
   save_path = saver.save(sess,"svhn_single.ckpt")
   print('Model saved in file: {}'.format(save_path))


train(X,y,X_valid,y_valid,True)

