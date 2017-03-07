
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
from mnist_model_seq import lenet 
from sklearn import preprocessing

from math import sqrt


DEBUG=False
IMAGE_SIZE=28
NUM_CHANNELS=1
def accuracy(prediction,labels):
  error=0
  for i in range(6):
   mask1=labels[:,i] > -1
   labels_m=labels[mask1,i]
   predict_m=prediction[mask1,i*10:i*10+10]
   error+=np.mean(np.argmax(predict_m, axis=1) == labels_m)
  return (error*100)/6

def accuracy_valid(prediction,labels):
  # Accuracy for individaual digits error
  error=0
  pred_nos=np.ndarray([labels.shape[0],6])
  for i in range(6):
   pred_nos[:,i]=np.argmax(prediction[:,i*10:i*10+10],1)
   mask1=labels[:,i] > -1
   labels_m=labels[mask1,i]
   predict_m=prediction[mask1,i*10:i*10+10]
   error+=np.mean(np.argmax(predict_m, axis=1) == labels_m)
  error=(error*100)/6  


  # Accuracy for sequence of digits
  predict_seq=np.ndarray([labels.shape[0]])
  label_seq=np.ndarray([labels.shape[0]])
  mask = labels == -1
  labels[mask] = 0
  for i in range(labels.shape[0]):
   predict_seq[i]=int(int((pred_nos[i,1]*10**(pred_nos[i,0]-1)))+ \
                  int((pred_nos[i,2]*10**(pred_nos[i,0]-2)))+ \ 
                  int((pred_nos[i,3]*10**(pred_nos[i,0]-3)))+ \
                  int((pred_nos[i,4]*10**(pred_nos[i,0]-4)))+ \
                  int((pred_nos[i,5]*10**(pred_nos[i,0]-5))))

   label_seq[i]=int(int((labels[i,1]*10**(float(labels[i,0]-1))*1.0))+ \
                    int((labels[i,2]*10**(float(labels[i,0]-2))*1.0))+ \ 
                    int((labels[i,3]*10**(float(labels[i,0]-3))*1.0))+ \
                    int((labels[i,4]*10**(float(labels[i,0]-4))*1.0))+ \
                    int((labels[i,5]*10**(float(labels[i,0]-5))*1.0)))
   
  error_seq=np.mean(predict_seq==label_seq)*100
  
  return  error, error_seq


lb = preprocessing.LabelBinarizer()
with open('./digit_seq_train.pickle', 'rb') as f:
    save = pickle.load(f)
    X = save['X']
    y = save['y']
    X = X.reshape(-1, X.shape[1], X.shape[2], NUM_CHANNELS)   
    print ('train: X => ', X.shape, 'y => ', y.shape)

with open('./digit_seq_valid.pickle', 'rb') as f:
    save = pickle.load(f)
    X_valid = save['X']
    y_valid = save['y']
    X_valid = X_valid.reshape(-1, X_valid.shape[1], X_valid.shape[2], NUM_CHANNELS)   
    print ('Valid: X => ', X_valid.shape, 'y_valid => ', y_valid.shape)

with open('./digit_seq_test.pickle', 'rb') as f:
    save = pickle.load(f)
    X_test = save['X']
    y_test = save['y']
    X_test = X_test.reshape(-1, X_test.shape[1], X_test.shape[2],NUM_CHANNELS )   
    print ('test: X => ', X_test.shape, 'y => ', y_test.shape)


BATCH_SIZE=500
EVAL_BATCH_SIZE=100
TEST_BATCH_SIZE=100
NUM_CHANNELS=1
NUM_LABELS=6
SAVE_FREQUENCY = 500
EVAL_FREQUENCY = 100
NUM_EPOCHS = 400

def train(X,y,X_valid,y_valid,restore=False):
 train_size = X.shape[0]
 global_step = tf.Variable(0, trainable=False)
 with tf.name_scope('input'):
  
  train_data_node = tf.placeholder(tf.float32,
                    shape=[BATCH_SIZE,IMAGE_SIZE ,IMAGE_SIZE*(NUM_LABELS-1) ,NUM_CHANNELS])
  train_labels_node = tf.placeholder(tf.int32,shape=[BATCH_SIZE, NUM_LABELS])
 
 with tf.name_scope('eval_input'):
  eval_data_node = tf.placeholder(tf.float32,
                    shape=(EVAL_BATCH_SIZE,IMAGE_SIZE ,IMAGE_SIZE*(NUM_LABELS-1) ,NUM_CHANNELS))

 with tf.name_scope('test_input'):
  test_data_node = tf.placeholder(tf.float32,
                    shape=(TEST_BATCH_SIZE,IMAGE_SIZE ,IMAGE_SIZE*(NUM_LABELS-1) ,NUM_CHANNELS))

 with tf.name_scope('input'):
  tf.summary.image('train_data_',train_data_node , 10)
 
 with tf.variable_scope('logits') as scope:
  [logits_0,logits_1,logits_2,logits_3,logits_4,logits_5,]=lenet(train_data_node,True)
  scope.reuse_variables()
  [logits_0_train,logits_1_train,logits_2_train,logits_3_train,logits_4_train,logits_5_train,]=lenet(train_data_node,False)
  [logits_0_eval,logits_1_eval,logits_2_eval,logits_3_eval,logits_4_eval,logits_5_eval,]=lenet(eval_data_node,False)
  layer1_weights = tf.get_variable('layer1_weights')
  grid1 = put_kernels_on_grid (layer1_weights)
  tf.summary.image("layer1",grid1, max_outputs=3)
  
  train_prediction=tf.concat([tf.nn.softmax(logits_0_train),
		   tf.nn.softmax(logits_1_train),
	  	   tf.nn.softmax(logits_2_train),	
		   tf.nn.softmax(logits_3_train),
		   tf.nn.softmax(logits_4_train),
		   tf.nn.softmax(logits_5_train)],1)

  eval_prediction=tf.concat([tf.nn.softmax(logits_0_eval),
		   tf.nn.softmax(logits_1_eval),
	  	   tf.nn.softmax(logits_2_eval),	
		   tf.nn.softmax(logits_3_eval),
		   tf.nn.softmax(logits_4_eval),
		   tf.nn.softmax(logits_5_eval)],1)


 with tf.name_scope('loss'):
  

  label_mask = tf.constant(-1,dtype=tf.int32,shape=(BATCH_SIZE,1))
  mask = tf.greater(train_labels_node,label_mask )
  mask0=mask[:,0]
  mask1=mask[:,1]
  mask2=mask[:,2]
  mask3=mask[:,3]
  mask4=mask[:,4]
  mask5=mask[:,5]
  logits_1_m=tf.boolean_mask(logits_1, mask1, name='boolean_mask1')
  logits_2_m=tf.boolean_mask(logits_2, mask2, name='boolean_mask2')
  logits_3_m=tf.boolean_mask(logits_3, mask3, name='boolean_mask3')
  logits_4_m=tf.boolean_mask(logits_4, mask4, name='boolean_mask4')
  logits_5_m=tf.boolean_mask(logits_5, mask5, name='boolean_mask5')
 
  label0 = tf.boolean_mask(train_labels_node[:,0],mask0)
  label1 = tf.boolean_mask(train_labels_node[:,1],mask1)
  label2 = tf.boolean_mask(train_labels_node[:,2],mask2)
  label3 = tf.boolean_mask(train_labels_node[:,3],mask3)
  label4 = tf.boolean_mask(train_labels_node[:,4],mask4)
  label5 = tf.boolean_mask(train_labels_node[:,5],mask5)
  
  loss0=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits( logits=logits_0  , labels=label0))
  loss1=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits( logits=logits_1_m, labels=label1))
  loss2=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits( logits=logits_2_m, labels=label2))
  loss3=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits( logits=logits_3_m, labels=label3))
  loss4=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits( logits=logits_4_m, labels=label4))
  loss5=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits( logits=logits_5_m, labels=label5))
			
  loss = loss0+loss1+loss2+loss3+loss4+loss5
  tf.summary.scalar('loss',loss)
 
 learning_rate = tf.train.exponential_decay(
      0.09,              		 # Base learning rate.
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
   predictions=np.ndarray(shape=(size,60), dtype=np.float32)
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
   writer = tf.summary.FileWriter("./log_mnist_2",graph=tf.get_default_graph())  # for 0.8
   tf.initialize_all_variables().run()
   reader = tf.train.NewCheckpointReader("./mnist_seq2.ckpt")
   reader.get_variable_to_shape_map()

   if restore:
     saver.restore(sess,"./mnist_seq2.ckpt")
     #saver.restore(sess,"./mnist_seq.ckpt")
     print("using checkpoint for initila model")  
    
   for step in xrange(int(NUM_EPOCHS * train_size) // BATCH_SIZE):
      offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
      batch_data = X[offset:(offset + BATCH_SIZE), ]
      batch_labels = y[offset:(offset + BATCH_SIZE),]
        
      # This dictionary maps the batch data (as a numpy array) to the
      # node in the graph is should be fed to.
      feed_dict = {train_data_node: batch_data,
                   eval_data_node: X_valid[:EVAL_BATCH_SIZE,],
                   train_labels_node: batch_labels}
      # Run the graph and fetch some of the nodes.
      _, l, lr, predictions,summary = sess.run(
          [optimizer, loss, learning_rate, train_prediction,merged_summary_op  ],
          feed_dict=feed_dict)
      writer.add_summary(summary,step)

      if step % EVAL_FREQUENCY == 0:
        elapsed_time = time.time() - start_time
        start_time = time.time()
        print('Step %d (epoch %.2f), %.1f ms' %
              (step, float(step) * BATCH_SIZE / train_size,
               1000 * elapsed_time / EVAL_FREQUENCY))
        print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
        accuracy,accuracy_seq=accuracy_valid(predictions, batch_labels)
        print('Minibatch accuracy: %.1f%%  sequence accuracy %.1f%%' % (accuracy,accuracy_seq))
        print('Validation accuracy: %.1f%% sequence accuracy %.1f%% ' % accuracy_valid(
            eval_in_batches(X_valid, sess), y_valid))
      if step % SAVE_FREQUENCY == 0:
       save_path = saver.save(sess, "mnist_seq3.ckpt")
       print('Model saved in file: {}'.format(save_path))
 
   sys.stdout.flush()
   # Finally print the result!
   test_error = accuracy_valid(eval_in_batches(X_test, sess), y_test)
   print('Test accuracy: %.1f%%' % test_error)
   save_path = saver.save(sess,"mnist_seq3.ckpt")
   print('Model saved in file: {}'.format(save_path))


train(X,y,X_valid,y_valid,False)

