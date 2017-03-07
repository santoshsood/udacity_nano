import tensorflow as tf
from pdb import set_trace as bp

# Variables.
DEBUG=False
IMAGE_SIZE=64
PATCH_SIZE = 5
NUM_CHANNELS = 3
DEPTH_L1 = 32
DEPTH_L2 = 64
DEPTH_L3 = 128
DEPTH_L4 = 160
STRIDE1 = 2
STRIDE2 = 2
STRIDE3 = 2
STRIDE4 = 2
FC_0 = 160
FC_1 = 256
DROPOUT1 = 1.0
DROPOUT2 = 1.0
DROPOUT3 = 1.0
DROPOUT4 = 1.0
DROPOUT5 = 1.0
DROPOUT5 = 1.0

NUM_LABELS=10

layer1_weights = tf.get_variable("layer1_weights", shape=[PATCH_SIZE, PATCH_SIZE, NUM_CHANNELS, DEPTH_L1], initializer=tf.contrib.layers.xavier_initializer(),regularizer=tf.contrib.layers.l2_regularizer(0.005))
layer1_biases = tf.Variable(tf.zeros([DEPTH_L1]), name='biases_1') 
  
layer2_weights = tf.get_variable("layer2_weights", shape=[PATCH_SIZE, PATCH_SIZE, DEPTH_L1, DEPTH_L2], initializer=tf.contrib.layers.xavier_initializer(),regularizer=tf.contrib.layers.l2_regularizer(0.005))
layer2_biases = tf.Variable(tf.zeros([DEPTH_L2]), name='biases_2') 

layer3_weights = tf.get_variable("layer3_weights", shape=[PATCH_SIZE, PATCH_SIZE, DEPTH_L2, DEPTH_L3], initializer=tf.contrib.layers.xavier_initializer(),regularizer=tf.contrib.layers.l2_regularizer(0.005))
layer3_biases = tf.Variable(tf.zeros([DEPTH_L3]), name='biases_3') 

layer4_weights = tf.get_variable("layer4_weights", shape=[PATCH_SIZE, PATCH_SIZE, DEPTH_L3, DEPTH_L4], initializer=tf.contrib.layers.xavier_initializer(),regularizer=tf.contrib.layers.l2_regularizer(0.005))
layer4_biases = tf.Variable(tf.zeros([DEPTH_L4], name='biases_4')) 


fc0_weights = tf.get_variable("fc0_weights",shape=[FC_0, FC_1])
fc0_biases = tf.Variable(tf.constant(0.05, shape=[FC_1]),name="fc0_biases") 

#fc1_weights = tf.get_variable("fc1_weights",shape=[FC_1, FC_2])
#fc1_biases = tf.Variable(tf.constant(0.05, shape=[FC_2]),name="fc1_biases") 
#
#fc2_weights = tf.get_variable("fc2_weights",shape=[FC_0, FC_1])
#fc2_biases = tf.Variable(tf.constant(0.05, shape=[FC_1]),name="fc2_biases") 
#
#fc3_weights = tf.get_variable("fc3_weights",shape=[FC_0, FC_1])
#fc3_biases = tf.Variable(tf.constant(0.05, shape=[FC_1]),name="fc3_biases") 
#
#fc4_weights = tf.get_variable("fc4_weights",shape=[FC_0, FC_1])
#fc4_biases = tf.Variable(tf.constant(0.05, shape=[FC_1]),name="fc4_biases") 
#
#fc5_weights = tf.get_variable("fc5_weights",shape=[FC_0, FC_1])
#fc5_biases = tf.Variable(tf.constant(0.05, shape=[FC_1]),name="fc5_biases") 


ws0_weights = tf.get_variable("ws0_weights", shape=[FC_1,NUM_LABELS], initializer=tf.contrib.layers.xavier_initializer(),regularizer=tf.contrib.layers.l2_regularizer(0.005))
ws0_biases = tf.Variable(tf.constant(0.05, shape=[NUM_LABELS]),name="ws0_biases")

ws1_weights = tf.get_variable("ws1_weights", shape=[FC_1,NUM_LABELS], initializer=tf.contrib.layers.xavier_initializer(),regularizer=tf.contrib.layers.l2_regularizer(0.005))
ws1_biases = tf.Variable(tf.constant(0.05, shape=[NUM_LABELS]),name="ws1_biases")

ws2_weights = tf.get_variable("ws2_weights", shape=[FC_1,NUM_LABELS], initializer=tf.contrib.layers.xavier_initializer(),regularizer=tf.contrib.layers.l2_regularizer(0.005))
ws2_biases = tf.Variable(tf.constant(0.05, shape=[NUM_LABELS]),name="ws2_biases")

ws3_weights = tf.get_variable("ws3_weights", shape=[FC_1,NUM_LABELS],initializer=tf.contrib.layers.xavier_initializer(),regularizer=tf.contrib.layers.l2_regularizer(0.005))
ws3_biases = tf.Variable(tf.constant(0.05, shape=[NUM_LABELS]),name="ws3_biases")

ws4_weights = tf.get_variable("ws4_weights", shape=[FC_1,NUM_LABELS], initializer=tf.contrib.layers.xavier_initializer(),regularizer=tf.contrib.layers.l2_regularizer(0.005))
ws4_biases = tf.Variable(tf.constant(0.05, shape=[NUM_LABELS]),name="ws4_biases")

ws5_weights = tf.get_variable("ws5_weights", shape=[FC_1,NUM_LABELS], initializer=tf.contrib.layers.xavier_initializer(),regularizer=tf.contrib.layers.l2_regularizer(0.005))
ws5_biases = tf.Variable(tf.constant(0.05, shape=[NUM_LABELS]),name="ws5_biases")


def lenet(data,train=True):

  tf.summary.image("name", data, collections=None)
  
  with tf.variable_scope('Layer_1',reuse=True ) as scope:
    conv = tf.nn.conv2d(data,layer1_weights,strides=[1, 1, 1, 1],padding='VALID' ,name= "CONV_1") 
    relu = tf.nn.relu(tf.nn.bias_add(conv, layer1_biases))                       
  pool = tf.nn.max_pool(relu,ksize=[1, STRIDE1, STRIDE1, 1],strides=[1, STRIDE1, STRIDE1, 1],padding='SAME')
  pool = tf.nn.local_response_normalization(pool, name="Normalize_1")
  pool_shape = pool.get_shape().as_list() 
  print ("POOL SHAPE1",pool_shape[0], pool_shape[1] , pool_shape[2] , pool_shape[3])
  #if train is True:
  #   print("Using drop out",DROPOUT1)
  #   pool = tf.nn.dropout(pool, DROPOUT1)

  with tf.variable_scope('Layer_2',reuse=True) as scope:
    conv = tf.nn.conv2d(pool,layer2_weights,strides=[1, 1, 1, 1],padding='VALID')
    relu = tf.nn.relu(tf.nn.bias_add(conv, layer2_biases))
  pool = tf.nn.max_pool(relu,ksize=[1, STRIDE2, STRIDE2, 1],strides=[1, STRIDE2, STRIDE2, 1],padding='SAME')
  pool = tf.nn.local_response_normalization(pool, name="Normalize_2")
  pool_shape = pool.get_shape().as_list() 
  print ("POOL SHAPE2",pool_shape[0], pool_shape[1] , pool_shape[2] , pool_shape[3])
  #if train is True:
  #   print("Using drop out",DROPOUT2)
  #   pool = tf.nn.dropout(pool, DROPOUT2)

  with tf.variable_scope('Layer_3',reuse=True) as scope:
    conv = tf.nn.conv2d(pool,layer3_weights,strides=[1, 1, 1, 1],padding='VALID')
    relu = tf.nn.relu(tf.nn.bias_add(conv, layer3_biases))
  pool = tf.nn.max_pool(relu,ksize=[1,STRIDE3,STRIDE3, 1],strides=[1, STRIDE3, STRIDE3, 1],padding='SAME') 
  pool = tf.nn.local_response_normalization(pool, name="Normalize_3")
  pool_shape = pool.get_shape().as_list() 
  print ("POOL SHAPE3",pool_shape[0], pool_shape[1] , pool_shape[2] , pool_shape[3])
  if train is True:
     print("Using drop out",DROPOUT3)
     pool = tf.nn.dropout(pool, DROPOUT3)

  with tf.variable_scope('Layer_4',reuse=True) as scope:
    conv = tf.nn.conv2d(pool,layer4_weights,strides=[1, 1, 1, 1],padding='VALID')
    relu = tf.nn.relu(tf.nn.bias_add(conv, layer4_biases))
  pool = tf.nn.max_pool(relu,ksize=[1,STRIDE4,STRIDE4, 1],strides=[1, STRIDE4, STRIDE4, 1],padding='SAME') 
  pool = tf.nn.local_response_normalization(pool, name="Normalize_4")
  pool_shape = pool.get_shape().as_list() 
  print ("POOL SHAPE4",pool_shape[0], pool_shape[1] , pool_shape[2] , pool_shape[3])
  if train is True:
     print("Using drop out",DROPOUT4)
     pool = tf.nn.dropout(pool, DROPOUT4)
  
  return pool


def classifier(data,train=True):
  pool = lenet(data,True)
  pool_shape = pool.get_shape().as_list() 
  print ("POOL SHAPE5",pool_shape[0], pool_shape[1] , pool_shape[2] , pool_shape[3])

  reshape = tf.reshape(pool,[pool_shape[0], -1])

  with tf.variable_scope('fully_connected_layer0',reuse=True) as scope:
    hidden_0 = tf.nn.relu(tf.matmul(reshape, fc0_weights) + fc0_biases)
  #if train is True:
  #   print("Using drop out",DROPOUT4)
  #   hidden_0 = tf.nn.dropout(hidden_0, DROPOUT4)

 # with tf.variable_scope('fully_connected_layer1',reuse=True) as scope:
 #   hidden_1 = tf.nn.relu(tf.matmul(hidden_0, fc1_weights) + fc1_biases)
 # if train is True:
 #    print("Using drop out",DROPOUT5)
 #    hidden_1 = tf.nn.dropout(hidden_1, DROPOUT5)

  with tf.variable_scope('Softmax_layer',reuse=True) as scope:
    logits_0=tf.matmul(hidden_0, ws0_weights)+ws0_biases
    logits_1=tf.matmul(hidden_0, ws1_weights)+ws1_biases
    logits_2=tf.matmul(hidden_0, ws2_weights)+ws2_biases
    logits_3=tf.matmul(hidden_0, ws3_weights)+ws3_biases
    logits_4=tf.matmul(hidden_0, ws4_weights)+ws4_biases
    logits_5=tf.matmul(hidden_0, ws5_weights)+ws5_biases

  return [logits_0,logits_1,logits_2,logits_3,logits_4,logits_5]




    
