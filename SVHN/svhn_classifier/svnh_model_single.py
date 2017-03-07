import tensorflow as tf
from pdb import set_trace as bp

# Variables.
DEBUG=False
IMAGE_SIZE=32
PATCH_SIZE = 5
NUM_CHANNELS = 3
DEPTH_L1 = 64
DEPTH_L2 = 128
DEPTH_L3 = 160
STRIDE1 = 2
STRIDE2 = 2
STRIDE3 = 2
STRIDE4 = 2
CONV_OUT = (IMAGE_SIZE/(STRIDE1*STRIDE2*STRIDE3)) 
print ("CONV_OUT",CONV_OUT)
FC_0 = 160
FC_1 = 512
DROPOUT = 1.0
NUM_LABELS=10


layer1_weights = tf.get_variable("layer1_weights", shape=[PATCH_SIZE, PATCH_SIZE, NUM_CHANNELS, DEPTH_L1])
layer1_biases = tf.Variable(tf.zeros([DEPTH_L1]), name='biases_1') 

layer2_weights = tf.get_variable("layer2_weights", shape=[PATCH_SIZE, PATCH_SIZE, DEPTH_L1, DEPTH_L2])
layer2_biases = tf.Variable(tf.zeros([DEPTH_L2]), name='biases_2') 

layer3_weights = tf.get_variable("layer3_weights", shape=[PATCH_SIZE, PATCH_SIZE, DEPTH_L2, DEPTH_L3])
layer3_biases = tf.Variable(tf.zeros([DEPTH_L3]), name='biases_3') 

#layer4_weights = tf.get_variable("layer4_weights", shape=[PATCH_SIZE, PATCH_SIZE, DEPTH_L3, DEPTH_L4])
#layer4_biases = tf.Variable(tf.zeros([DEPTH_L4]), name='biases_4') 

#fc0_weights = tf.get_variable("fc0_weights",shape=[DEPTH_L4, FC_0])
#fc0_biases = tf.Variable(tf.constant(0.05, shape=[FC_0]),name="fc0_biases") 

ws0_weights = tf.get_variable("ws0_weights", shape=[FC_0,NUM_LABELS])
ws0_biases = tf.Variable(tf.constant(0.05, shape=[NUM_LABELS]),name="ws0_biases")



def lenet(data, train=True):
  tf.summary.image("name", data, collections=None)
  
  with tf.variable_scope('Layer_1',reuse=True ) as scope:
    conv = tf.nn.conv2d(data,layer1_weights,strides=[1, 1, 1, 1],padding='VALID' ,name= "CONV_1") 
    relu = tf.nn.relu(tf.nn.bias_add(conv, layer1_biases))                       
    pool = tf.nn.max_pool(relu,ksize=[1, STRIDE1, STRIDE1, 1],strides=[1, STRIDE1, STRIDE1, 1],padding='SAME')
  #pool = tf.nn.local_response_normalization(pool, name="Normalize_1")
  pool_shape = pool.get_shape().as_list() 
  print ("POOL SHAPE",pool_shape[0], pool_shape[1] , pool_shape[2] , pool_shape[3])
  with tf.variable_scope('Layer_2',reuse=True) as scope:
    conv = tf.nn.conv2d(pool,layer2_weights,strides=[1, 1, 1, 1],padding='VALID')
    relu = tf.nn.relu(tf.nn.bias_add(conv, layer2_biases))
    pool = tf.nn.max_pool(relu,ksize=[1, STRIDE2, STRIDE2, 1],strides=[1, STRIDE2, STRIDE2, 1],padding='SAME')
  #pool = tf.nn.local_response_normalization(pool, name="Normalize_2")
  pool_shape = pool.get_shape().as_list() 
  print ("POOL SHAPE",pool_shape[0], pool_shape[1] , pool_shape[2] , pool_shape[3])

  with tf.variable_scope('Layer_3',reuse=True) as scope:
    conv = tf.nn.conv2d(pool,layer3_weights,strides=[1, 1, 1, 1],padding='VALID')
    relu = tf.nn.relu(tf.nn.bias_add(conv, layer3_biases))
  pool = tf.nn.max_pool(relu,ksize=[1,STRIDE3,STRIDE3, 1],strides=[1, STRIDE3, STRIDE3, 1],padding='SAME') 
  #pool = tf.nn.local_response_normalization(pool, name="Normalize_3")
  pool_shape = pool.get_shape().as_list() 
  print ("POOL SHAPE",pool_shape[0], pool_shape[1] , pool_shape[2] , pool_shape[3])
 
  #with tf.variable_scope('Layer_4',reuse=True) as scope:
  #  conv = tf.nn.conv2d(pool,layer4_weights,strides=[1, 1, 1, 1],padding='VALID')
  #  relu = tf.nn.relu(tf.nn.bias_add(conv, layer4_biases))
  #pool = tf.nn.max_pool(relu,ksize=[1,STRIDE4,STRIDE4, 1],strides=[1, STRIDE4, STRIDE4, 1],padding='SAME') 
  ##pool = tf.nn.local_response_normalization(pool, name="Normalize_3")
  #pool_shape = pool.get_shape().as_list() 
  #print ("POOL SHAPE",pool_shape[0], pool_shape[1] , pool_shape[2] , pool_shape[3])

  if train is True:
     print("Using drop out")
     pool = tf.nn.dropout(pool, DROPOUT)
  reshape = tf.reshape(pool,[pool_shape[0], -1])

  #with tf.variable_scope('FC0_layer',reuse=True) as scope:
  #  hidden = tf.nn.relu(tf.matmul(reshape, fc0_weights) + fc0_biases)

  with tf.variable_scope('REG_layer',reuse=True) as scope:
    logits_0=tf.matmul(reshape, ws0_weights)+ws0_biases

  return [logits_0]




    
