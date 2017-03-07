import tensorflow as tf
from pdb import set_trace as bp

# Variables.
DEBUG=False
IMAGE_SIZE=28
PATCH_SIZE = 5
NUM_CHANNELS = 1
DEPTH_L1 = 16
DEPTH_L2 = 32
DEPTH_L3 = 48
STRIDE1 = 2
STRIDE2 = 2
STRIDE3 = 2
NUM_CHARS=5
CONV_OUT = (IMAGE_SIZE/(STRIDE1*STRIDE2*STRIDE3))*((IMAGE_SIZE*NUM_CHARS)/(STRIDE1*STRIDE2*STRIDE3)) 
print ("CONV_OUT",CONV_OUT)
FC_0 = 3456
FC_1 = 128
DROPOUT = 0.9
NUM_LABELS=10


def lenet(data, train=True):
  layer1_weights = tf.get_variable("layer1_weights", shape=[PATCH_SIZE, PATCH_SIZE, NUM_CHANNELS, DEPTH_L1])
  layer1_biases = tf.Variable(tf.zeros([DEPTH_L1]), name='biases_1') 
  
  layer2_weights = tf.get_variable("layer2_weights", shape=[PATCH_SIZE, PATCH_SIZE, DEPTH_L1, DEPTH_L2])
  layer2_biases = tf.Variable(tf.zeros([DEPTH_L2]), name='biases_2') 
  
  layer3_weights = tf.get_variable("layer3_weights", shape=[PATCH_SIZE, PATCH_SIZE, DEPTH_L2, DEPTH_L3])
  layer3_biases = tf.Variable(tf.zeros([DEPTH_L3]), name='biases_3') 
  
  fc0_weights = tf.get_variable("fc0_weights",shape=[DEPTH_L3*CONV_OUT, FC_0])
  fc0_biases = tf.Variable(tf.constant(0.1, shape=[FC_0]),name="fc0_biases") 
  
  ws0_weights = tf.get_variable("ws0_weights", shape=[FC_0,NUM_LABELS])
  ws0_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]),name="ws0_biases")
  
  ws1_weights = tf.get_variable("ws1_weights", shape=[FC_0,NUM_LABELS])
  ws1_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]),name="ws1_biases")
  
  ws2_weights = tf.get_variable("ws2_weights", shape=[FC_0,NUM_LABELS])
  ws2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]),name="ws2_biases")

  ws3_weights = tf.get_variable("ws3_weights", shape=[FC_0,NUM_LABELS])
  ws3_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]),name="ws3_biases")

  ws4_weights = tf.get_variable("ws4_weights", shape=[FC_0,NUM_LABELS])
  ws4_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]),name="ws4_biases")

  ws5_weights = tf.get_variable("ws5_weights", shape=[FC_0,NUM_LABELS])
  ws5_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]),name="ws5_biases")


  data1=tf.reshape(data,[-1,IMAGE_SIZE, IMAGE_SIZE*NUM_CHARS, NUM_CHANNELS])
  tf.summary.image("name", data1, collections=None)
  
  with tf.variable_scope('Layer_1',reuse=True ) as scope:
    conv = tf.nn.conv2d(data,layer1_weights,strides=[1, 1, 1, 1],padding='SAME' ,name= "CONV_1") 
    relu = tf.nn.relu(tf.nn.bias_add(conv, layer1_biases))                       
    pool = tf.nn.max_pool(relu,ksize=[1, STRIDE1, STRIDE1, 1],strides=[1, STRIDE1, STRIDE1, 1],padding='SAME')
    pool = tf.nn.local_response_normalization(pool, name="Normalize_1")
    pool_shape = pool.get_shape().as_list() 
    print ("POOL SHAPE",pool_shape[0], pool_shape[1] , pool_shape[2] , pool_shape[3])
  with tf.variable_scope('Layer_2',reuse=True) as scope:
    conv = tf.nn.conv2d(pool,layer2_weights,strides=[1, 1, 1, 1],padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, layer2_biases))
    pool = tf.nn.max_pool(relu,ksize=[1, STRIDE2, STRIDE2, 1],strides=[1, STRIDE2, STRIDE2, 1],padding='SAME')
    pool = tf.nn.local_response_normalization(pool, name="Normalize_2")

    pool_shape = pool.get_shape().as_list() 
    print ("POOL SHAPE",pool_shape[0], pool_shape[1] , pool_shape[2] , pool_shape[3])

  with tf.variable_scope('Layer_3',reuse=True) as scope:
    conv = tf.nn.conv2d(pool,layer3_weights,strides=[1, 1, 1, 1],padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, layer3_biases))
    pool = tf.nn.max_pool(relu,ksize=[1,STRIDE3,STRIDE3, 1],strides=[1, STRIDE3, STRIDE3, 1],padding='SAME') 
    pool = tf.nn.local_response_normalization(pool, name="Normalize_3")
    pool_shape = pool.get_shape().as_list() 
    print ("POOL SHAPE",pool_shape[0], pool_shape[1] , pool_shape[2] , pool_shape[3])

    if train is True:
     print("Using drop out")
     pool = tf.nn.dropout(pool, DROPOUT)
    reshape = tf.reshape(pool,[pool_shape[0], -1])
  
  

  #with tf.variable_scope('FC0_layer',reuse=True) as scope:
  #  hidden = tf.nn.relu(tf.matmul(reshape, fc0_weights) + fc0_biases)

  with tf.variable_scope('REG_layer',reuse=True) as scope:
    logits_0=tf.matmul(reshape, ws0_weights)+ws0_biases
    logits_1=tf.matmul(reshape, ws1_weights)+ws1_biases
    logits_2=tf.matmul(reshape, ws2_weights)+ws2_biases
    logits_3=tf.matmul(reshape, ws3_weights)+ws3_biases
    logits_4=tf.matmul(reshape, ws4_weights)+ws4_biases
    logits_5=tf.matmul(reshape, ws5_weights)+ws5_biases

  if DEBUG:
   layer1_weights = tf.Print(layer1_weights, [layer1_weights], "layer1_weights: ")
   layer3_weights = tf.Print(layer2_weights, [layer1_weights], "layer2_weights: ")
   layer3_weights = tf.Print(layer3_weights, [layer1_weights], "layer3_weights: ")
   layer1_biases = tf.Print(layer1_biases, [layer1_biases], "layer1_biases: ")
   layer3_biases = tf.Print(layer2_biases, [layer1_biases], "layer2_biases: ")
   layer3_biases = tf.Print(layer3_biases, [layer1_biases], "layer3_biases: ")
   fc0_biases = tf.Print(fc0_biases, [fc0_biases], "FC0_Bias: ")
   fc0_weights = tf.Print(fc0_weights, [fc0_weights], "FC0_Weight: ")
   ws0_biases = tf.Print(ws0_biases, [ws0_biases], "WS0_Bias: ")
   ws0_weights = tf.Print(ws0_weights, [ws0_weights], "WS0_Weight: ")
   logits_0 = tf.Print(logits_0, [logits_0], "Logit0: ")
   logits_1 = tf.Print(logits_1, [logits_0], "Logit1: ")
   logits_2 = tf.Print(logits_2, [logits_0], "Logit2: ")
   logits_3 = tf.Print(logits_3, [logits_0], "Logit3: ")
   logits_4 = tf.Print(logits_4, [logits_0], "Logit4: ")

  return [logits_0,logits_1,logits_2,logits_3,logits_4,logits_5]




    
