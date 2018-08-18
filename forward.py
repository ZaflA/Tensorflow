import tensorflow as tf
import config

def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape))
    if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b

def conv2d(x,w):
    return tf.nn.conv2d(x,w,[1,1,1,1],padding='SAME')

def pool_max_2x2(x):
    return tf.nn.max_pool(x,[1,2,2,1],[1,2,2,1],padding='SAME')

def forward(x,train, regularizer):
    #CNN 1&2
    w1 = get_weight([config.CONV1_SIZE,config.CONV1_SIZE,config.CHANNEL_NUM,config.CONV1_KERNEL_NUM], regularizer)
    b1 = get_bias([config.CONV1_KERNEL_NUM])
    conv1 = conv2d(x,w1)
    y1 = tf.nn.relu(tf.nn.bias_add(conv1, b1))
    pool1 = pool_max_2x2(y1)

    w2 = get_weight([config.CONV2_SIZE,config.CONV2_SIZE,config.CONV1_KERNEL_NUM,config.CONV2_KERNEL_NUM], regularizer)
    b2 = get_bias([config.CONV2_KERNEL_NUM])
    conv2 = conv2d(pool1,w2)
    y2 = tf.nn.relu(tf.nn.bias_add(conv2, b2))
    pool2 = pool_max_2x2(y2)

    #reshape for input
    shape = pool2.get_shape().as_list()
    all_pix = shape[1] * shape[2] * shape[3]
    reshape = tf.reshape(pool2,[shape[0],all_pix])

    #fully connected layer 1&2
    fcl1_w = get_weight([all_pix,config.FCL1_NODE], regularizer)
    fcl1_b = get_bias([config.FCL1_NODE])
    fcl1 = tf.nn.relu(tf.matmul(reshape,fcl1_w) + fcl1_b)
    if train:
        fcl1 = tf.nn.dropout(fcl1, 0.5)

    fcl2_w = get_weight([config.FCL1_NODE,config.OUTPUT_NODE], regularizer)
    fcl2_b = get_bias([config.OUTPUT_NODE])
    fcl2 = tf.nn.relu(tf.matmul(fcl1,fcl2_w) + fcl2_b)
    return fcl2


