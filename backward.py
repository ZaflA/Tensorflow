import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import forward
import numpy as np
import os
import config

def backward(mnist):
    #input
    xs = tf.placeholder(tf.float32,[config.BATCH_SIZE,config.IMAGE_SIZE,config.IMAGE_SIZE,config.CHANNEL_NUM])
    ys = tf.placeholder(tf.float32,[None,config.OUTPUT_NODE])

    #prediction
    y = forward.forward(xs, True, config.REGULARIZER)

    #cross entropy & loss
    '''ce = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(ys,1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))'''
    loss = tf.reduce_mean(-tf.reduce_sum(tf.square(ys * tf.log(y)), reduction_indices=[1]))

    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        #restore model
        ckpt = tf.train.get_checkpoint_state(config.MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(config.STEPS):
            x_data, y_data = mnist.train.next_batch(config.BATCH_SIZE)
            re_x = np.reshape(x_data, (
                config.BATCH_SIZE,
                config.IMAGE_SIZE,
                config.IMAGE_SIZE,
                config.CHANNEL_NUM))

            sess.run(train_step, feed_dict={xs: re_x, ys: y_data})
            if i % 1000 == 0:
                print(sess.run(loss, feed_dict={xs: re_x, ys: y_data}))
                saver.save(sess, os.path.join(config.MODEL_SAVE_PATH, config.MODEL_NAME))

def main():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    backward(mnist)

if __name__ == '__main__':
    main()