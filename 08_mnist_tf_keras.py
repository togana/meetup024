import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from tensorflow.contrib.keras.python import keras
from tensorflow.contrib.keras.python.keras import backend as K

def load_data():
    dirn = './MNIST_data'
    mnist = input_data.read_data_sets(dirn, one_hot=True)

    print(mnist.train.num_examples, 'train samples')
    print(mnist.test.num_examples, 'test samples')
    print(mnist.validation.num_examples, 'validation samples (not used)')

    return mnist

def mlp_model(input):
    # MLP network model
    with tf.variable_scope('mlp_model'):
        x = keras.layers.Dense(units=512, activation='relu')(input)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(units=512, activation='relu')(x)
        x = keras.layers.Dropout(0.2)(x)
        y_pred = keras.layers.Dense(units=10, activation='softmax')(x)

    return y_pred

if __name__ == '__main__':
    mnist = load_data()
    # tensorflow placeholders
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    # define TF graph
    y_pred = mlp_model(x)
    loss = tf.losses.softmax_cross_entropy(y_, y_pred)
    train_step = tf.train.AdagradOptimizer(0.05).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        print('Training...')
        for i in range(10001):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            train_fd = {x: batch_xs, y_: batch_ys, K.learning_phase(): 1}
            train_step.run(feed_dict=train_fd)
            if i % 1000 == 0:
                batch_xv, batch_yv = mnist.test.next_batch(200)
                val_accuracy = accuracy.eval(
                    {x: batch_xv, y_: batch_yv, K.learning_phase(): 0})
                print('  step, accurary = %6d: %6.3f' % (i, val_accuracy))

        test_fd = {x: mnist.test.images, y_: mnist.test.labels,
                    K.learning_phase(): 0}
        test_accuracy = accuracy.eval(feed_dict=test_fd)
        print('Test accuracy:', test_accuracy)
