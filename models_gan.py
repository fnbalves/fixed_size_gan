#Based on https://medium.com/@stepanulyanin/dcgan-adventures-with-cifar10-905fb0a24d21

import tensorflow as tf
import numpy as np

class DCGAN_generator:
    def __init__(self, input_data, training=True, lrelu_slope=0.2, kernel_size=5, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):
            self.input_dense = tf.layers.dense(inputs=input_data, units=2*2*256)
            self.input_volume = tf.reshape(tensor=self.input_dense, shape=(-1, 2, 2, 256))
            self.h1 = tf.layers.batch_normalization(inputs=self.input_volume, training=training)
            self.h1_lrelu = tf.maximum(self.h1*lrelu_slope, self.h1)

            self.h2 = tf.layers.conv2d_transpose(filters=128, strides=2, kernel_size=kernel_size,
            padding='same', inputs=self.h1_lrelu, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.h2_bn = tf.layers.batch_normalization(inputs=self.h2, training=training)
            self.h2_lrelu = tf.maximum(self.h2_bn*lrelu_slope, self.h2_bn)

            self.h3 = tf.layers.conv2d_transpose(filters=64, strides=2, kernel_size=kernel_size,
                                                padding='same', inputs=self.h2_lrelu, activation=None,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.h3_bn = tf.layers.batch_normalization(inputs=self.h3, training=training)
            self.h3_lrelu = tf.maximum(self.h3_bn*lrelu_slope, self.h3_bn)

            self.h4 = tf.layers.conv2d_transpose(filters=32, strides=2, kernel_size=kernel_size,
                                                padding='same', inputs=self.h3_lrelu, activation=None,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.h4_bn = tf.layers.batch_normalization(inputs=self.h4, training=training)
            self.h4_lrelu = tf.maximum(self.h4_bn*lrelu_slope, self.h4_bn)

            self.logits = tf.layers.conv2d_transpose(filters=3, strides=2, kernel_size=kernel_size,
                                                    padding='same', inputs=self.h4_lrelu, activation=None,
                                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.out = tf.tanh(x=self.logits)

class DCGAN_discriminator:
    def __init__(self, input_data, reuse=False, lrelu_slope=0.2, kernel_size=5):
        with tf.variable_scope('discriminator', reuse=reuse):
            self.h1 = tf.layers.conv2d(inputs=input_data, filters=32, strides=2,
                                        kernel_size=kernel_size, padding='same', 
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
            
            self.h1_lrelu = tf.maximum(self.h1*lrelu_slope, self.h1)
            
            self.h2 = tf.layers.conv2d(inputs=self.h1_lrelu, filters=64, strides=2,
                                        kernel_size=kernel_size, padding='same',
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.h2_bn = tf.layers.batch_normalization(inputs=self.h2, training=True)
            self.h2_lrelu = tf.maximum(self.h2_bn*lrelu_slope, self.h2_bn)

            self.h3 = tf.layers.conv2d(inputs=self.h2_lrelu, filters=128, strides=2,
                                        kernel_size=kernel_size, padding='same',
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.h3_bn = tf.layers.batch_normalization(inputs=self.h3, training=True)
            self.h3_lrelu = tf.maximum(self.h3_bn*lrelu_slope, self.h3_bn)
            
            self.h4 = tf.layers.conv2d(inputs=self.h3_lrelu, filters=256, strides=2, 
                            kernel_size=kernel_size, padding='same', 
                            kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.h4_bn = tf.layers.batch_normalization(inputs=self.h4, training=True)
            self.h4_lrelu = tf.maximum(self.h4_bn * lrelu_slope, self.h4_bn)

            self.flatten = tf.reshape(tensor=self.h4_lrelu, shape=(-1, 2*2*256))

            self.logits = tf.layers.dense(inputs=self.flatten, units=1, activation=None, 
                               kernel_initializer=tf.contrib.layers.xavier_initializer())

            self.out = tf.sigmoid(x=self.logits)


class DCGAN_generator_conditional:
    def __init__(self, input_data, input_label, training=True, lrelu_slope=0.2, kernel_size=5, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):
            self.full_input = tf.concat([input_data, input_label], axis=1)
            
            self.input_dense = tf.layers.dense(inputs=self.full_input, units=2*2*256)
            self.input_volume = tf.reshape(tensor=self.input_dense, shape=(-1, 2, 2, 256))
            self.h1 = tf.layers.batch_normalization(inputs=self.input_volume, training=training)
            self.h1_lrelu = tf.maximum(self.h1*lrelu_slope, self.h1)

            self.h2 = tf.layers.conv2d_transpose(filters=128, strides=2, kernel_size=kernel_size,
            padding='same', inputs=self.h1_lrelu, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.h2_bn = tf.layers.batch_normalization(inputs=self.h2, training=training)
            self.h2_lrelu = tf.maximum(self.h2_bn*lrelu_slope, self.h2_bn)

            self.h3 = tf.layers.conv2d_transpose(filters=64, strides=2, kernel_size=kernel_size,
                                                padding='same', inputs=self.h2_lrelu, activation=None,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.h3_bn = tf.layers.batch_normalization(inputs=self.h3, training=training)
            self.h3_lrelu = tf.maximum(self.h3_bn*lrelu_slope, self.h3_bn)

            self.h4 = tf.layers.conv2d_transpose(filters=32, strides=2, kernel_size=kernel_size,
                                                padding='same', inputs=self.h3_lrelu, activation=None,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.h4_bn = tf.layers.batch_normalization(inputs=self.h4, training=training)
            self.h4_lrelu = tf.maximum(self.h4_bn*lrelu_slope, self.h4_bn)

            self.logits = tf.layers.conv2d_transpose(filters=3, strides=2, kernel_size=kernel_size,
                                                    padding='same', inputs=self.h4_lrelu, activation=None,
                                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.out = tf.tanh(x=self.logits)


class DCGAN_discriminator_conditional:
    def __init__(self, input_data, input_label, reuse=False, lrelu_slope=0.2, kernel_size=5):
        with tf.variable_scope('discriminator', reuse=reuse):
            self.h1 = tf.layers.conv2d(inputs=input_data, filters=32, strides=2,
                                        kernel_size=kernel_size, padding='same', 
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
            
            self.h1_lrelu = tf.maximum(self.h1*lrelu_slope, self.h1)
            
            self.h2 = tf.layers.conv2d(inputs=self.h1_lrelu, filters=64, strides=2,
                                        kernel_size=kernel_size, padding='same',
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.h2_bn = tf.layers.batch_normalization(inputs=self.h2, training=True)
            self.h2_lrelu = tf.maximum(self.h2_bn*lrelu_slope, self.h2_bn)

            self.h3 = tf.layers.conv2d(inputs=self.h2_lrelu, filters=128, strides=2,
                                        kernel_size=kernel_size, padding='same',
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.h3_bn = tf.layers.batch_normalization(inputs=self.h3, training=True)
            self.h3_lrelu = tf.maximum(self.h3_bn*lrelu_slope, self.h3_bn)
            
            self.h4 = tf.layers.conv2d(inputs=self.h3_lrelu, filters=256, strides=2, 
                            kernel_size=kernel_size, padding='same', 
                            kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.h4_bn = tf.layers.batch_normalization(inputs=self.h4, training=True)
            self.h4_lrelu = tf.maximum(self.h4_bn * lrelu_slope, self.h4_bn)

            self.flatten = tf.reshape(tensor=self.h4_lrelu, shape=(-1, 2*2*256))
            self.full_flatten = tf.concat([self.flatten, input_label], axis=1)
            
            self.logits = tf.layers.dense(inputs=self.full_flatten, units=1, activation=None, 
                               kernel_initializer=tf.contrib.layers.xavier_initializer())

            self.out = tf.sigmoid(x=self.logits)


class DCGAN_generator_conditional_info:
    def __init__(self, input_data, input_label, info_data, training=True, lrelu_slope=0.2, kernel_size=5, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):
            self.full_input = tf.concat([input_data, input_label, info_data], axis=1)
            
            self.input_dense = tf.layers.dense(inputs=self.full_input, units=2*2*256)
            self.input_volume = tf.reshape(tensor=self.input_dense, shape=(-1, 2, 2, 256))
            self.h1 = tf.layers.batch_normalization(inputs=self.input_volume, training=training)
            self.h1_lrelu = tf.maximum(self.h1*lrelu_slope, self.h1)

            self.h2 = tf.layers.conv2d_transpose(filters=128, strides=2, kernel_size=kernel_size,
            padding='same', inputs=self.h1_lrelu, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.h2_bn = tf.layers.batch_normalization(inputs=self.h2, training=training)
            self.h2_lrelu = tf.maximum(self.h2_bn*lrelu_slope, self.h2_bn)

            self.h3 = tf.layers.conv2d_transpose(filters=64, strides=2, kernel_size=kernel_size,
                                                padding='same', inputs=self.h2_lrelu, activation=None,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.h3_bn = tf.layers.batch_normalization(inputs=self.h3, training=training)
            self.h3_lrelu = tf.maximum(self.h3_bn*lrelu_slope, self.h3_bn)

            self.h4 = tf.layers.conv2d_transpose(filters=32, strides=2, kernel_size=kernel_size,
                                                padding='same', inputs=self.h3_lrelu, activation=None,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.h4_bn = tf.layers.batch_normalization(inputs=self.h4, training=training)
            self.h4_lrelu = tf.maximum(self.h4_bn*lrelu_slope, self.h4_bn)

            self.logits = tf.layers.conv2d_transpose(filters=3, strides=2, kernel_size=kernel_size,
                                                    padding='same', inputs=self.h4_lrelu, activation=None,
                                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.out = tf.tanh(x=self.logits)


class Info_check_network():
    def __init__(self, input_data, reuse=False, num_info=10, lrelu_slope=0.2, kernel_size=5):
        self.num_info = num_info

        with tf.variable_scope('info_check', reuse=reuse):
            self.h1 = tf.layers.conv2d(inputs=input_data, filters=32, strides=2,
                                        kernel_size=kernel_size, padding='same', 
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
            
            self.h1_lrelu = tf.maximum(self.h1*lrelu_slope, self.h1)
            
            self.h2 = tf.layers.conv2d(inputs=self.h1_lrelu, filters=64, strides=2,
                                        kernel_size=kernel_size, padding='same',
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.h2_bn = tf.layers.batch_normalization(inputs=self.h2, training=True)
            self.h2_lrelu = tf.maximum(self.h2_bn*lrelu_slope, self.h2_bn)

            self.h3 = tf.layers.conv2d(inputs=self.h2_lrelu, filters=128, strides=2,
                                        kernel_size=kernel_size, padding='same',
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.h3_bn = tf.layers.batch_normalization(inputs=self.h3, training=True)
            self.h3_lrelu = tf.maximum(self.h3_bn*lrelu_slope, self.h3_bn)
            
            self.h4 = tf.layers.conv2d(inputs=self.h3_lrelu, filters=256, strides=2, 
                            kernel_size=kernel_size, padding='same', 
                            kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.h4_bn = tf.layers.batch_normalization(inputs=self.h4, training=True)
            self.h4_lrelu = tf.maximum(self.h4_bn * lrelu_slope, self.h4_bn)

            self.flatten = tf.reshape(tensor=self.h4_lrelu, shape=(-1, 2*2*256))

            self.logits = tf.layers.dense(inputs=self.flatten, units=self.num_info, activation=None, 
                               kernel_initializer=tf.contrib.layers.xavier_initializer())

            self.out = tf.nn.softmax(self.logits)


class DCGAN_generator_conditional_dropout:
    def __init__(self, input_data, input_label, dropout_rate=0.1, training=True, lrelu_slope=0.2, kernel_size=5, reuse=False):
        self.dropout_rate = dropout_rate

        with tf.variable_scope('generator', reuse=reuse):
            self.full_input = tf.concat([input_data, input_label], axis=1)
            
            self.input_dense = tf.layers.dense(inputs=self.full_input, units=2*2*256)
            self.input_volume = tf.reshape(tensor=self.input_dense, shape=(-1, 2, 2, 256))
            self.h1 = tf.layers.batch_normalization(inputs=self.input_volume, training=training)
            self.h1_lrelu = tf.maximum(self.h1*lrelu_slope, self.h1)
            self.h1_dropout = tf.layers.dropout(self.h1_lrelu, rate=self.dropout_rate, training=True)

            self.h2 = tf.layers.conv2d_transpose(filters=128, strides=2, kernel_size=kernel_size,
            padding='same', inputs=self.h1_dropout, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.h2_bn = tf.layers.batch_normalization(inputs=self.h2, training=training)
            self.h2_lrelu = tf.maximum(self.h2_bn*lrelu_slope, self.h2_bn)
            self.h2_dropout = tf.layers.dropout(self.h2_lrelu, rate=self.dropout_rate, training=True)

            self.h3 = tf.layers.conv2d_transpose(filters=64, strides=2, kernel_size=kernel_size,
                                                padding='same', inputs=self.h2_dropout, activation=None,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.h3_bn = tf.layers.batch_normalization(inputs=self.h3, training=training)
            self.h3_lrelu = tf.maximum(self.h3_bn*lrelu_slope, self.h3_bn)
            self.h3_dropout = tf.layers.dropout(self.h3_lrelu, rate=self.dropout_rate, training=True)

            self.h4 = tf.layers.conv2d_transpose(filters=32, strides=2, kernel_size=kernel_size,
                                                padding='same', inputs=self.h3_dropout, activation=None,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.h4_bn = tf.layers.batch_normalization(inputs=self.h4, training=training)
            self.h4_lrelu = tf.maximum(self.h4_bn*lrelu_slope, self.h4_bn)
            self.h4_dropout = tf.layers.dropout(self.h4_lrelu, rate=self.dropout_rate, training=True)

            self.logits = tf.layers.conv2d_transpose(filters=3, strides=2, kernel_size=kernel_size,
                                                    padding='same', inputs=self.h4_dropout, activation=None,
                                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.out = tf.tanh(x=self.logits)


class Minibatch_discriminator:
    def minibatch(self, input, num_kernels=10, kernel_dim=3):
        x = tf.layers.dense(inputs=input, units=num_kernels * kernel_dim, activation=None, 
                               kernel_initializer=tf.contrib.layers.xavier_initializer())
        activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
        print('Shape activation', activation.get_shape())
        diffs = tf.expand_dims(activation, 3) - \
        tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
        abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
        print('My batch shape', abs_diffs.get_shape())
        minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
        return minibatch_features

    def __init__(self,  discriminator_outputs, reuse=False, num_kernels=10):
        with tf.variable_scope('minibatch_discriminator', reuse=reuse):
            self.minibatch_features = self.minibatch(discriminator_outputs, num_kernels=num_kernels)
            self.logits = tf.layers.dense(inputs=self.minibatch_features, units=2, activation=None, 
                               kernel_initializer=tf.contrib.layers.xavier_initializer())        
            self.output = tf.nn.softmax(self.logits)


class Minibatch_discriminator2:
    def minibatch(self, input, input_labels, num_kernels=10, kernel_dim=3):
        x = tf.layers.dense(inputs=input, units=num_kernels * kernel_dim, activation=None, 
                               kernel_initializer=tf.contrib.layers.xavier_initializer())
        labels_expanded = tf.expand_dims(input_labels, 2)
        
        diff_labels = tf.expand_dims(labels_expanded,3) - \
        tf.expand_dims(tf.transpose(labels_expanded, [1, 2, 0]), 0)

        diff_labels = tf.exp((-100.0)*tf.norm(diff_labels, axis=1))

        
        activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
        
        diffs = tf.expand_dims(activation, 3) - \
        tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
        abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
        abs_diffs = tf.multiply(abs_diffs, diff_labels)
        minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)

        return minibatch_features

    def __init__(self,  discriminator_outputs, labels,  reuse=False, num_kernels=10):
        with tf.variable_scope('minibatch_discriminator', reuse=reuse):
            self.minibatch_features = self.minibatch(discriminator_outputs, labels, num_kernels=num_kernels)
            self.logits = tf.layers.dense(inputs=self.minibatch_features, units=2, activation=None, 
                               kernel_initializer=tf.contrib.layers.xavier_initializer())        
            self.output = tf.nn.softmax(self.logits)

