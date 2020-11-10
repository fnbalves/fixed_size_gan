# Based on https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

import tensorflow as tf
import numpy as np


def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1, verbose_shapes=False, batch_norm=False, reuse=False):
    """Convolution function that can be split in multiple GPUs"""
    # Get number of input chennels
    input_channels = int(x.get_shape()[-1])

    if verbose_shapes:
        print('INPUT_CHANNELS', input_channels)
        print('X SHAPE conv', x.get_shape())

    convolve = lambda i, k: tf.nn.conv2d(i, k, strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name, reuse=reuse) as scope:
        weights = tf.get_variable('weights',
        shape=[filter_height, filter_width, input_channels / groups, num_filters],
        trainable=True,
        initializer=tf.contrib.layers.xavier_initializer())
        
        biases = tf.get_variable('biases', shape=[num_filters], trainable=True,
        initializer=tf.contrib.layers.xavier_initializer())
        
        if groups == 1:
            conv = convolve(x, weights)
        else:
            input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
            weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=weights)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

            conv = tf.concat(axis=3, values=output_groups)

        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        
        if batch_norm:
            norm = lrn(bias, 2, 2e-05, 0.75, name=scope.name)
            relu = tf.nn.relu(norm, name=scope.name)
        else:
            relu = tf.nn.relu(bias, name=scope.name)

        return relu


def fc(x, num_in, num_out, name, relu=True, use_biases=True, reuse=False):
    """Full connected layer"""
    with tf.variable_scope(name, reuse=reuse) as scope:
        weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True,
        initializer=tf.contrib.layers.xavier_initializer())
        
        if use_biases:
            biases = tf.get_variable('biases', [num_out], trainable=True,
                                         initializer=tf.contrib.layers.xavier_initializer())
            act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
        else:
            act = tf.matmul(x, weights)

        if relu == True:
            relu = tf.nn.relu(act)
            return relu
        else:
            return act


def max_pool(x, filter_height, filter_width, stride_y, stride_x,
             name, padding='SAME', verbose_shapes=False):
    """Max pool layer"""
    if verbose_shapes:
        print('X SHAPE maxpool', x.get_shape())

    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0, verbose_shapes=False):
    """Batch normalization"""
    if verbose_shapes:
        print('X SHAPE lrn', x.get_shape())

    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)


def normalize_images(x):
    """Normalize images before feeding into a CNN"""
    return tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), x)


def dropout(x, keep_prob):
    """Dropout layer"""
    return tf.nn.dropout(x, keep_prob)


class AlexNet32(object):
    """AlexNet model"""
    def __init__(self, x, num_classes, reuse=False):
        self.X = x
        self.NUM_CLASSES = num_classes
        self.reuse = reuse
        self.create()

    def create(self):
        self.conv1 = conv(self.X, 5, 5, 64, 1, 1, padding='VALID', name='conv1', reuse=self.reuse)
        norm1 = lrn(self.conv1, 2, 2e-05, 0.75, name='norm1')
        pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')

        self.conv2 = conv(pool1, 5, 5, 64, 1, 1, groups=2, name='conv2', reuse=self.reuse)
        norm2 = lrn(self.conv2, 2, 2e-05, 0.75, name='norm2')
        pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')

        self.flattened = tf.reshape(pool2, [-1, 6 * 6 * 64])
        
        self.fc3 = fc(self.flattened, 6 * 6 * 64, 384, name='fc3', reuse=self.reuse)

        self.fc4 = fc(self.fc3, 384, 192, name='fc4', reuse=self.reuse)

        self.fc5 = fc(self.fc4, 192, self.NUM_CLASSES, relu=False, name='fc5', reuse=self.reuse)


class Composite_model32(object):

    """Visual-semantic embedding"""
    def __init__(self, x, word2vec_size, use_vgg=False, reuse=False):
        self.X = x
        self.WORD2VEC_SIZE = word2vec_size
        self.use_vgg = use_vgg
        self.reuse = reuse

        if self.use_vgg:
            self.image_repr_model = VGG19(self.X, 0.5, 2)
        else:
            self.image_repr_model = AlexNet32(self.X, 2, reuse=self.reuse)
        self.create()

    def create(self):
        if self.use_vgg:
            self.image_repr = self.image_repr_model.fc7
            self.projection_layer = fc(self.image_repr, 4096, self.WORD2VEC_SIZE, name='proj', relu=False,
                                       use_biases=True, reuse=self.reuse)
        else:
            self.image_repr = self.image_repr_model.fc4
            self.projection_layer = fc(self.image_repr, 192, self.WORD2VEC_SIZE, name='proj', relu=False,
                                       use_biases=True, reuse=self.reuse)