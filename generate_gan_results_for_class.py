import tensorflow as tf
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from datetime import datetime
from models_gan import *
from models_zero_shot import Composite_model32
from utils.batch_making import *
from utils.gan import *

batch_size = 30
WORD2VEC_SIZE = 200
gen_dims = 100
noise_den = 1.0
checkpoint_to_load = 'fixed_size_GAN_new_minibatch_checkpoints/model_epoch19.ckpt' #change here the checkpoint to load

tf.reset_default_graph()

real_label = tf.placeholder(tf.float32, shape=[batch_size, WORD2VEC_SIZE], name='image-labels')
gen_input = tf.placeholder(tf.float32, shape=[batch_size, gen_dims], name='z-noise')

generator = DCGAN_generator_conditional(gen_input, real_label, training=False)
gen_images = generator.out

loader = tf.train.Saver()


def create_session():
    sess = tf.Session()
    loader.restore(sess, checkpoint_to_load)
    return sess


def show_images(label, sess, img_name):
    wv = find_word_vec(label)
    labels = [wv] * batch_size

    sample_noise = np.random.uniform(low=-1.0 / noise_den, high=1.0 / noise_den, size=(batch_size, gen_dims))

    gen_samples = sess.run(gen_images, feed_dict={gen_input: sample_noise, real_label: labels})

    plt.close("all")
    view_samples(gen_samples, [label] * batch_size, 6, figsize=(10, 5))
    plt.savefig(img_name, format='svg')
    plt.show(block=False)

    print('Done')

    return gen_samples, labels

session = create_session()
show_images('apple', session, 'apple.svg')
