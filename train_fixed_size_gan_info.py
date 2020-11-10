import tensorflow as tf
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from models_gan import *
from utils.batch_making import *
from datetime import datetime
from utils.zero_shot_loss_functions import *
from models_zero_shot import Composite_model32
from utils.training import *
from utils.gan import *

selected_labels = pickle.load(open('pickle_files/selected_labels.pickle', 'rb'))
data_train = pickle.load(open('pickle_files/all_data_train.pickle', 'rb'))
data_selected = [x for x in data_train if x[1] not in selected_labels]

# Change here if necessary
filewriter_path = 'fixed_size_GAN_info_history/'
checkpoint_path = 'fixed_size_GAN_info_checkpoints/'

lr = 0.0005
beta1 = 0.5
beta2 = 0.999
gen_dims = 100
noise_den = np.sqrt(gen_dims * (0.5 ** 2))

epochs = 20
batch_size = 128
IMAGE_SIZE = 32
WORD2VEC_SIZE = 200
INFO_SIZE = 2

print('Creating reprs')

labels_reprs_1 = [(word, find_word_vec(normalize_label(word)), [0.0, 1.0]) for word in selected_labels]
labels_reprs_0 = [(word, find_word_vec(normalize_label(word)), [1.0, 0.0]) for word in selected_labels]
labels_reprs = labels_reprs_1 + labels_reprs_0

print('Done')

fake_label_batch = [find_word_vec('fake')] * batch_size

samples = []

if not os.path.isdir(filewriter_path): os.mkdir(filewriter_path)
if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)


def sample_c(size_c_vec):
    return np.random.multinomial(1, INFO_SIZE * [1.0 / INFO_SIZE], size=size_c_vec)

def build_loss(model_output, target_labels):
    """Change here which loss function you wish to use"""
    multiplying_term = 10.0
    return tf.sigmoid(multiplying_term * build_tendency_loss(model_output, target_labels, selected_labels))


tf.reset_default_graph()

real_input = tf.placeholder(tf.float32,
                            shape=[batch_size, 32, 32, 3],
                            name='x-images')

real_label = tf.placeholder(tf.float32, shape=[batch_size, WORD2VEC_SIZE], name='image-labels')
fake_label = tf.placeholder(tf.float32, shape=[batch_size, WORD2VEC_SIZE], name='fake-label')
info_input = tf.placeholder(tf.float32, shape=[batch_size, INFO_SIZE], name='info_input')
gen_input = tf.placeholder(tf.float32, shape=[batch_size, gen_dims], name='z-noise')

real_label_test = tf.placeholder(tf.float32, shape=[None, WORD2VEC_SIZE], name='image-labels_test')
gen_input_test = tf.placeholder(tf.float32, shape=[None, gen_dims], name='z-noise_test')
info_input_test = tf.placeholder(tf.float32, shape=[None, INFO_SIZE], name='info_input_test')

generator = DCGAN_generator_conditional_info(gen_input, real_label, info_input)
gen_images = generator.out

info_check = Info_check_network(gen_images, num_info=INFO_SIZE)
info_out = info_check.out

with tf.name_scope('zero_shot_discriminator'):
    zs_model_real = Composite_model32(real_input, WORD2VEC_SIZE, reuse=False)
    zs_model_output_real = zs_model_real.projection_layer

    zs_model_fake = Composite_model32(gen_images, WORD2VEC_SIZE, reuse=True)
    zs_model_output_fake = zs_model_fake.projection_layer

# Multual info loss
cond_ent = tf.reduce_mean(-tf.reduce_sum(tf.log(info_out + 1e-8) * info_input, 1))
ent = tf.reduce_mean(-tf.reduce_sum(tf.log(info_input + 1e-8) * info_input, 1))
# info_check_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
#    labels=info_input,
#    logits=info_out)) #cond_ent + ent
info_check_loss = cond_ent + ent

# generator loss
gen_loss = build_loss(zs_model_output_fake, real_label)
gen_loss += info_check_loss

# discriminator loss
disc_loss_real_images = build_loss(zs_model_output_real, real_label)
disc_loss_gen_images = build_loss(zs_model_output_fake, fake_label)

disc_loss = disc_loss_real_images + disc_loss_gen_images

print('TRAINABLE VARS', tf.trainable_variables())
# get the variables for the generator and discriminator
generator_variables = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
discriminator_variables = [var for var in tf.trainable_variables() if
                           not var.name.startswith('generator') and ('info_check' not in var.name)]
info_check_variables = [var for var in tf.trainable_variables() if 'info_check' in var.name]

print('---------------------------------')
print('Generator variables', generator_variables)
print('---------------------------------')
print('Discriminator variables', discriminator_variables)
print('---------------------------------')
print('Info check variables', info_check_variables)
print('---------------------------------')

# setup the optimizers
# comtrol for the global sample mean and variance
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    generator_optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1, beta2=beta2).minimize(gen_loss,
                                                                                                      var_list=generator_variables + info_check_variables)
    discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1, beta2=beta2).minimize(disc_loss,
                                                                                                          var_list=discriminator_variables)

saver = tf.train.Saver()

with tf.Session() as sess:
    # initialize the variables
    sess.run(tf.global_variables_initializer())

    current_c = None

    # train the network
    for epoch in range(epochs):
        print('Current epoch', epoch)
        train_generator = get_batches(data_selected, batch_size, IMAGE_SIZE, word2vec=True)

        for batch_xs, batch_ys in train_generator:
            normalized_batch = normalize_batch(batch_xs)
            # generate the noise
            noise = np.random.uniform(low=-1.0 / noise_den, high=1.0 / noise_den, size=(batch_size, gen_dims))
            current_c = sample_c(batch_size)

            # feed the noise through the generator
            sess.run(generator_optimizer, feed_dict={gen_input: noise, real_input: normalized_batch,
                                                     real_label: batch_ys, fake_label: fake_label_batch,
                                                     info_input: current_c})

            # feed the channel and the noise to the discriminator
            sess.run(discriminator_optimizer, feed_dict={gen_input: noise, real_input: normalized_batch,
                                                         real_label: batch_ys, fake_label: fake_label_batch,
                                                         info_input: current_c})

        # sample more noise
        sample_noise = np.random.uniform(low=-1.0 / noise_den, high=1.0 / noise_den,
                                             size=(len(labels_reprs), gen_dims))

        repr_vecs = [x[1] for x in labels_reprs]
        repr_texts = [x[0] for x in labels_reprs]
        repr_cs = [x[2] for x in labels_reprs]

        # generate images
        generator_test = DCGAN_generator_conditional_info(gen_input_test, real_label_test, info_input_test,
                                                              reuse=True, training=False)
        gen_images_test = generator_test.out
        gen_samples = sess.run(gen_images_test, feed_dict={gen_input_test: sample_noise,
                                                               real_label_test: repr_vecs, info_input_test: repr_cs})

        plt.close("all")
        view_samples(gen_samples, repr_texts, 6, figsize=(10, 5))
        plt.savefig(os.path.join(checkpoint_path, "epoch%d.svg" % (epoch)), format='svg')
        print('Sample figure generated')

        checkpoint_name = os.path.join(checkpoint_path, 'model_epoch' + str(epoch) + '.ckpt')
        save_path = saver.save(sess, checkpoint_name)

        print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
