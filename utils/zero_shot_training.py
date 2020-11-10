# Partialy based on https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html
# Code to train the visual-semantic model
import tensorflow as tf
import numpy as np
import pickle
import math
import os
import time
from matplotlib.font_manager import FontProperties

from datetime import datetime
from models_zero_shot import Composite_model32
from utils.batch_making import *
from utils.zero_shot_loss_functions import *
from sklearn.manifold import TSNE
from matplotlib import cm
from utils.training import *


def train_network(loss_function, train_data, test_data,
                  known_labels, filewriter_path, checkpoint_path, log_file_name,
                  numeric_history_file_name,
                  learning_rate=0.01, momentum=0.9,
                  max_num_epochs=300, batch_size=128, word2vec_size=200,
                  image_size=32, max_violate=3):

    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, [batch_size, image_size, image_size, 3])
    y = tf.placeholder(tf.float32, [batch_size, word2vec_size])

    model = Composite_model32(x, word2vec_size)
    model_output = model.projection_layer

    var_list = [v for v in tf.trainable_variables()]

    with tf.name_scope("loss"):
        loss = loss_function(model_output, y, known_labels)

    with tf.name_scope('train'):
        gradients = tf.gradients(loss, var_list)
        gradients = list(zip(gradients, var_list))
        print('GRADIENTS', gradients)
        for g in gradients:
            try:
                print(g[1])
                print(g[0].get_shape())
                print('-------')
            except:
                pass
        global_step = tf.Variable(0)

        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
        train_op = optimizer.apply_gradients(grads_and_vars=gradients, global_step=global_step)

        saver = tf.train.Saver ()

    if not os.path.isdir(filewriter_path): os.mkdir(filewriter_path)
    if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print_in_file("{} Start training...".format(datetime.now()), log_file_name)
    print_in_file("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                              filewriter_path), log_file_name)

    train_generator = get_batches(train_data, batch_size, image_size, word2vec=True)
    val_generator = get_batches(test_data, batch_size, image_size, word2vec=True)

    smallest_loss = 100000000.0
    num_violate = 0
    best_model = ''
    loss_history = []

    for epoch in range(max_num_epochs):

        print_in_file("{} Epoch number: {}".format(datetime.now(), epoch + 1), log_file_name)
        checkpoint_name = os.path.join(checkpoint_path, 'model_epoch' + str(epoch) + '.ckpt')
        times_batches = []

        for batch_xs, batch_ys in train_generator:
            normalized_batch = normalize_batch(batch_xs)

            start = time.time()

            sess.run(train_op, feed_dict={x: normalized_batch,
                                          y: batch_ys})

            end = time.time()

            difference = float(end - start)
            print(difference)
            times_batches.append(difference)

        mean_processing_time = np.mean(times_batches)
        print('Mean time per batch', mean_processing_time)
        out = open(checkpoint_name + '_mt.pickle', 'wb')
        pickle.dump(mean_processing_time, out)
        out.close()

        print_in_file("{} Start validation".format(datetime.now()), log_file_name)
        test_loss = 0.
        test_count = 0

        for batch_tx, batch_ty in val_generator:
            normalized_batch = normalize_batch(batch_tx)
            new_loss = sess.run(loss, feed_dict={x: normalized_batch,
                                                 y: batch_ty})
            if math.isnan(new_loss):
                print('Loss has NaN')
            test_loss += new_loss
            test_count += 1

        test_loss /= test_count

        print_in_file("Validation Loss = %s %.4f" % (datetime.now(), test_loss), log_file_name)

        loss_history.append(test_loss)
        save_in_pickle(loss_history, numeric_history_file_name)

        if test_loss < smallest_loss:
            smallest_loss = test_loss
            num_violate = 0
            best_model = checkpoint_name

        else:
            num_violate += 1

            if num_violate >= max_violate:
                print_in_file("Finishing training", log_file_name)
                break

        # Reset the file pointer of the image data generator
        train_generator = get_batches(train_data, batch_size, image_size, word2vec=True)
        val_generator = get_batches(test_data, batch_size, image_size, word2vec=True)

        print_in_file("{} Saving checkpoint of model...".format(datetime.now()), log_file_name)

        # save checkpoint of the model
        save_path = saver.save(sess, checkpoint_name)

        print_in_file("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name), log_file_name)

    session_data = {
        'session': sess,
        'x': x,
        'y': y,
        'model': model,
        'model_output': model_output,
        'numeric_history': loss_history,
        'best_model': best_model
    }

    return session_data


def create_session_data_from_checkpoint(checkpoint,  batch_size=128, word2vec_size=200,
                  image_size=32):

    x = tf.placeholder(tf.float32, [batch_size, image_size, image_size, 3])

    model = Composite_model32(x, word2vec_size)
    model_output = model.projection_layer

    sess = tf.Session()

    session_data = {
        'session': sess,
        'x': x,
        'model': model,
        'model_output': model_output,
        'best_model': checkpoint
    }

    return session_data


def get_results(session_data, all_data, possible_labels_on_inference, zeroshot_labels,
                batch_size=128, image_size=32, use_cosine_dist=True):

    data_to_use = [x for x in all_data if x[1] in zeroshot_labels]

    data_generator = get_batches(data_to_use, batch_size, image_size, word2vec=True,
                                 send_raw_str=True)

    points = {}

    for label in zeroshot_labels:
        points[normalize_label(label)] = []

    sess = session_data['session']
    best_model = session_data['best_model']
    model_output = session_data['model_output']
    x = session_data['x']

    restorer = tf.train.Saver()
    restorer.restore(sess, best_model)

    accuracies = {}

    for batch_x, batch_y, batch_labels in data_generator:
        normalized_batch = normalize_batch(batch_x)
        output = sess.run(model_output, {x: normalized_batch})
        for i, o in enumerate(output):
            closest_words = get_closest_words_in_set(o, possible_labels_on_inference,
                                                     use_cosine_dist=use_cosine_dist)[:5]
            correct_label = batch_labels[i]

            if correct_label not in accuracies:
                if correct_label in closest_words:
                    accuracies[correct_label] = [1.0, 1.0]
                else:
                    accuracies[correct_label] = [0.0, 1.0]
            else:
                if correct_label in closest_words:
                    accuracies[correct_label][0] = accuracies[correct_label][0] + 1.0
                accuracies[correct_label][1] = accuracies[correct_label][1] + 1.0

    for key in accuracies.keys():
        accuracies[key] = accuracies[key][0] / accuracies[key][1]

    print('Mean accuracy', np.mean(list(accuracies.values())))

    return accuracies


def make_tsne(session_data, data, labels,
              fig_filename='tsne.svg',
              batch_size=128, image_size=32):
    data_to_use = [x for x in data if x[1] in labels]

    data_generator = get_batches(data_to_use, batch_size, image_size, word2vec=True,
                                 send_raw_str=True)

    sess = session_data['session']
    best_model = session_data['best_model']
    model_output = session_data['model_output']
    x = session_data['x']

    restorer = tf.train.Saver()
    restorer.restore(sess, best_model)

    points = []
    p_labels = []

    for batch_x, batch_y, batch_labels in data_generator:
        output = sess.run(model_output, {x: batch_x})
        for i, o in enumerate(output):
            label = normalize_label(batch_labels[i])
            points.append(o)
            p_labels.append(label)

    labels = [normalize_label(L) for L in labels]
    label_points = [find_word_vec(L) for L in labels]

    for i, L in enumerate(labels):
        points.append(label_points[i])
        p_labels.append('LABEL-' + normalize_label(L))

    print('RUNNING TSNE')
    manifold = TSNE(n_components=2, metric='cosine').fit_transform(points)
    print('DONE')

    output_points = [a for i, a in enumerate(manifold) if 'LABEL' not in p_labels[i]]
    class_points = [[a, p_labels[i]] for i, a in enumerate(manifold) if 'LABEL' in p_labels[i]]
    output_labels = [L for L in p_labels if 'LABEL' not in L]
    output_labels = [labels.index(normalize_label(L)) for L in output_labels]

    x_output = [a[0] for a in output_points]
    y_output = [a[1] for a in output_points]
    x_class_points = [a[0][0] for a in class_points]
    y_class_points = [a[0][1] for a in class_points]
    l_class_points = [a[1].split('-')[1] for a in class_points]

    fig, ax = plt.subplots()
    num_colors = len(labels)
    cm = plt.get_cmap('rainbow')
    ax.set_color_cycle([cm(1. * i / num_colors) for i in range(num_colors)])

    for c in range(len(labels)):
        x_L = [x for i, x in enumerate(x_output) if output_labels[i] == c]
        y_L = [y for i, y in enumerate(y_output) if output_labels[i] == c]
        ax.scatter(x_L, y_L, marker='x', label=labels[c])

    lgd = ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))

    font0 = FontProperties()
    font0.set_weight('bold')
    font0.set_size(7)

    for i, L in enumerate(l_class_points):
        ax.text(x_class_points[i], y_class_points[i], L,
                fontproperties=font0, bbox=dict(facecolor='white',
                                                edgecolor='black',
                                                alpha=0.8))

    ax.set_xlim([-50, 50])

    fig.savefig(fig_filename, bbox_extra_artists=(lgd,), bbox_inches='tight', format='svg')
    plt.show(block=False)