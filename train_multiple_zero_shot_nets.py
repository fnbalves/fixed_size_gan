# Partialy based on https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html
import tensorflow as tf
import numpy as np
import pickle
import math
import os
import random
from utils.zero_shot_training import *

all_labels = pickle.load(open('pickle_files/all_labels.pickle', 'rb'))

all_data_train = pickle.load(open('pickle_files/all_data_train.pickle', 'rb'))
all_data_test = pickle.load(open('pickle_files/all_data_test.pickle', 'rb'))

random.shuffle(all_data_train)
random.shuffle(all_data_test)

all_data = all_data_train + all_data_test


def separate_labels(all_labels, num_known_labels):
    copy_all_labels = all_labels[:]
    random.shuffle(copy_all_labels)
    known_labels = copy_all_labels[:num_known_labels]
    zeroshot_labels = copy_all_labels[num_known_labels:]
    return known_labels, zeroshot_labels


def create_train_and_validation_sets(data_train, data_validation,
                                     labels):
    train_data = [x for x in data_train if x[1] in labels]
    validation_data = [x for x in data_validation if x[1] in labels]

    return train_data, validation_data

num_trainings = 3
max_epochs = 120

num_known_classes = input('Insert the number of known classes')
num_known_classes = int(num_known_classes)

training_folder = 'multiple_trainings_%d_classes' % num_known_classes
name_first_loss = 'my_loss'
name_second_loss = 'devise_loss'

first_loss = build_tendency_loss
second_loss = build_devise_loss

for i in range(num_trainings):
    print('Model', i, 'of', num_trainings)
    known_labels, zeroshot_labels = separate_labels(all_labels, num_known_classes)
    train_data, validation_data = create_train_and_validation_sets(all_data_train, all_data_test, known_labels)

    checkpoint_first_loss_history_dir = os.path.join(training_folder, 'checkpoints_composite_' + name_first_loss +
                                                  '_%d_history' % (i))
    checkpoint_first_loss_files_dir = os.path.join(training_folder, 'checkpoints_composite_' + name_first_loss +
                                                   '_%d' % (i))
    output_first_loss_file = os.path.join(checkpoint_first_loss_files_dir, 'train_output_' + name_first_loss + '_%d.txt' % (i))
    history_first_loss_file = os.path.join(checkpoint_first_loss_files_dir, 'loss_history_%d.txt' % (i))

    checkpoint_second_loss_history_dir = os.path.join(training_folder, 'checkpoints_composite_' + name_second_loss +
                                            '_%d_history' % (i))
    checkpoint_second_loss_files_dir = os.path.join(training_folder, 'checkpoints_composite_' + name_second_loss +
                                          '_%d' % (i))
    output_second_loss_file = os.path.join(checkpoint_second_loss_files_dir, 'train_output_' + name_second_loss + '_%d.txt' % (i))
    history_second_loss_file = os.path.join(checkpoint_second_loss_files_dir, 'loss_history_%d.txt' % (i))

    session_data_first_loss = train_network(first_loss, train_data,
                                            validation_data,
                                            known_labels,
                                            checkpoint_first_loss_history_dir,
                                            checkpoint_first_loss_files_dir,
                                            output_first_loss_file,
                                            history_first_loss_file,
                                            max_num_epochs=max_epochs)

    results_first_loss_zero_shot = get_results(session_data_first_loss, all_data, zeroshot_labels, zeroshot_labels)
    results_first_loss_gen_zero_shot = get_results(session_data_first_loss, all_data, all_labels, zeroshot_labels)
    results_first_loss_known = get_results(session_data_first_loss, all_data_test, known_labels, known_labels)
    results_first_loss_known_vs_zeroshot = get_results(session_data_first_loss, all_data_test, all_labels, known_labels)

    best_model_first_loss = session_data_first_loss['best_model']
    info_used_first_loss = {'known_labels': known_labels,
                   'zeroshot_labels': zeroshot_labels,
                   'accuracies_zero_shot': results_first_loss_zero_shot,
                   'accuracies_zero_shot_gen': results_first_loss_gen_zero_shot,
                   'accuracies_known': results_first_loss_known,
                   'accuracies_known_vs_zero_shot': results_first_loss_known_vs_zeroshot}

    out_first_loss = open(os.path.join(checkpoint_first_loss_files_dir, 'info.pickle'), 'wb')
    pickle.dump(info_used_first_loss, out_first_loss)
    out_first_loss.close()

    session_data_second_loss = train_network(second_loss, train_data,
                                             validation_data,
                                             known_labels,
                                             checkpoint_second_loss_history_dir,
                                             checkpoint_second_loss_files_dir,
                                             output_second_loss_file,
                                             history_second_loss_file,
                                             max_num_epochs=max_epochs)

    results_second_loss_zero_shot = get_results(session_data_second_loss, all_data, zeroshot_labels, zeroshot_labels)
    results_second_gen_zero_shot = get_results(session_data_second_loss, all_data, all_labels, zeroshot_labels)
    results_second_known = get_results(session_data_second_loss, all_data_test, known_labels, known_labels)
    results_second_loss_known_vs_zeroshot = get_results(session_data_second_loss, all_data_test, all_labels, known_labels)

    best_model_second_loss = session_data_second_loss['best_model']
    info_used_second_loss = {'known_labels': known_labels,
                   'zeroshot_labels': zeroshot_labels,
                   'accuracies_zero_shot': results_second_loss_zero_shot,
                   'accuracies_zero_shot_gen': results_second_gen_zero_shot,
                   'accuracies_known': results_second_known,
                   'accuracies_known_vs_zero_shot': results_second_loss_known_vs_zeroshot
                             }

    out_second_loss = open(os.path.join(checkpoint_second_loss_files_dir, 'info.pickle'), 'wb')
    pickle.dump(info_used_second_loss, out_second_loss)
    out_second_loss.close()

    print('MEAN acc ' + name_first_loss + ' zero shot', np.mean(list(results_first_loss_zero_shot.values())),
          'MEAN acc ' + name_first_loss + ' zero shot gen', np.mean(list(results_first_loss_gen_zero_shot.values())),
          'MEAN acc ' + name_first_loss + ' known labels', np.mean(list(results_first_loss_known.values())),
          'MEAN acc ' + name_first_loss + ' known labels vs zeroshot', np.mean(list(results_first_loss_known_vs_zeroshot.values())),
          'MEAN acc ' + name_second_loss + ' zero shot', np.mean(list(results_second_loss_zero_shot.values())),
          'MEAN acc ' + name_second_loss + ' zero shot gen', np.mean(list(results_second_gen_zero_shot.values())),
          'MEAN acc ' + name_second_loss + ' known labels', np.mean(list(results_second_known.values())),
          'MEAN acc ' + name_second_loss + ' known labels vs zeroshot', np.mean(list(results_second_loss_known_vs_zeroshot.values())))