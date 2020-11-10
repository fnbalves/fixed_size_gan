import pandas as pd
import csv
import numpy as np
import pickle

glove_data_file = 'glove.6B/glove.6B.200d.txt'

print('Loading glove model')
words = pd.read_table(glove_data_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)

norm_mean = 5.5293

composite_words = {
    'pine_tree': 'pine',
    'sweet_pepper': 'pepper',
    'maple_tree': 'maple',
    'aquarium_fish': 'fish',
    'flatfish': 'fish',
    'willow_tree': 'willow',
    'pickup_truck': 'pickup',
    'palm_tree': 'palm',
    'lawn_mower': 'mower',
    'oak_tree': 'oak',
    'streetcar': 'car'
}


def normalize_label(label):
    if label in composite_words:
        return composite_words[label]
    else:
        return label


def find_norm_mean():
    all_words = words.index.values
    count = .0
    norm_sum = .0

    for w in all_words:
        new_norm = np.linalg.norm(words.loc[w].as_matrix())
        norm_sum += new_norm
        count += 1
    norm_sum /= count
    return norm_sum


def find_word_vec(word):
    try:
        return words.loc[word].as_matrix() / norm_mean
    except:
        return None


def cosine_similarity(vect1, vect2):
    return np.dot(vect1, vect2)/(np.linalg.norm(vect1)*np.linalg.norm(vect2))


def cosine_distance(vect1, vect2):
    return 1.0 - cosine_similarity(vect1, vect2)


def find_closest_word(my_word, black_list=None, use_cosine_dist=False):
    num_words = np.shape(words)[0]
    minimun_distance = 1000000
    best_word = 0
    for i in range(num_words):
        if words.iloc[i, :].name in black_list:
                continue
        current_word = words.iloc[i, :] / norm_mean

        if use_cosine_dist:
            current_distance = cosine_distance(current_word, my_word)
        else:
            current_distance = np.linalg.norm(current_word - my_word)

        if current_distance < minimun_distance:
            best_word = i
            minimun_distance = current_distance

    return words.iloc[best_word, :].name


def compute_all_cosine_similarities(word):
    num_words = np.shape(words)[0]
    dist_list = []
    for i in range(num_words):
        current_word_vector = words.iloc[i, :].as_matrix() / norm_mean
        current_word = words.iloc[i, :].name
        current_distance = cosine_similarity(current_word_vector, word)
        dist_list.append([current_word, current_distance])
    return dist_list


def compute_all_distances(word, use_cosine_dist=False):
    num_words = np.shape(words)[0]
    dist_list = []
    for i in range(num_words):
        current_word_vector = words.iloc[i, :].as_matrix() / norm_mean
        current_word = words.iloc[i, :].name
        if use_cosine_dist:
            current_distance = cosine_distance(current_word_vector, word)
        else:
            current_distance = np.linalg.norm(current_word_vector - word)
        dist_list.append([current_word, current_distance])
    return dist_list


def get_closest_words_in_set(vector, possible_words_representation,
                             use_cosine_dist=True):
    all_distances = []

    for label in possible_words_representation:
        wv = find_word_vec(normalize_label(label))
        if use_cosine_dist:
            all_distances.append([label, cosine_distance(vector, wv)])
        else:
            all_distances.append([label, np.linalg.norm(vector - wv)])
    sorted_dist = sorted(all_distances, key=lambda x: x[1])
    return [s[0] for s in sorted_dist]
