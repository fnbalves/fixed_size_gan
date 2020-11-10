import tensorflow as tf
from utils.glove_interface import *


def build_labels_representation_matrix(labels):
    all_repr = []
    word2vec_size = 0
    for label in labels:
        wv = find_word_vec(normalize_label(label))
        if word2vec_size == 0 and wv is not None:
            word2vec_size = max(np.shape(wv))
        all_repr.append(wv)
    return tf.constant(np.array(all_repr), shape=[len(labels), word2vec_size], dtype=tf.float32)


def build_diffs_frobenius(model_output, all_labels_representation):
    num_labels = all_labels_representation.get_shape()[0]
    batch_size = model_output.get_shape()[0]

    labels_representations = tf.split(all_labels_representation, 
                            num_labels, axis=0)
    diffs = []

    for L in labels_representations:
        repeated_L = tf.reshape(tf.stack([L] * batch_size), model_output.get_shape())
        new_diffs = tf.norm(model_output - repeated_L)
        diffs.append(new_diffs)
    diff_tensor = tf.convert_to_tensor(diffs)
    return diff_tensor


def build_diffs_euclidean(model_output, all_labels_representation):
    num_labels = all_labels_representation.get_shape()[0]
    batch_size = model_output.get_shape()[0]
    labels_representations = tf.split(all_labels_representation,
                                      num_labels, axis=0)
    diffs = []

    for L in labels_representations:
        repeated_L = tf.reshape(tf.stack([L] * batch_size), model_output.get_shape())
        new_diffs = mean_norm(model_output, repeated_L)
        diffs.append(new_diffs)
    diff_tensor = tf.convert_to_tensor(diffs)
    return diff_tensor


def build_frobenius_unormalized_tendency_loss(model_output, target_labels, known_labels):
    R = build_labels_representation_matrix(known_labels)
    proj1 = tf.norm(model_output - target_labels)
    proj2 = (-1) * build_diffs_frobenius(model_output, R)
    proj_sum = proj1 + proj2
    proj_mean = tf.reduce_mean(proj_sum)

    final_loss = proj_mean

    return final_loss


def build_diffs_cosine(model_output, all_labels_representation):
    num_labels = all_labels_representation.get_shape()[0]
    batch_size = model_output.get_shape()[0]
    labels_representations = tf.split(all_labels_representation,
                                      num_labels, axis=0)
    diffs = []

    for L in labels_representations:
        repeated_L = tf.reshape(tf.stack([L] * batch_size), model_output.get_shape())
        new_diffs = mean_cosine_distance(model_output, repeated_L)
        diffs.append(new_diffs)
    diff_tensor = tf.convert_to_tensor(diffs)
    return diff_tensor


def mean_norm(m1, m2):
    return tf.reduce_mean(tf.norm(m1 - m2, axis=1))


def mean_cosine_distance(m1, m2):
    normalized_m1 = tf.nn.l2_normalize(m1, 1)
    normalized_m2 = tf.nn.l2_normalize(m2, 1)
    cos_similarity = tf.reduce_sum(tf.multiply(normalized_m1, normalized_m2), axis=1)
    return tf.reduce_mean(1.0 - cos_similarity)


def build_tendency_loss(model_output, target_labels, known_labels):
    R = build_labels_representation_matrix(known_labels)
    proj1 = mean_norm(model_output, target_labels)
    proj2 = (-1) * build_diffs_euclidean(model_output, R)
    proj_sum = proj1 + proj2
    proj_mean = tf.reduce_mean(proj_sum)

    final_loss = proj_mean

    return final_loss


def build_cosine_tendency_loss(model_output, target_labels, known_labels):
    R = build_labels_representation_matrix(known_labels)
    proj1 = mean_cosine_distance(model_output, target_labels)
    proj2 = (-1) * build_diffs_cosine(model_output, R)
    proj_sum = proj1 + proj2
    proj_mean = tf.reduce_mean(proj_sum)

    final_loss = proj_mean

    return final_loss


def build_devise_loss(model_output, target_labels, known_labels, loss_margin=0.1):
    R = build_labels_representation_matrix(known_labels)

    proj1 = tf.diag_part(tf.matmul(model_output, tf.transpose(target_labels)))

    sum1 = loss_margin - proj1
    sum2 = tf.matmul(model_output, tf.transpose(R))

    sum3 = tf.transpose(sum1 + tf.transpose(sum2))

    relu_sum3 = tf.nn.relu(sum3)
    mean = tf.reduce_mean(relu_sum3)

    final_loss = mean

    return final_loss
