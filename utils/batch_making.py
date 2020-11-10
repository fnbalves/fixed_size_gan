from utils.img import *
from utils.glove_interface import *
import numpy as np
import math
import random

random.seed(0)
np.random.seed(0)

NUM_CHANNELS = 3


def adjust_data(image_array, image_size):
    image_matrix = image_array_to_image_matrix(image_array)
    resized_image = resize_image_matrix(image_matrix, image_size, image_size)
    return resized_image


def word2vec_batch(word_batch):
    new_batch = []
    for word in word_batch:
        wv = find_word_vec(normalize_label(word))
        new_batch.append(wv)
    return new_batch


def get_batches(data, size_batch, image_size, word2vec=False,
                send_raw_str=False, should_adjust=True,
                randomize=True, return_raw=False, vectorizer=None):

    if not word2vec and vectorizer is None:
        raise ValueError("You should provide a vectorizer if word2vec=False")

    if randomize:
        random.shuffle(data)
    len_data = len(data)
    num_batches = int(max(math.floor(len_data / size_batch), 1))

    for i in range(num_batches):
        new_batch = data[i * size_batch:min(len_data, (i + 1) * size_batch)]
        images_only = [b[0] for b in new_batch]

        if should_adjust:
            Xs = [adjust_data(b[0], image_size) for b in new_batch]
        else:
            Xs = [b[0] for b in new_batch]
        
        raw_Ys = [b[1] for b in new_batch]
        if not word2vec:
            reshaped_Ys = np.array(raw_Ys).reshape(-1, 1)
            print(reshaped_Ys)
            Ys = vectorizer.transform(reshaped_Ys).todense()
        else:
            Ys = word2vec_batch(raw_Ys)

        if not send_raw_str:
            if not return_raw:
                yield [Xs, Ys]
            else:
                yield [Xs, Ys, images_only]
        else:
            if not return_raw:
                yield [Xs, Ys, raw_Ys]
            else:
                yield [Xs, Ys, raw_Ys, images_only]