import pickle


def normalize_batch(initial_batch):
    normalized_1 = [x/255.0 for x in initial_batch]
    normalized_2 = [2.0*x - 1.0 for x in normalized_1]
    return normalized_2


def print_in_file(string, output_filename):
    output_file = open(output_filename, 'a')
    output_file.write(string + '\n')
    print(string)
    output_file.close()


def save_in_pickle(data, file_name):
    out = open(file_name, 'wb')
    pickle.dump(data, out)
    out.close()


