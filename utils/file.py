from os import listdir
from os.path import isfile, join


def get_subfolders(folder):
    folder_only = [f for f in listdir(folder) if not isfile(join(folder, f))]
    non_history = [join(folder, f) for f in folder_only if 'history' not in f]
    return non_history


def get_number_from_string(string):
    numbers = [n for n in string if n.isdigit()]
    return int(''.join(numbers))


def get_checkpoint_to_use(folder, name_only=True):
    files_only = [f for f in listdir(folder) if isfile(join(folder, f))]
    ckpts_only = [f for f in files_only if '.ckpt' in f]
    numbers = [get_number_from_string(f) for f in ckpts_only]
    min_number = min(numbers)
    return join(folder, 'model_epoch' + str(min_number) + '.ckpt')