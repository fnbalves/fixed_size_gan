from utils.zero_shot_training import *
from matplotlib.font_manager import FontProperties

all_labels = pickle.load(open('pickle_files/all_labels.pickle', 'rb'))

all_data_train = pickle.load(open('pickle_files/all_data_train.pickle'))
all_data_test = pickle.load(open('pickle_files/all_data_test.pickle'))


def create_accuracy_histogram(accuracies_dict, title, filename_to_save):
    plt.figure()
    keys = list(accuracies_dict.keys())
    vals = [[k, accuracies_dict[k]] for k in keys]
    sorted_vals = sorted(vals, reverse=True, key=lambda x: x[1])
    plt.bar(x=range(len(sorted_vals)), height=[100.0*x[1] for x in sorted_vals])

    plt.ylabel('Acurácia top-5 ' + title + ' (%)')
    plt.xticks(range(len(sorted_vals)),
               [normalize_label(x[0]) for x in sorted_vals],
               rotation=90)
    plt.gcf().subplots_adjust(bottom=0.25)
    plt.savefig(filename_to_save)
    plt.show(block=False)

#change here
s = create_session_data_from_checkpoint('multiple_trainings_80_classes/checkpoints_composite_my_loss_0/model_epoch16.ckpt')
i = pickle.load(open('multiple_trainings_80_classes/checkpoints_composite_my_loss_0/info_used_0.txt', 'rb'))
make_tsne(s, all_data_test, i['zeroshot_labels'], fig_filename='tsne_second_loss_zs.svg')
create_accuracy_histogram(i['accuracies_zero_shot'], 'zero shot', 'zero_shot_accuracy.svg')