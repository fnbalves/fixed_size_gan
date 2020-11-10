# Partialy based on https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html
import tensorflow as tf
import numpy as np
import pickle
import math
import os
import random
import matplotlib.pyplot as plt

from utils.file import *
from tqdm import tqdm

folders_to_check = ['multiple_trainings_20_classes',
                    'multiple_trainings_40_classes',
                    'multiple_trainings_60_classes',
                    'multiple_trainings_80_classes']

folders_with_ckpt_my_loss = [[sf for sf in get_subfolders(f) if 'my_loss' in sf]
                             for f in folders_to_check]
checkpoints_to_use_my_loss = [[get_checkpoint_to_use(sf) for sf in f] for f in folders_with_ckpt_my_loss]

folders_with_ckpt_second_loss = [[sf for sf in get_subfolders(f) if 'second_loss' in sf]
                             for f in folders_to_check]
checkpoints_to_use_second_loss = [[get_checkpoint_to_use(sf) for sf in f] for f in folders_with_ckpt_second_loss]


def get_info_for_folder(f):
    zero_shot_infos = os.path.join(f, 'info.pickle')

    zs = pickle.load(open(zero_shot_infos, 'rb'))

    zs_mean = np.mean(list(zs['accuracies_zero_shot'].values()))
    zs_mean_gen = np.mean(list(zs['accuracies_zero_shot_gen'].values()))

    kn_mean = np.mean(list(zs['accuracies_known'].values()))
    kn_mean_gen = np.mean(list(zs['accuracies_known_vs_zero_shot'].values()))

    return zs_mean, zs_mean_gen, kn_mean, kn_mean_gen

zs_acc = []
zs_acc_es = []
zs_acc_ei = []

zs_gen_acc = []
zs_gen_acc_es = []
zs_gen_acc_ei = []

kn_acc = []
kn_acc_es = []
kn_acc_ei = []

kn_g_acc = []
kn_g_acc_es = []
kn_g_acc_ei = []

for i, f in tqdm(enumerate(folders_to_check)):
    tf.reset_default_graph()

    sub_f_my_loss = folders_with_ckpt_my_loss[i]
    sub_f_second_loss = folders_with_ckpt_second_loss[i]

    loss_zs = []
    loss_zs_g = []
    loss_kn = []
    loss_kn_g = []

    for j, sf in enumerate(sub_f_my_loss):
        sf_my_loss = sf
        sf_second_loss = sub_f_second_loss[j]

        zs1, zs_g_1, kn_1, kn_g_1 = get_info_for_folder(sf_my_loss)
        zs2, zs_g_2, kn_2, kn_g_2 = get_info_for_folder(sf_second_loss)

        loss_zs.append([zs1, zs2])
        loss_zs_g.append([zs_g_1, zs_g_2])
        loss_kn.append([kn_1, kn_2])
        loss_kn_g.append([kn_g_1, kn_g_2])

    zs_acc.append([np.mean([x[0] for x in loss_zs]),
                   np.mean([x[1] for x in loss_zs])])
    zs_acc_es.append([np.max([x[0] for x in loss_zs]) -
                      np.mean([x[0] for x in loss_zs]),
                      np.max([x[1] for x in loss_zs]) -
                      np.mean([x[1] for x in loss_zs])])
    zs_acc_ei.append([np.mean([x[0] for x in loss_zs]) -
                      np.min([x[0] for x in loss_zs]),
                      np.mean([x[1] for x in loss_zs]) -
                      np.min([x[1] for x in loss_zs])])

    zs_gen_acc.append([np.mean([x[0] for x in loss_zs_g]),
                   np.mean([x[1] for x in loss_zs_g])])
    zs_gen_acc_es.append([np.max([x[0] for x in loss_zs_g]) -
                      np.mean([x[0] for x in loss_zs_g]),
                      np.max([x[1] for x in loss_zs_g]) -
                      np.mean([x[1] for x in loss_zs_g])])
    zs_gen_acc_ei.append([np.mean([x[0] for x in loss_zs_g]) -
                      np.min([x[0] for x in loss_zs_g]),
                      np.mean([x[1] for x in loss_zs_g]) -
                      np.min([x[1] for x in loss_zs_g])])

    kn_acc.append([np.mean([x[0] for x in loss_kn]),
                       np.mean([x[1] for x in loss_kn])])
    kn_acc_es.append([np.max([x[0] for x in loss_kn]) -
                          np.mean([x[0] for x in loss_kn]),
                          np.max([x[1] for x in loss_kn]) -
                          np.mean([x[1] for x in loss_kn])])
    kn_acc_ei.append([np.mean([x[0] for x in loss_kn]) -
                          np.min([x[0] for x in loss_kn]),
                          np.mean([x[1] for x in loss_kn]) -
                          np.min([x[1] for x in loss_kn])])

    kn_g_acc.append([np.mean([x[0] for x in loss_kn_g]),
                   np.mean([x[1] for x in loss_kn_g])])
    kn_g_acc_es.append([np.max([x[0] for x in loss_kn_g]) -
                      np.mean([x[0] for x in loss_kn_g]),
                      np.max([x[1] for x in loss_kn_g]) -
                      np.mean([x[1] for x in loss_kn_g])])
    kn_g_acc_ei.append([np.mean([x[0] for x in loss_kn_g]) -
                      np.min([x[0] for x in loss_kn_g]),
                      np.mean([x[1] for x in loss_kn_g]) -
                      np.min([x[1] for x in loss_kn_g])])

plt.figure()
plt.xlabel('Número de classes conhecidas')
plt.ylabel('Acurácia top-5 zero shot (%)')
plt.errorbar([20,40,60,80], [100.0*x[0] for x in zs_acc],
             yerr=[[100.0*x[0] for x in zs_acc_es], [100.0*x[0] for x in zs_acc_ei]],
             fmt='-o', capsize=5)
plt.errorbar([20,40,60,80], [100.0*x[1] for x in zs_acc],
             yerr=[[100.0*x[1] for x in zs_acc_es], [100.0*x[1] for x in zs_acc_ei]],
             fmt='-o', capsize=5)
plt.legend(['Função de custo proposta', 'Função de custo DeVISE'])
plt.savefig('zn_acc_2.png')

plt.figure()
plt.xlabel('Número de classes conhecidas')
plt.ylabel('Acurácia top-5 zero shot generalizada (%)')
plt.errorbar([20,40,60,80], [100.0*x[0] for x in zs_gen_acc],
             yerr=[[100.0*x[0] for x in zs_gen_acc_es], [100.0*x[0] for x in zs_gen_acc_ei]],
             fmt='-o',capsize=5)
plt.errorbar([20,40,60,80], [100.0*x[1] for x in zs_gen_acc],
             yerr=[[100.0*x[1] for x in zs_gen_acc_es], [100.0*x[1] for x in zs_gen_acc_ei]],
             fmt='-o',capsize=5)
plt.legend(['Função de custo proposta', 'Função de custo DeVISE'])
plt.savefig('zn_acc_gen_2.png')

plt.show(block=False)

plt.figure()
plt.xlabel('Número de classes conhecidas')
plt.ylabel('Acurácia top-5 nas classes conhecidas (%)')
plt.errorbar([20,40,60,80], [100.0*x[0] for x in kn_acc],
             yerr=[[100.0*x[0] for x in kn_acc_es], [100.0*x[0] for x in kn_acc_ei]],
             fmt='-o',capsize=5)
plt.errorbar([20,40,60,80], [100.0*x[1] for x in kn_acc],
             yerr=[[100.0*x[1] for x in kn_acc_es], [100.0*x[1] for x in kn_acc_ei]],
             fmt='-o',capsize=5)
plt.legend(['Função de custo proposta', 'Função de custo DeVISE'])
plt.savefig('k_acc_2.png')

plt.show(block=False)

plt.figure()
plt.xlabel('Número de classes conhecidas')
plt.ylabel('Acurácia top-5 generalizada nas classes conhecidas (%)')
plt.errorbar([20,40,60,80], [100.0*x[0] for x in kn_g_acc],
             yerr=[[100.0*x[0] for x in kn_g_acc_es], [100.0*x[0] for x in kn_g_acc_ei]],
             fmt='-o',capsize=5)
plt.errorbar([20,40,60,80], [100.0*x[1] for x in kn_g_acc],
             yerr=[[100.0*x[1] for x in kn_g_acc_es], [100.0*x[1] for x in kn_g_acc_ei]],
             fmt='-o',capsize=5)
plt.legend(['Função de custo proposta', 'Função de custo DeVISE'])
plt.savefig('k_g_acc_2.png')

plt.show(block=False)