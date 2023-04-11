import os
import logging
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn.functional as F
import random
import copy
from collections import OrderedDict, defaultdict
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from datasets import MNIST_truncated, CIFAR10_truncated, CIFAR100_truncated, SVHN_custom, FashionMNIST_truncated, CustomTensorDataset, CelebA_custom, FEMNIST, Generated, genData, CharacterDataset, SubFEMNIST
from math import sqrt

import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import torch.optim as optim
import torchvision.utils as vutils
import time
import random

from models.mnist_model import Generator, Discriminator, DHead, QHead
from config import params
import sklearn.datasets as sk
from sklearn.datasets import load_svmlight_file
from constants import *
from datastore import *

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass


def distance_calculate(x, y, mode, device, sigma=None, sigma_matrix=None):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)
    if mode == 'Eulidean':
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        return torch.sqrt(torch.pow(x - y, 2).sum(2))
    elif mode == 'Mahalanobis':
        # dist1 = torch.zeros(n,m).to(device)
        # for inst in range(n):
        #     feature = x[inst,:]
        #     for proto in range(m):
        #         class_proto = y[proto,:]
        #         # print("class_proto: ", class_proto)
        #         sigma_proto = sigma[proto]
        #         # print("index: ", proto)
        #         print("sigma: ", sigma_proto)
        #         # sigma_proto2 = torch.eye(128).to(device)
        #         print("sigma_matrix: ", sigma_matix[proto])
        #         part1 = feature - class_proto
        #         # print("part1: ", part1)
        #         dix = torch.mm(part1.view(1,-1), sigma_proto)
        #         # print("dix: ",dix)
        #         m_dix = torch.mm(dix, part1.view(-1,1))
        #         # print("m_dix: ", m_dix)
        #         dist1[inst,proto] = m_dix.data
        #     assert False
        # print("dist1: ", dist1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        sigma_matrix = sigma_matrix.unsqueeze(0).expand(n, m, d, d)
        dif = x-y
        dif1 = dif.unsqueeze(2)

        # dif = rearrange(dif, 'n m d -> (n m) d').unsqueeze(1)
        # sigma_matrix1 = rearrange(sigma_matrix, 'n m d1 d2 -> (n m) d1 d2')
        # part1 = torch.bmm(dif, sigma_matrix1).squeeze(1)
        # # print("version2: ",part1)
        # part1 = rearrange(part1, '(n m) d -> n m d', n=n)
        # dif = rearrange(dif.squeeze(1), '(n m) d -> n m d', n=n)
        # dist2 = torch.mul(part1, dif).sum(2)
        # print("dist2: ",dist2)

        part1 = torch.einsum('ijkl,ijlp->ijkp',[dif1, sigma_matrix]).squeeze(2)
        # print("version3: ",part1)
        dif = dif1.squeeze(2)
        dist3 = torch.einsum('ijk,ijk->ijk',[part1, dif]).sum(2)
        # print("dist3: ",dist3)
        # print(torch.sqrt(dist3))
        # assert False
        return torch.sqrt(dist3)


def get_form(model):
    tmpm = []
    tmpv = []
    for name in model.state_dict().keys():
        if 'running_mean' in name:
            tmpm.append(model.state_dict()[name].detach().to('cpu').numpy())
        if 'running_var' in name:
            tmpv.append(model.state_dict()[name].detach().to('cpu').numpy())
    return tmpm, tmpv


def get_wasserstein(m1, v1, m2, v2, mode='nosquare'):
    w = 0
    bl = len(m1)
    for i in range(bl):
        tw = 0
        tw += (np.sum(np.square(m1[i]-m2[i])))
        tw += (np.sum(np.square(np.sqrt(v1[i]) - np.sqrt(v2[i]))))
        if mode == 'square':
            w += tw
        else:
            w += math.sqrt(tw)
    return w


def get_weight_matrix1(bnmlist, bnvlist):
    model_momentum = 0.1
    client_num = len(bnmlist)
    weight_m = np.zeros((client_num, client_num))
    for i in range(client_num):
        for j in range(client_num):
            if i == j:
                weight_m[i, j] = 0
            else:
                tmp = get_wasserstein(
                    bnmlist[i], bnvlist[i], bnmlist[j], bnvlist[j])
                if tmp == 0:
                    weight_m[i, j] = 100000000000000
                else:
                    weight_m[i, j] = 1/tmp
    weight_s = np.sum(weight_m, axis=1)
    weight_s = np.repeat(weight_s, client_num).reshape(
        (client_num, client_num))
    weight_m = (weight_m / weight_s)*(1 - model_momentum)
    for i in range(client_num):
        weight_m[i, i] = model_momentum
    return weight_m


def set_client_weight(dataidxs_train_map, dataidxs_test_map, g_model, bn_list, device, args):
    bnmlist1, bnvlist1 = [], []
    n_clients = len(bn_list)
    for net_id in range(n_clients):
        model = copy.deepcopy(g_model)
        model.load_state_dict(bn_list[net_id], strict=False)
        model.eval()
        avgmeta = metacount(get_form(model)[0])
        dataidxs_train = dataidxs_train_map[net_id]
        dataidxs_test = dataidxs_test_map[net_id]
        train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 2*args.batch_size, dataidxs_train, dataidxs_test, 0, net_id, args.n_parties-1, drop_last=True)
        with torch.no_grad():
            for data, _ in train_dl_local:
                data = data.to(device).float()
                fea = model.getallfea(data)
                nl = len(data)
                tm, tv = [], []
                for item in fea:
                    if len(item.shape) == 4:
                        tm.append(torch.mean(
                            item, dim=[0, 2, 3]).detach().to('cpu').numpy())
                        tv.append(
                            torch.var(item, dim=[0, 2, 3]).detach().to('cpu').numpy())
                    else:
                        tm.append(torch.mean(
                            item, dim=0).detach().to('cpu').numpy())
                        tv.append(
                            torch.var(item, dim=0).detach().to('cpu').numpy())
                avgmeta.update(nl, tm, tv)
        bnmlist1.append(avgmeta.getmean())
        bnvlist1.append(avgmeta.getvar())
    weight_m = get_weight_matrix1(bnmlist1, bnvlist1)
    return weight_m


class metacount(object):
    def __init__(self, numpyform):
        super(metacount, self).__init__()
        self.count = 0
        self.mean = []
        self.var = []
        self.bl = len(numpyform)
        for i in range(self.bl):
            self.mean.append(np.zeros(len(numpyform[i])))
            self.var.append(np.zeros(len(numpyform[i])))

    def update(self, m, tm, tv):
        tmpcount = self.count+m
        for i in range(self.bl):
            tmpm = (self.mean[i]*self.count + tm[i]*m)/tmpcount
            self.var[i] = (self.count*(self.var[i]+np.square(tmpm -
                           self.mean[i])) + m*(tv[i]+np.square(tmpm-tm[i])))/tmpcount
            self.mean[i] = tmpm
        self.count = tmpcount

    def getmean(self):
        return self.mean

    def getvar(self):
        return self.var


def euclidean_dist( x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.sqrt(torch.pow(x - y, 2).sum(2))


def load_cifar10_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)


def load_cifar100_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar100_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=transform)
    cifar100_test_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target

    return (X_train, y_train, X_test, y_test)


def record_net_data_stats(y_train, net_dataidx_map, logdir=None):

    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    if logdir != None:
        logger.info('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts


def renormalize(weights, index):
    """
    :param weights: vector of non negative weights summing to 1.
    :type weights: numpy.array
    :param index: index of the weight to remove
    :type index: int
    """
    renormalized_weights = np.delete(weights, index)
    renormalized_weights /= renormalized_weights.sum()

    return renormalized_weights


def partition_data(dataset, datadir, partition, n_parties, beta=0.4, logdir=None):
    #np.random.seed(2020)
    #torch.manual_seed(2020)

    if dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
        y = np.concatenate([y_train, y_test], axis=0)
    elif dataset == 'cifar100':
        X_train, y_train, X_test, y_test = load_cifar100_data(datadir)
        y = np.concatenate([y_train, y_test], axis=0)

    n_train = y_train.shape[0]
    n_test = y_test.shape[0]

    if partition == "homo":
        idxs_train = np.random.permutation(n_train)
        idxs_test = np.random.permutation(n_test)

        # print(idxs_test)

        batch_idxs_train = np.array_split(idxs_train, n_parties)
        batch_idxs_test = np.array_split(idxs_test, n_parties)
        # print(batch_idxs_test)
        
        net_dataidx_map_train = {i: batch_idxs_train[i] for i in range(n_parties)}
        net_dataidx_map_test = {i: batch_idxs_test[i] for i in range(n_parties)}
        # print(net_dataidx_map_test)
        # assert False

    elif partition == "2-cluster":
        if dataset == "cifar10":
            num = 5
            k = 10
        elif dataset == "cifar100":
            num = 50
            k = 100

        # -------------------------- #
        # Create class index mapping #
        # -------------------------- #
        data_class_idx_train_part1 = {i: np.where(y_train == i)[0] for i in range(int(k/2))}
        data_class_idx_test_part1 = {i: np.where(y_test == i)[0] for i in range(int(k/2))}

        data_class_idx_train_part2 = {i: np.where(y_train == i)[0] for i in range(int(k/2), k)}
        data_class_idx_test_part2 = {i: np.where(y_test == i)[0] for i in range(int(k/2), k)}

        num_samples_train = {i: len(data_class_idx_train_part1[i]) for i in range(int(k/2))}.update({i: len(data_class_idx_train_part2[i]) for i in range(int(k/2), k)})
        num_samples_test = {i: len(data_class_idx_test_part1[i]) for i in range(int(k/2))}.update({i: len(data_class_idx_test_part2[i]) for i in range(int(k/2), k)})

        # --------- #
        # Shuffling #
        # --------- #
        idxs_train_part1 = []
        idxs_train_part2 = []
        idxs_test_part1 = []
        idxs_test_part2 = []

        for data_idx in data_class_idx_train_part1.values():
            idxs_train_part1.extend(data_idx)
        for data_idx in data_class_idx_test_part1.values():
            idxs_test_part1.extend(data_idx)
        for data_idx in data_class_idx_train_part2.values():
            idxs_train_part2.extend(data_idx)
        for data_idx in data_class_idx_test_part2.values():
            idxs_test_part2.extend(data_idx)

        random.shuffle(idxs_train_part1)
        random.shuffle(idxs_train_part2)
        random.shuffle(idxs_test_part1)
        random.shuffle(idxs_test_part2)

        batch_idxs_train_part1 = np.array_split(idxs_train_part1, int(n_parties/2))
        batch_idxs_test_part1 = np.array_split(idxs_test_part1, int(n_parties/2))
        batch_idxs_train_part1 = [list(x) for x in batch_idxs_train_part1]
        batch_idxs_test_part1 = [list(x) for x in batch_idxs_test_part1]

        batch_idxs_train_part2 = np.array_split(idxs_train_part2, int(n_parties/2))
        batch_idxs_test_part2 = np.array_split(idxs_test_part2, int(n_parties/2))
        batch_idxs_train_part2 = [list(x) for x in batch_idxs_train_part2]
        batch_idxs_test_part2 = [list(x) for x in batch_idxs_test_part2]
        
        net_dataidx_map_train = {i: batch_idxs_train_part1[i] for i in range(int(n_parties/2))}
        net_dataidx_map_test = {i: batch_idxs_test_part1[i] for i in range(int(n_parties/2))}

        net_dataidx_map_train2 = {i+int(n_parties/2): batch_idxs_train_part2[i] for i in range(int(n_parties/2))}
        net_dataidx_map_test2 = {i+int(n_parties/2): batch_idxs_test_part2[i] for i in range(int(n_parties/2))}

        net_dataidx_map_train.update(net_dataidx_map_train2)
        net_dataidx_map_test.update(net_dataidx_map_test2)

       	# print(net_dataidx_map_test.keys())

    elif partition == "noniid-labeldir":
        min_size = 0
        min_require_size = 10
        if dataset == 'cifar10':
            K = 10
        elif dataset == "cifar100":
            K = 100
        elif dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            K = 2
            # min_require_size = 100
        else:
            assert False
            print("Choose Dataset in readme.")

        N_train = y_train.shape[0]
        N_test = y_test.shape[0]
        # print("train: ",y_train.shape[0])
        # print("test: ",y_test.shape[0])
        #np.random.seed(2020)
        net_dataidx_map_train = {}
        net_dataidx_map_test = {}

        while min_size < min_require_size:
            idx_batch_train = [[] for _ in range(n_parties)]
            idx_batch_test = [[] for _ in range(n_parties)]
            for k in range(K):
                train_idx_k = np.where(y_train == k)[0]
                test_idx_k = np.where(y_test == k)[0]

                np.random.shuffle(train_idx_k)
                np.random.shuffle(test_idx_k)

                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                # logger.info("proportions1: ", proportions)
                # logger.info("sum pro1:", np.sum(proportions))
                # print("1.: ",proportions)
                ## Balance
                proportions = np.array([p * (len(idx_j) < N_train / n_parties) for p, idx_j in zip(proportions, idx_batch_train)])
                # logger.info("proportions2: ", proportions)
                # print("2.: ",proportions)
                proportions = proportions / proportions.sum()
                # logger.info("proportions3: ", proportions)
                proportions_train = (np.cumsum(proportions) * len(train_idx_k)).astype(int)[:-1]
                proportions_test = (np.cumsum(proportions) * len(test_idx_k)).astype(int)[:-1]
                # print("3.: ",proportions)
                # logger.info("proportions4: ", proportions)
                idx_batch_train = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_train, np.split(train_idx_k, proportions_train))]
                idx_batch_test = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_test, np.split(test_idx_k, proportions_test))]
                
                min_size_train = min([len(idx_j) for idx_j in idx_batch_train])
                min_size_test = min([len(idx_j) for idx_j in idx_batch_test])
                min_size = min(min_size_train, min_size_test)

        for j in range(n_parties):
            np.random.shuffle(idx_batch_train[j])
            np.random.shuffle(idx_batch_test[j])
            net_dataidx_map_train[j] = idx_batch_train[j]
            net_dataidx_map_test[j] = idx_batch_test[j]

    elif partition == "iid-label100":
        seed = 12345
        n_fine_labels = 100
        n_coarse_labels = 20
        coarse_labels = \
            np.array([
                4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                18, 1, 2, 15, 6, 0, 17, 8, 14, 13
            ])
        rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
        rng = random.Random(rng_seed)
        np.random.seed(rng_seed)

        n_samples_train = y_train.shape[0]
        n_samples_test = y_test.shape[0]

        selected_indices_train = rng.sample(list(range(n_samples_train)), n_samples_train)
        selected_indices_test = rng.sample(list(range(n_samples_test)), n_samples_test)

        n_samples_by_client_train = int((n_samples_train / n_parties) // 5)
        n_samples_by_client_test = int((n_samples_test / n_parties) // 5)

        indices_by_fine_labels_train = {k: list() for k in range(n_fine_labels)}
        indices_by_coarse_labels_train = {k: list() for k in range(n_coarse_labels)}

        indices_by_fine_labels_test = {k: list() for k in range(n_fine_labels)}
        indices_by_coarse_labels_test = {k: list() for k in range(n_coarse_labels)}

        for idx in selected_indices_train:
            fine_label = y_train[idx]
            coarse_label = coarse_labels[fine_label]

            indices_by_fine_labels_train[fine_label].append(idx)
            indices_by_coarse_labels_train[coarse_label].append(idx)

        for idx in selected_indices_test:
            fine_label = y_test[idx]
            coarse_label = coarse_labels[fine_label]

            indices_by_fine_labels_test[fine_label].append(idx)
            indices_by_coarse_labels_test[coarse_label].append(idx)

        fine_labels_by_coarse_labels = {k: list() for k in range(n_coarse_labels)}

        for fine_label, coarse_label in enumerate(coarse_labels):
            fine_labels_by_coarse_labels[coarse_label].append(fine_label)

        net_dataidx_map_train = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
        net_dataidx_map_test = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}

        for client_idx in range(n_parties):
            coarse_idx = client_idx // 5
            fine_idx = fine_labels_by_coarse_labels[coarse_idx]
            for k in range(5):
                fine_label = fine_idx[k]
                sample_idx = rng.sample(list(indices_by_fine_labels_train[fine_label]), n_samples_by_client_train)
                net_dataidx_map_train[client_idx] = np.append(net_dataidx_map_train[client_idx], sample_idx)
                for idx in sample_idx:
                    indices_by_fine_labels_train[fine_label].remove(idx)

        for client_idx in range(n_parties):
            coarse_idx = client_idx // 5
            fine_idx = fine_labels_by_coarse_labels[coarse_idx]
            for k in range(5):
                fine_label = fine_idx[k]
                sample_idx = rng.sample(list(indices_by_fine_labels_test[fine_label]), n_samples_by_client_test)
                net_dataidx_map_test[client_idx] = np.append(net_dataidx_map_test[client_idx], sample_idx)
                for idx in sample_idx:
                    indices_by_fine_labels_test[fine_label].remove(idx)

    elif partition == "noniid-labeldir100":
        seed = 12345
        alpha = 10
        n_fine_labels = 100
        n_coarse_labels = 20
        coarse_labels = \
            np.array([
                4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                18, 1, 2, 15, 6, 0, 17, 8, 14, 13
            ])

        rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
        rng = random.Random(rng_seed)
        np.random.seed(rng_seed)

        n_samples = y.shape[0]

        selected_indices = rng.sample(list(range(n_samples)), n_samples)

        n_samples_by_client = n_samples // n_parties

        indices_by_fine_labels = {k: list() for k in range(n_fine_labels)}
        indices_by_coarse_labels = {k: list() for k in range(n_coarse_labels)}

        for idx in selected_indices:
            fine_label = y[idx]
            coarse_label = coarse_labels[fine_label]

            indices_by_fine_labels[fine_label].append(idx)
            indices_by_coarse_labels[coarse_label].append(idx)

        available_coarse_labels = [ii for ii in range(n_coarse_labels)]

        fine_labels_by_coarse_labels = {k: list() for k in range(n_coarse_labels)}

        for fine_label, coarse_label in enumerate(coarse_labels):
            fine_labels_by_coarse_labels[coarse_label].append(fine_label)

        net_dataidx_map = [[] for i in range(n_parties)]

        for client_idx in range(n_parties):
            coarse_labels_weights = np.random.dirichlet(alpha=beta * np.ones(len(fine_labels_by_coarse_labels)))
            weights_by_coarse_labels = dict()

            for coarse_label, fine_labels in fine_labels_by_coarse_labels.items():
                weights_by_coarse_labels[coarse_label] = np.random.dirichlet(alpha=alpha * np.ones(len(fine_labels)))

            for ii in range(n_samples_by_client):
                coarse_label_idx = int(np.argmax(np.random.multinomial(1, coarse_labels_weights)))
                coarse_label = available_coarse_labels[coarse_label_idx]
                fine_label_idx = int(np.argmax(np.random.multinomial(1, weights_by_coarse_labels[coarse_label])))
                fine_label = fine_labels_by_coarse_labels[coarse_label][fine_label_idx]
                sample_idx = int(rng.choice(list(indices_by_fine_labels[fine_label])))

                net_dataidx_map[client_idx] = np.append(net_dataidx_map[client_idx], sample_idx)

                indices_by_fine_labels[fine_label].remove(sample_idx)
                indices_by_coarse_labels[coarse_label].remove(sample_idx)


                if len(indices_by_fine_labels[fine_label]) == 0:
                    fine_labels_by_coarse_labels[coarse_label].remove(fine_label)

                    weights_by_coarse_labels[coarse_label] = renormalize(weights_by_coarse_labels[coarse_label],fine_label_idx)

                    if len(indices_by_coarse_labels[coarse_label]) == 0:
                        fine_labels_by_coarse_labels.pop(coarse_label, None)
                        available_coarse_labels.remove(coarse_label)

                        coarse_labels_weights = renormalize(coarse_labels_weights, coarse_label_idx)

        random.shuffle(net_dataidx_map)
        net_dataidx_map_train = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
        net_dataidx_map_test = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
        for i, index in enumerate(net_dataidx_map):
            net_dataidx_map_train[i] = np.append(net_dataidx_map_train[i], index[index < 50_000]).astype(int)
            net_dataidx_map_test[i] = np.append(net_dataidx_map_test[i], index[index >= 50_000]-50000).astype(int)

    elif partition == "noniid-labeluni":
        if dataset == "cifar10":
            num = 2
        elif dataset == "cifar100":
            num = 10
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            num = 1
            K = 2
        elif dataset == 'cifar100':
            K = 100
        elif dataset == 'cifar10':
            K = 10
        else:
            assert False
            print("Choose Dataset in readme.")

        # -------------------------------------------#
        # Divide classes + num samples for each user #
        # -------------------------------------------#
        assert (num * n_parties) % K == 0, "equal classes appearance is needed"
        count_per_class = (num * n_parties) // K
        class_dict = {}
        for i in range(K):
            # sampling alpha_i_c
            probs = np.random.uniform(0.4, 0.6, size=count_per_class)
            # normalizing
            probs_norm = (probs / probs.sum()).tolist()
            class_dict[i] = {'count': count_per_class, 'prob': probs_norm}

        # -------------------------------------#
        # Assign each client with data indexes #
        # -------------------------------------#
        class_partitions = defaultdict(list)
        for i in range(n_parties):
            c = []
            for _ in range(num):
                class_counts = [class_dict[i]['count'] for i in range(K)]
                max_class_counts = np.where(np.array(class_counts) == max(class_counts))[0]
                c.append(np.random.choice(max_class_counts))
                class_dict[c[-1]]['count'] -= 1
            class_partitions['class'].append(c)
            class_partitions['prob'].append([class_dict[i]['prob'].pop() for i in c])

        # -------------------------- #
        # Create class index mapping #
        # -------------------------- #
        data_class_idx_train = {i: np.where(y_train == i)[0] for i in range(K)}
        data_class_idx_test = {i: np.where(y_test == i)[0] for i in range(K)}

        num_samples_train = {i: len(data_class_idx_train[i]) for i in range(K)}
        num_samples_test = {i: len(data_class_idx_test[i]) for i in range(K)}

        # --------- #
        # Shuffling #
        # --------- #
        for data_idx in data_class_idx_train.values():
            random.shuffle(data_idx)
        for data_idx in data_class_idx_test.values():
            random.shuffle(data_idx)

        # ------------------------------ #
        # Assigning samples to each user #
        # ------------------------------ #
        net_dataidx_map_train ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}
        net_dataidx_map_test ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}

        for usr_i in range(n_parties):
            # print("Client: ", usr_i)
            for c, p in zip(class_partitions['class'][usr_i], class_partitions['prob'][usr_i]):
                end_idx_train = int(num_samples_train[c] * p)
                end_idx_test = int(num_samples_test[c] * p)

                # print("Before: ",net_dataidx_map_train[usr_i])
                net_dataidx_map_train[usr_i] = np.append(net_dataidx_map_train[usr_i], data_class_idx_train[c][:end_idx_train])
                net_dataidx_map_test[usr_i] = np.append(net_dataidx_map_test[usr_i], data_class_idx_test[c][:end_idx_test])
                # print("After: ",net_dataidx_map_train[usr_i])
                # net_dataidx_map_train[usr_i].extend(data_class_idx_train[c][:end_idx_train])

                # print("Class ", c, " before: ", data_class_idx_train[c])
                data_class_idx_train[c] = data_class_idx_train[c][end_idx_train:]
                data_class_idx_test[c] = data_class_idx_test[c][end_idx_test:]
        #         print("Class ", c, " after: ", data_class_idx_train[c])

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map_train, logdir)
    print(traindata_cls_counts)
    testdata_cls_counts = record_net_data_stats(y_test, net_dataidx_map_test, logdir)
    print(testdata_cls_counts)
    return (X_train, y_train, X_test, y_test, net_dataidx_map_train, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts)


def compute_accuracy(model, dataloader, get_confusion_matrix=False, device="cpu"):

    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    if type(dataloader) == type([1]):
        pass
    else:
        dataloader = [dataloader]

    correct, total = 0, 0
    with torch.no_grad():
        for tmp in dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device,dtype=torch.int64)
                out = model(x)
                _, pred_label = torch.max(out.data, 1)

                total += x.data.size()[0]

                correct += (pred_label == target.data).sum().item()

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

    if get_confusion_matrix:
        return correct/float(total), conf_matrix

    return correct/float(total)


def compute_accuracy_shakes(model, dataloader, device="cpu"):

    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    if type(dataloader) == type([1]):
        pass
    else:
        dataloader = [dataloader]

    correct, total = 0, 0
    global_loss = 0.
    global_metric = 0.
    n_samples = 0

    all_characters = string.printable
    labels_weight = torch.ones(len(all_characters), device=device)
    for character in CHARACTERS_WEIGHTS:
        labels_weight[all_characters.index(character)] = CHARACTERS_WEIGHTS[character]
    labels_weight = labels_weight * 8
    criterion = nn.CrossEntropyLoss(reduction="none", weight=labels_weight).to(device)

    with torch.no_grad():
        for tmp in dataloader:
            for x, y, indices in tmp:

                # print("x: ", x)
                # print('y: ', y)
                # print("indices: ", indices)
                
                x = x.to(device)
                y = y.to(device)
                n_samples += y.size(0)

                chunk_len = y.size(1)

                y_pred, _ = model(x)
                global_loss += criterion(y_pred, y).sum().item() / chunk_len
                _, predicted = torch.max(y_pred, 1)
                correct = (predicted == y).float()
                acc = correct.sum()
                global_metric += acc.item() / chunk_len

    if was_training:
        model.train()

    return global_metric, n_samples, global_loss/n_samples


def compute_accuracy_pfedKL(net, p_net, dataloader, device="cpu"):

    was_training = False
    if net.training:
        net.eval()
        p_net.eval()
        was_training = True

    criterion = nn.CrossEntropyLoss().to(device)
    net.to(device)
    p_net.to(device)

    if type(dataloader) == type([1]):
        pass
    else:
        dataloader = [dataloader]

    correct, total, total_loss, batch_count = 0, 0, 0, 0
    g_correct, total_g_loss = 0, 0
    p_correct, total_p_loss = 0, 0
    with torch.no_grad():
        for tmp in dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device,dtype=torch.int64)
                output_g = net(x)
                output_p = p_net(x)
                out = output_g.detach() + output_p.detach()

                loss = criterion(out, target)
                g_loss = criterion(output_g, target)
                p_loss = criterion(output_p, target)

                _, pred_label = torch.max(out.data, 1)
                _, g_pred_label = torch.max(output_g.data, 1)
                _, p_pred_label = torch.max(output_p.data, 1)

                total_loss += loss.item()
                total_g_loss += g_loss.item()
                total_p_loss += p_loss.item()

                batch_count += 1
                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()
                g_correct += (g_pred_label == target.data).sum().item()
                p_correct += (p_pred_label == target.data).sum().item()

    if was_training:
        net.train()
        p_net.train()

    cor_vec = [correct, g_correct, p_correct]
    los_vec = [total_loss/batch_count, total_g_loss/batch_count, total_p_loss/batch_count]
    return cor_vec, total, los_vec


def compute_accuracy_fedRod(model, p_head, dataloader, sample_per_class, args, device="cpu"):

    was_training = False
    if model.training:
        model.eval()
        p_head.eval()
        was_training = True

    if args.dataset == "cifar10":
        class_number = 10
    elif args.dataset == "cifar100":
        class_number = 100

    criterion = nn.CrossEntropyLoss().to(device)
    model.to(device)

    if type(dataloader) == type([1]):
        pass
    else:
        dataloader = [dataloader]

    correct, total, total_loss, batch_count = 0, 0, 0, 0
    with torch.no_grad():
        for tmp in dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device,dtype=torch.int64)

                rep = model.produce_feature(x)
                if args.model == 'cnn':
                    out_g = model.fc3(rep)
                elif args.model == 'vit':
                    out_g = model.mlp_head(rep)

                if args.use_hyperRod:
                    linear_whole_para = p_head(sample_per_class).view(-1, class_number)
                    inner_dimm = linear_whole_para.shape[0]
                    linear_w = linear_whole_para[:inner_dimm-1,:] 
                    linear_b = linear_whole_para[inner_dimm-1:,:]
                    rep_h = rep.detach()
                    multi_out = torch.mm(rep_h, linear_w)
                    out_p = torch.add(multi_out, linear_b)
                else:
                    out_p = p_head(rep.detach())
                out = out_g.detach() + out_p
                loss = criterion(out, target)
                _, pred_label = torch.max(out.data, 1)

                total_loss += loss.item()
                batch_count += 1
                total += x.data.size()[0]

                correct += (pred_label == target.data).sum().item()

    if was_training:
        model.train()
        p_head.train()

    return correct, total, total_loss/batch_count
    # return correct/float(total)


def compute_accuracy_fedproto(model, global_protos, dataloader, args, device='cpu'):
    was_training = False
    loss_mse = nn.MSELoss().to(device)
    if global_protos == None:
        return 0, 1, 0.1
    if model.training:
        model.eval()
        was_training = True

    if args.dataset == "cifar10":
        class_number = 10
    elif args.dataset == "cifar100":
        class_number = 100
        
    criterion = nn.CrossEntropyLoss().to(device)
    model.to(device)

    if type(dataloader) == type([1]):
        pass
    else:
        dataloader = [dataloader]

    correct, total, total_loss, batch_count = 0, 0, 0, 0
    with torch.no_grad():
        for tmp in dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device,dtype=torch.int64)

                rep = model.produce_feature(x)
                
                out = float('inf') * torch.ones(target.shape[0], class_number).to(device)
                for i, r in enumerate(rep):
                    for j, pro in global_protos.items():
                        out[i, j] = loss_mse(r, pro)

                # test_acc += (torch.sum(torch.argmin(out, dim=1) == target)).item()
                pred_label = torch.argmin(out, dim=1)
                loss = criterion(out, target)
                total_loss += loss.item()
                batch_count += 1
                total += x.data.size()[0]

                correct += (pred_label == target.data).sum().item()

    if was_training:
        model.train()

    return correct, total, total_loss/batch_count


def compute_accuracy_loss(model, dataloader, device="cpu"):

    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])
    criterion = nn.CrossEntropyLoss().to(device)
    model.to(device)

    if type(dataloader) == type([1]):
        pass
    else:
        dataloader = [dataloader]

    correct, total, total_loss, batch_count = 0, 0, 0, 0
    with torch.no_grad():
        for tmp in dataloader:

            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device,dtype=torch.int64)
                out = model(x)
                loss = criterion(out, target)
                _, pred_label = torch.max(out.data, 1)

                total_loss += loss.item()
                batch_count += 1
                total += x.data.size()[0]
                
                correct += (pred_label == target.data).sum().item()

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

    if was_training:
        model.train()

    return correct, total, total_loss/batch_count


def compute_accuracy_loss_hypervit(model, dataloader, num_class, device="cpu"):
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])
    criterion = nn.CrossEntropyLoss().to(device)
    model.to(device)

    if type(dataloader) == type([1]):
        pass
    else:
        dataloader = [dataloader]

    correct, total, total_loss, batch_count = 0, 0, 0, 0
    with torch.no_grad():
        for tmp in dataloader:
            sample_per_class = torch.zeros(num_class)
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device, dtype=torch.int64)
                out = model(x)

                # balanced softmax
                for k in range(num_class):
                    sample_per_class[k] = (target == k).sum()
                out = out + torch.log(sample_per_class).to(device)

                loss = criterion(out, target)
                _, pred_label = torch.max(out.data, 1)

                total_loss += loss.item()
                batch_count += 1
                total += x.data.size()[0]

                correct += (pred_label == target.data).sum().item()

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

    if was_training:
        model.train()

    return correct, total, total_loss / batch_count


def compute_accuracy_loss_cal(model, train_dataloader, test_dataloader, args, calibrated_model, round=0, device="cpu"):
       
    true_labels_list, pred_labels_list = np.array([]), np.array([])
    # ori_criterion = nn.CrossEntropyLoss().to(device)
    calibrated_model.to(device)
    criterion = nn.NLLLoss().to(device)
    # logsoft_max_function = nn.LogSoftmax(dim=1)
    soft_max_function = nn.Softmax(dim=1)
    model.to(device)

    if type(test_dataloader) == type([1]):
        pass
    else:
        test_dataloader = [test_dataloader]
        train_dataloader = [train_dataloader]

    correct, total, total_loss, batch_count = 0, 0, 0, 0

    if args.dataset == "cifar10":
        class_id = {idd:[torch.zeros(1,128).to(device), 1, []] for idd in range(10)}
    elif args.dataset == "cifar100":
        class_id = {idd:[torch.zeros(1,128).to(device), 1, []] for idd in range(100)}

    class_id_list = {}
    
    with torch.no_grad():
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device,dtype=torch.int64)
                feature = calibrated_model.produce_feature(x).detach()

                for ins in range(len(target)):
                    c_id = int(target[ins])
                    class_id[c_id][0] += feature[ins].view(1,-1)
                    class_id[c_id][1] += 1
                    class_id[c_id][2].append(feature[ins])
                    if c_id not in class_id_list.keys():
                        class_id_list[c_id] = 1
                    else:
                        class_id_list[c_id] += 1
                         
        z_proto = 0
        ff = 0
        # sigma = []
        sigma_matrix = 0
        # dim_v = class_id[0][0].shape[1]
        for cc in range(len(class_id.keys())):
            class_id[cc][0] = class_id[cc][0]/class_id[cc][1]
            feature_set = class_id[cc][2]
            if len(feature_set)==0:
                ss = torch.eye(128).to(device)
                # sigma.append(ss)
                # print(sigma)
            elif len(feature_set)==1:
                # part1 = feature_set[0].view(1,-1) - class_id[cc][0]
                # part2 = part1.view(-1,1)
                # ss = (torch.mm(part2, part1)).inverse()

                pou = torch.eye(128).to(device)
                second = (torch.pow((feature_set[0].view(1,-1) - class_id[cc][0]), 2)).expand(128, 128) * pou
                ss = second.inverse()
                # sigma.append(ss)
            else:
                ss = 0

                for e_id in range(len(feature_set)):
                    ele = feature_set[e_id]
                    # if e_id == 0:
                    #     feat = ele.view(1,-1)
                    # else:
                    #     feat = torch.cat((feat,ele.view(1,-1)),0)

                    # part1 = ele.view(1,-1) - class_id[cc][0]
                    # part2 = part1.view(-1,1)
                    # pou = torch.mm(part2, part1)

                    pou = torch.eye(128).to(device)
                    second = (torch.pow((ele.view(1,-1) - class_id[cc][0]), 2)).expand(128,128) * pou
                    ss += second

                    # for jj in range(dim_v):
                    #     pou[jj][jj] = ((ele[jj] - class_id[cc][0][0][jj])**2).data
                    # ss += pou
                    # print("ss1: ", ss)
                    # print("ss2: ", ss2)
   
                # feat = feat.cpu().numpy().T
                # s2 = np.cov(feat)

                ss = (ss/(len(feature_set)-1)).inverse()
                # sigma.append(ss)

            if ff == 0:
                z_proto = class_id[cc][0]
                sigma_matrix = ss.view(1,128,128)
                ff = 1
            else:
                sigma_matrix = torch.cat((sigma_matrix, ss.view(1,128,128)), 0)
                z_proto = torch.cat((z_proto, class_id[cc][0]), 0)


        appear_cls_id = list(class_id_list.keys())
        # print("class id list: ", appear_cls_id)
        # print("class id dict: ", class_id_list)
        # print("Prototype Shape: ",z_proto.shape)
        # print("Prototype: ",z_proto)

        for tmp in test_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device,dtype=torch.int64)
                feature = model.produce_feature(x).detach()
                logits1 = model.mlp_head(feature)
                probs1 = soft_max_function(logits1)
                # print("probs1: ", probs1)

                # dists = euclidean_dist(feature, z_proto)
                dists = distance_calculate(feature, z_proto, 'Mahalanobis', device, sigma_matrix = sigma_matrix)
                mask_matrix = torch.zeros_like(dists)
                mask_matrix[:,appear_cls_id] = 1
                # print("Dists0: ", dists)

                # logits2 = -dists*mask_matrix
                logits2 = -dists/100
                normed_logits2 = soft_max_function(logits2) * mask_matrix
                probs2 = normed_logits2/(torch.sum(normed_logits2, dim=1).view(-1,1))
                # if round==1:
                #     print("logits2: ", logits2[0])
                #     print("test: ", soft_max_function(logits2))
                #     print("normed_logits2: ", normed_logits2[0])
                #     print("appear_cls_id: ", appear_cls_id)
                #     print("mask_matrix: ", mask_matrix[0])
                #     print("probs2: ", probs2[0])


                if args.no_mlp_head:
                    probs = probs2
                else:
                    lam = args.lambda_value
                    probs = (1-lam)*probs1 + lam*probs2
                    # probs = probs/(torch.sum(probs, dim=1).view(-1,1))

                logits = torch.log(probs)
                loss = criterion(logits, target)

                # if round==1:
                #     print("probs: ",probs[0])
                #     print("logits: ", logits[0])
                #     print('loss item: ',loss.item())
                #     print("    ")

                _, pred_label = torch.max(probs.data, 1)
                # print()
                # print("batch_idx: ", batch_idx)
                # print("loss item: ", loss.item())
                # print("logits: ", logits[0])
                # print("predict: ", pred_label)
                # print("GD: ", target.data)

                total_loss += loss.item()
                batch_count += 1
                total += x.data.size()[0]
                
                correct += (pred_label == target.data).sum().item()

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

    # print( )
    # print("Total loss: ", total_loss)
    # print("batch_count: ", batch_count)

    return correct, total, total_loss/batch_count


def compute_accuracy_loss_knn(model, train_dataloader, test_dataloader, datastore, embedding_dim, args, device="cpu"):

    criterion = nn.CrossEntropyLoss().to(device)
    model.to(device)

    datastore.clear()

    n_samples = len(train_dataloader.dataset)
    total = len(test_dataloader.dataset)

    if type(test_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]
        test_dataloader = [test_dataloader]

    if args.dataset == 'cifar10':
        n_classes = 10
    elif args.dataset == 'cifar100':
        n_classes = 100

    with torch.no_grad():
        train_features = 0
        train_labels = 0

        ff = 0
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device,dtype=torch.int64)

                if args.alg=="knn-per":
                    activation = {}
                    def hook_fn(model, input_, output):
                        activation["features"] = output.squeeze().cpu().numpy()
                    model.features.register_forward_hook(hook_fn)
                    out = model(x)
                    t_feature = activation["features"]

                # elif args.alg in ["protoVit",'hyperVit']:
                else:

                    # t_feature = model.cal_feature(x).detach()
                    # if model.pool == 'mean':
                    #     tf_feature = t_feature.mean(dim = 1)
                    # else:
                    #     tf_feature = t_feature[:, 0]
                    # tf_feature = model.to_latent(tf_feature)
                    # t_feature = t_feature.view(-1, 65*128) 
                    # out = model.mlp_head(tf_feature)

                    t_feature = model.produce_feature(x).detach()
                    out = model.mlp_head(t_feature)
                    t_feature = t_feature.cpu().numpy()

                if ff == 0:
                    ff = 1
                    train_labels = target.data.cpu().numpy()
                    train_features = t_feature
                else:
                    train_labels = np.hstack((train_labels, target.data.cpu().numpy()))
                    train_features = np.vstack((train_features, t_feature))

            # print("train_labels: ", train_labels.shape)
            # print("train_features: ", train_features.shape)
            # assert False

        test_features = 0
        test_labels = 0
        test_outputs = 0
        ff = 0
        for tmp in test_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device,dtype=torch.int64)

                if args.alg=="knn-per":
                    activation = {}
                    def hook_fn(model, input_, output):
                        activation["features"] = output.squeeze().cpu().numpy()
                    model.features.register_forward_hook(hook_fn)
                    out = model(x)
                    t_feature = activation["features"]
                    
                # elif args.alg in ["protoVit",'hyperVit']:
                else:

                    # t_feature = model.cal_feature(x).detach()
                    # if model.pool == 'mean':
                    #     tf_feature = t_feature.mean(dim = 1)
                    # else:
                    #     tf_feature = t_feature[:, 0]
                    # tf_feature = model.to_latent(tf_feature)
                    # t_feature = t_feature.view(-1, 65*128) 
                    # out = model.mlp_head(tf_feature)

                    t_feature = model.produce_feature(x).detach()
                    out = model.mlp_head(t_feature)
                    t_feature = t_feature.cpu().numpy()

                if ff == 0:
                    ff = 1
                    test_labels = target.data.cpu().numpy()
                    test_features = t_feature
                    test_outputs = F.softmax(out, dim=1).cpu().numpy()
                else:
                    test_labels = np.hstack((test_labels, target.data.cpu().numpy()))
                    test_features = np.vstack((test_features, t_feature))
                    test_outputs = np.vstack((test_outputs, F.softmax(out, dim=1).cpu().numpy()))

        datastore.build(train_features, train_labels)
        distances, indices = datastore.index.search(test_features, args.k_value)
        similarities = np.exp(-distances / (embedding_dim * 1.))
        neighbors_labels = datastore.labels[indices]
        masks = np.zeros(((n_classes,) + similarities.shape))
        for class_id in range(n_classes):
            masks[class_id] = neighbors_labels == class_id

        knn_outputs = (similarities * masks).sum(axis=2) / similarities.sum(axis=1)
        knn_outputs = knn_outputs.T
        outputs = args.knn_weight * knn_outputs + (1 - args.knn_weight) * test_outputs

        predictions = np.argmax(outputs, axis=1)
        correct = (test_labels == predictions).sum()

    total_loss = criterion(torch.tensor(outputs), torch.tensor(test_labels))
    return correct, total, total_loss


def compute_accuracy_local(nets, args, net_dataidx_map_train, net_dataidx_map_test, device="cpu"):
    if args.train_acc_pre:
        train_results = defaultdict(lambda: defaultdict(list))
    test_results = defaultdict(lambda: defaultdict(list))
    for net_id in range(args.n_parties):
        # print("net_id: ", net_id)
        local_model = copy.deepcopy(nets[net_id])
        local_model.eval()

        if args.dataset == 'shakespeare':
            train_dl_local = net_dataidx_map_train[net_id]
            test_dl_local = net_dataidx_map_test[net_id]

            test_correct, test_total, test_avg_loss = compute_accuracy_shakes(local_model, test_dl_local, device=device)
            if args.train_acc_pre:
                train_correct, train_total, train_avg_loss = compute_accuracy_shakes(local_model, train_dl_local, device=device)
        else:
            dataidxs_train = net_dataidx_map_train[net_id]
            dataidxs_test = net_dataidx_map_test[net_id]
            if args.noise_type == 'space':
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, net_id, args.n_parties-1)
            else:
                noise_level = args.noise / (args.n_parties - 1) * net_id
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level)

            test_correct, test_total, test_avg_loss = compute_accuracy_loss(local_model, test_dl_local, device=device)
            if args.train_acc_pre:
                train_correct, train_total, train_avg_loss = compute_accuracy_loss(local_model, train_dl_local, device=device)
        
        if args.train_acc_pre:
            train_results[net_id]['loss'] = train_avg_loss 
            train_results[net_id]['correct'] = train_correct
            train_results[net_id]['total'] = train_total

        test_results[net_id]['loss'] = test_avg_loss 
        test_results[net_id]['correct'] = test_correct
        test_results[net_id]['total'] = test_total

    test_total_correct = sum([val['correct'] for val in test_results.values()])
    test_total_samples = sum([val['total'] for val in test_results.values()])
    test_avg_loss = np.mean([val['loss'] for val in test_results.values()])
    test_avg_acc = test_total_correct / test_total_samples

    test_all_acc = [val['correct'] / val['total'] for val in test_results.values()]

    if args.train_acc_pre:
        train_total_correct = sum([val['correct'] for val in train_results.values()])
        train_total_samples = sum([val['total'] for val in train_results.values()])
        train_avg_loss = np.mean([val['loss'] for val in train_results.values()])
        train_acc_pre = train_total_correct / train_total_samples

        train_all_acc = [val['correct'] / val['total'] for val in train_results.values()]
        return train_results, train_avg_loss, train_acc_pre, train_all_acc, test_results, test_avg_loss, test_avg_acc, test_all_acc
    else:
        return 0, 0, 0, 0, test_results, test_avg_loss, test_avg_acc, test_all_acc


def compute_accuracy_local_per(personal_qkv_list, nets, args, net_dataidx_map_train, net_dataidx_map_test, device="cpu"):
    if args.train_acc_pre:
        train_results = defaultdict(lambda: defaultdict(list))
    test_results = defaultdict(lambda: defaultdict(list))
    for net_id in range(args.n_parties):
        # print("net_id: ", net_id)
        local_model = copy.deepcopy(nets[net_id])
        node_weights = personal_qkv_list[net_id]
        local_model.load_state_dict(node_weights, strict=False)
        local_model.eval()

        dataidxs_train = net_dataidx_map_train[net_id]
        dataidxs_test = net_dataidx_map_test[net_id]
        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level)

        test_correct, test_total, test_avg_loss = compute_accuracy_loss(local_model, test_dl_local, device=device)
        if args.train_acc_pre:
            train_correct, train_total, train_avg_loss = compute_accuracy_loss(local_model, train_dl_local, device=device)
            train_results[net_id]['loss'] = train_avg_loss 
            train_results[net_id]['correct'] = train_correct
            train_results[net_id]['total'] = train_total

        test_results[net_id]['loss'] = test_avg_loss 
        test_results[net_id]['correct'] = test_correct
        test_results[net_id]['total'] = test_total

    test_total_correct = sum([val['correct'] for val in test_results.values()])
    test_total_samples = sum([val['total'] for val in test_results.values()])
    test_avg_loss = np.mean([val['loss'] for val in test_results.values()])
    test_avg_acc = test_total_correct / test_total_samples

    test_all_acc = [val['correct'] / val['total'] for val in test_results.values()]

    if args.train_acc_pre:
        train_total_correct = sum([val['correct'] for val in train_results.values()])
        train_total_samples = sum([val['total'] for val in train_results.values()])
        train_avg_loss = np.mean([val['loss'] for val in train_results.values()])
        train_acc_pre = train_total_correct / train_total_samples

        train_all_acc = [val['correct'] / val['total'] for val in train_results.values()]
        return train_results, train_avg_loss, train_acc_pre, train_all_acc, test_results, test_avg_loss, test_avg_acc, test_all_acc
    else:
        return 0, 0, 0, 0, test_results, test_avg_loss, test_avg_acc, test_all_acc


def compute_accuracy_local_ft(nets, args, net_dataidx_map_train, net_dataidx_map_test, device="cpu"):
    if args.train_acc_pre:
        train_results = defaultdict(lambda: defaultdict(list))
    test_results = defaultdict(lambda: defaultdict(list))
    epochs = args.epochs
    print("Finetune before testing")
    for net_id in range(args.n_parties):
        # print("net_id: ", net_id)
        local_model = copy.deepcopy(nets[net_id])
        local_model.eval()

        dataidxs_train = net_dataidx_map_train[net_id]
        dataidxs_test = net_dataidx_map_test[net_id]
        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level)

        optimizer = optim.SGD([p for p in local_model.parameters() if p.requires_grad], lr=args.lr, momentum=args.rho, weight_decay=args.reg)
        criterion = nn.CrossEntropyLoss().to(device)

        cnt = 0
        local_model.train()
        if type(train_dl_local) == type([1]):
            pass
        else:
            train_dl_local = [train_dl_local]

        if args.dataset == "cifar100":
            num_class = 100
        elif args.dataset == "cifar10":
            num_class = 10

        for epoch in range(epochs):
            for tmp in train_dl_local:
                sample_per_class = torch.zeros(num_class)
                for batch_idx, (x, target) in enumerate(tmp):
                    x, target = x.to(device), target.to(device)
                    optimizer.zero_grad()
                    x.requires_grad = True
                    target.requires_grad = False
                    target = target.long()
                    out = local_model(x)
                    loss = criterion(out, target)
                    loss.backward()
                    optimizer.step()

        test_correct, test_total, test_avg_loss = compute_accuracy_loss(local_model, test_dl_local, device=device)
        if args.train_acc_pre:
            train_correct, train_total, train_avg_loss = compute_accuracy_loss(local_model, train_dl_local, device=device)
            train_results[net_id]['loss'] = train_avg_loss 
            train_results[net_id]['correct'] = train_correct
            train_results[net_id]['total'] = train_total

        test_results[net_id]['loss'] = test_avg_loss 
        test_results[net_id]['correct'] = test_correct
        test_results[net_id]['total'] = test_total

    test_total_correct = sum([val['correct'] for val in test_results.values()])
    test_total_samples = sum([val['total'] for val in test_results.values()])
    test_avg_loss = np.mean([val['loss'] for val in test_results.values()])
    test_avg_acc = test_total_correct / test_total_samples
    test_all_acc = [val['correct'] / val['total'] for val in test_results.values()]

    if args.train_acc_pre:
        train_total_correct = sum([val['correct'] for val in train_results.values()])
        train_total_samples = sum([val['total'] for val in train_results.values()])
        train_avg_loss = np.mean([val['loss'] for val in train_results.values()])
        train_acc_pre = train_total_correct / train_total_samples

        train_all_acc = [val['correct'] / val['total'] for val in train_results.values()]
        return train_results, train_avg_loss, train_acc_pre, train_all_acc, test_results, test_avg_loss, test_avg_acc, test_all_acc
    else:
        return 0, 0, 0, 0, test_results, test_avg_loss, test_avg_acc, test_all_acc


def compute_accuracy_two_branch(personal_bn_list, global_model, p_nets, args, net_dataidx_map_train, net_dataidx_map_test, device="cpu"):

    if args.train_acc_pre:
        train_results = defaultdict(lambda: defaultdict(list))
    test_results = defaultdict(lambda: defaultdict(list))
    for net_id in range(args.n_parties):

        node_weights = personal_bn_list[net_id]
        g_net = copy.deepcopy(global_model)
        g_net.load_state_dict(node_weights, strict=False)
        g_net.eval()
        p_net = copy.deepcopy(p_nets[net_id])
        p_net.eval()

        dataidxs_train = net_dataidx_map_train[net_id]
        dataidxs_test = net_dataidx_map_test[net_id]
        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, net_id, args.n_parties-1)
        elif args.noise_type == 'increasing':
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, apply_noise=True)
        else:
            noise_level = 0
            train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level)
       
        
        test_correct, test_total, test_avg_loss = compute_accuracy_pfedKL(g_net, p_net, test_dl_local, device=device)
        if args.train_acc_pre:
            train_correct, train_total, train_avg_loss = compute_accuracy_pfedKL(g_net, p_net, train_dl_local, device=device)
            train_results[net_id]['loss'] = train_avg_loss 
            train_results[net_id]['correct'] = train_correct
            train_results[net_id]['g_loss'] = train_avg_loss[1] 
            train_results[net_id]['g_correct'] = train_correct[1]
            train_results[net_id]['p_loss'] = train_avg_loss[2] 
            train_results[net_id]['p_correct'] = train_correct[2]
            train_results[net_id]['total'] = train_total

        test_results[net_id]['loss'] = test_avg_loss[0] 
        test_results[net_id]['correct'] = test_correct[0]
        test_results[net_id]['g_loss'] = test_avg_loss[1] 
        test_results[net_id]['g_correct'] = test_correct[1]
        test_results[net_id]['p_loss'] = test_avg_loss[2] 
        test_results[net_id]['p_correct'] = test_correct[2]
        test_results[net_id]['total'] = test_total

    test_total_correct = sum([val['correct'] for val in test_results.values()])
    test_g_total_correct = sum([val['g_correct'] for val in test_results.values()])
    test_p_total_correct = sum([val['p_correct'] for val in test_results.values()])
    test_total_samples = sum([val['total'] for val in test_results.values()])
    test_avg_loss = np.mean([val['loss'] for val in test_results.values()])
    test_g_avg_loss = np.mean([val['g_loss'] for val in test_results.values()])
    test_p_avg_loss = np.mean([val['p_loss'] for val in test_results.values()]) 

    test_avg_acc = test_total_correct / test_total_samples
    test_all_acc = [val['correct'] / val['total'] for val in test_results.values()]
    test_g_avg_acc = test_g_total_correct / test_total_samples
    test_g_all_acc = [val['g_correct'] / val['total'] for val in test_results.values()]
    test_p_avg_acc = test_p_total_correct / test_total_samples
    test_p_all_acc = [val['p_correct'] / val['total'] for val in test_results.values()]

    if args.train_acc_pre:
        train_total_correct = sum([val['correct'] for val in train_results.values()])
        train_total_samples = sum([val['total'] for val in train_results.values()])
        train_avg_loss = np.mean([val['loss'] for val in train_results.values()])
        train_acc_pre = train_total_correct / train_total_samples

        train_all_acc = [val['correct'] / val['total'] for val in train_results.values()]
        return train_results, train_avg_loss, train_acc_pre, train_all_acc, test_results, test_avg_loss, test_avg_acc, test_all_acc, test_g_avg_loss, test_g_avg_acc, test_g_all_acc, test_p_avg_loss, test_p_avg_acc, test_p_all_acc
    else:
        return 0, 0, 0, 0, test_results, test_avg_loss, test_avg_acc, test_all_acc, test_g_avg_loss, test_g_avg_acc, test_g_all_acc, test_p_avg_loss, test_p_avg_acc, test_p_all_acc


def compute_accuracy_per_client_simple(global_model, args, net_dataidx_map_train, net_dataidx_map_test, nets=None, device="cpu"):
    if args.train_acc_pre:
        train_results = defaultdict(lambda: defaultdict(list))
    test_results = defaultdict(lambda: defaultdict(list))
    for net_id in range(args.n_parties):

        local_model = copy.deepcopy(global_model)
        local_model.eval()

        if args.dataset == 'shakespeare':
            train_dl_local = net_dataidx_map_train[net_id]
            test_dl_local = net_dataidx_map_test[net_id]
            test_correct, test_total, test_avg_loss = compute_accuracy_shakes(local_model, test_dl_local, device=device)
            if args.train_acc_pre:
                train_correct, train_total, train_avg_loss = compute_accuracy_shakes(local_model, train_dl_local, device=device)

        else:
            dataidxs_train = net_dataidx_map_train[net_id]
            dataidxs_test = net_dataidx_map_test[net_id]
            if args.noise_type == 'space':
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, net_id, args.n_parties-1)
            elif args.noise_type == 'increasing':
                noise_level = args.noise / (args.n_parties - 1) * net_id
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, apply_noise=True)
            else:
                noise_level = 0
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level)
     
            test_correct, test_total, test_avg_loss = compute_accuracy_loss(local_model, test_dl_local, device=device)
            if args.train_acc_pre:
                train_correct, train_total, train_avg_loss = compute_accuracy_loss(local_model, train_dl_local, device=device)


        if args.train_acc_pre:
            train_results[net_id]['loss'] = train_avg_loss 
            train_results[net_id]['correct'] = train_correct
            train_results[net_id]['total'] = train_total

        test_results[net_id]['loss'] = test_avg_loss 
        test_results[net_id]['correct'] = test_correct
        test_results[net_id]['total'] = test_total

    test_total_correct = sum([val['correct'] for val in test_results.values()])
    test_total_samples = sum([val['total'] for val in test_results.values()])
    test_avg_loss = np.mean([val['loss'] for val in test_results.values()])
    test_avg_acc = test_total_correct / test_total_samples

    test_all_acc = [val['correct'] / val['total'] for val in test_results.values()]

    if args.train_acc_pre:
        train_total_correct = sum([val['correct'] for val in train_results.values()])
        train_total_samples = sum([val['total'] for val in train_results.values()])
        train_avg_loss = np.mean([val['loss'] for val in train_results.values()])
        train_acc_pre = train_total_correct / train_total_samples

        train_all_acc = [val['correct'] / val['total'] for val in train_results.values()]
        return train_results, train_avg_loss, train_acc_pre, train_all_acc, test_results, test_avg_loss, test_avg_acc, test_all_acc
    else:
        return 0, 0, 0, 0, test_results, test_avg_loss, test_avg_acc, test_all_acc


def compute_accuracy_per_client_cluster(global_model1, global_model2, args, net_dataidx_map_train, net_dataidx_map_test, nets=None, device="cpu"):
    if args.train_acc_pre:
        train_results = defaultdict(lambda: defaultdict(list))
    test_results = defaultdict(lambda: defaultdict(list))
    epochs = args.epochs
    for net_id in range(args.n_parties):

        if net_id < int(args.n_parties/2):
            local_model = copy.deepcopy(global_model1)
        else:
            local_model = copy.deepcopy(global_model2)

        dataidxs_train = net_dataidx_map_train[net_id]
        dataidxs_test = net_dataidx_map_test[net_id]

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, net_id, args.n_parties-1)
        elif args.noise_type == 'increasing':
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, apply_noise=True)
        else:
            noise_level = 0
            train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level)

        # optimizer = optim.SGD([p for p in local_model.parameters() if p.requires_grad], lr=args.lr, momentum=args.rho, weight_decay=args.reg)
        # criterion = nn.CrossEntropyLoss().to(device)

        # cnt = 0
        # local_model.train()
        # if type(train_dl_local) == type([1]):
        #     pass
        # else:
        #     train_dl_local = [train_dl_local]

        # if args.dataset == "cifar100":
        #     num_class = 100
        # elif args.dataset == "cifar10":
        #     num_class = 10

        # for epoch in range(epochs):
        #     for tmp in train_dl_local:
        #         sample_per_class = torch.zeros(num_class)
        #         for batch_idx, (x, target) in enumerate(tmp):
        #             x, target = x.to(device), target.to(device)
        #             optimizer.zero_grad()
        #             x.requires_grad = True
        #             target.requires_grad = False
        #             target = target.long()
        #             out = local_model(x)
        #             loss = criterion(out, target)
        #             loss.backward()
        #             optimizer.step()

        local_model.eval()
        test_correct, test_total, test_avg_loss = compute_accuracy_loss(local_model, test_dl_local, device=device)
        if args.train_acc_pre:
            train_correct, train_total, train_avg_loss = compute_accuracy_loss(local_model, train_dl_local, device=device)
            train_results[net_id]['loss'] = train_avg_loss 
            train_results[net_id]['correct'] = train_correct
            train_results[net_id]['total'] = train_total

        test_results[net_id]['loss'] = test_avg_loss 
        test_results[net_id]['correct'] = test_correct
        test_results[net_id]['total'] = test_total

    test_total_correct = sum([val['correct'] for val in test_results.values()])
    test_total_samples = sum([val['total'] for val in test_results.values()])
    test_avg_loss = np.mean([val['loss'] for val in test_results.values()])
    test_avg_acc = test_total_correct / test_total_samples

    test_all_acc = [val['correct'] / val['total'] for val in test_results.values()]

    if args.train_acc_pre:
        train_total_correct = sum([val['correct'] for val in train_results.values()])
        train_total_samples = sum([val['total'] for val in train_results.values()])
        train_avg_loss = np.mean([val['loss'] for val in train_results.values()])
        train_acc_pre = train_total_correct / train_total_samples

        train_all_acc = [val['correct'] / val['total'] for val in train_results.values()]
        return train_results, train_avg_loss, train_acc_pre, train_all_acc, test_results, test_avg_loss, test_avg_acc, test_all_acc
    else:
        return 0, 0, 0, 0, test_results, test_avg_loss, test_avg_acc, test_all_acc


def compute_accuracy_per_client(hyper, nets, global_model, args, net_dataidx_map_train, net_dataidx_map_test, num_class, device="cpu"):  
    hyper.eval()
    if args.train_acc_pre:
        train_results = defaultdict(lambda: defaultdict(list))
    test_results = defaultdict(lambda: defaultdict(list))

    for net_id in range(args.n_parties):

        node_weights = hyper(torch.tensor([net_id], dtype=torch.long).to(device), True)
        local_model = copy.deepcopy(global_model)
        local_model.load_state_dict(node_weights, strict=False)
        local_model.eval()

        if args.calibrated:
            calibrated_model = copy.deepcopy(nets[net_id])
            calibrated_model.eval()
        else:
            calibrated_model = None

        if args.dataset == "shakespeare":
            train_dl_local = net_dataidx_map_train[net_id]
            test_dl_local = net_dataidx_map_test[net_id]

            test_correct, test_total, test_avg_loss = compute_accuracy_shakes(local_model, test_dl_local, device=device)
            if args.train_acc_pre:
                train_correct, train_total, train_avg_loss = compute_accuracy_shakes(local_model, train_dl_local, device=device)
        else:
            dataidxs_train = net_dataidx_map_train[net_id]
            dataidxs_test = net_dataidx_map_test[net_id]
            if args.noise_type == 'space':
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, net_id, args.n_parties-1)
            elif args.noise_type == 'increasing':
                noise_level = args.noise / (args.n_parties - 1) * net_id
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, apply_noise=True)
            else:
                noise_level = 0
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level)

            if args.calibrated:
                test_correct, test_total, test_avg_loss = compute_accuracy_loss_cal(local_model, train_dl_local, test_dl_local, args, calibrated_model, device=device)
                if args.train_acc_pre:
                    train_correct, train_total, train_avg_loss = compute_accuracy_loss_cal(local_model, train_dl_local, train_dl_local, args, calibrated_model, device=device)

            elif args.k_neighbor:
                n_train_samples = len(train_dl_local.dataset)
                capacity = int(args.capacity_ratio * n_train_samples)
                rng = np.random.default_rng(seed=args.init_seed)
                # vec_dim = 128*65
                vec_dim = 128
                datastore = DataStore(capacity, "random", vec_dim, rng)
                
                test_correct, test_total, test_avg_loss = compute_accuracy_loss_knn(local_model, train_dl_local, test_dl_local, datastore, vec_dim, args, device=device)
                if args.train_acc_pre:
                    train_correct, train_total, train_avg_loss = compute_accuracy_loss_knn(local_model, train_dl_local, train_dl_local, datastore, vec_dim, args, device=device)

            else:
                # test_correct, test_total, test_avg_loss = compute_accuracy_loss_hypervit(local_model, test_dl_local, num_class, device=device)
                # if args.train_acc_pre:
                #     train_correct, train_total, train_avg_loss = compute_accuracy_loss_hypervit(local_model, train_dl_local, num_class, device=device)
                test_correct, test_total, test_avg_loss = compute_accuracy_loss(local_model, test_dl_local, device=device)
                if args.train_acc_pre:
                    train_correct, train_total, train_avg_loss = compute_accuracy_loss(local_model, train_dl_local, device=device)

        if args.train_acc_pre:
            train_results[net_id]['loss'] = train_avg_loss 
            train_results[net_id]['correct'] = train_correct
            train_results[net_id]['total'] = train_total

        test_results[net_id]['loss'] = test_avg_loss 
        test_results[net_id]['correct'] = test_correct
        test_results[net_id]['total'] = test_total

    test_total_correct = sum([val['correct'] for val in test_results.values()])
    test_total_samples = sum([val['total'] for val in test_results.values()])
    test_avg_loss = np.mean([val['loss'] for val in test_results.values()])
    test_avg_acc = test_total_correct / test_total_samples

    test_all_acc = [val['correct'] / val['total'] for val in test_results.values()]
    if args.train_acc_pre:
        train_total_correct = sum([val['correct'] for val in train_results.values()])
        train_total_samples = sum([val['total'] for val in train_results.values()])
        train_avg_loss = np.mean([val['loss'] for val in train_results.values()])
        train_acc_pre = train_total_correct / train_total_samples

        train_all_acc = [val['correct'] / val['total'] for val in train_results.values()]
        return train_results, train_avg_loss, train_acc_pre, train_all_acc, test_results, test_avg_loss, test_avg_acc, test_all_acc
    else:
        return 0, 0, 0, 0, test_results, test_avg_loss, test_avg_acc, test_all_acc


def compute_accuracy_per_client_proto_cluster(hyper, nets, global_model, args, net_dataidx_map_train, net_dataidx_map_test, num_class, device="cpu"):  
    hyper.eval()
    if args.train_acc_pre:
        train_results = defaultdict(lambda: defaultdict(list))
    test_results = defaultdict(lambda: defaultdict(list))

    for net_id in range(args.n_parties):
        node_weights = hyper(torch.tensor([net_id], dtype=torch.long).to(device), True)
        local_model = copy.deepcopy(nets[net_id])
        local_model.load_state_dict(node_weights, strict=False)
        local_model.eval()

        dataidxs_train = net_dataidx_map_train[net_id]
        dataidxs_test = net_dataidx_map_test[net_id]
        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, net_id, args.n_parties-1)
        elif args.noise_type == 'increasing':
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, apply_noise=True)
        else:
            noise_level = 0
            train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level)

        test_correct, test_total, test_avg_loss = compute_accuracy_loss(local_model, test_dl_local, device=device)
        if args.train_acc_pre:
            train_correct, train_total, train_avg_loss = compute_accuracy_loss(local_model, train_dl_local, device=device)
            train_results[net_id]['loss'] = train_avg_loss 
            train_results[net_id]['correct'] = train_correct
            train_results[net_id]['total'] = train_total

        test_results[net_id]['loss'] = test_avg_loss 
        test_results[net_id]['correct'] = test_correct
        test_results[net_id]['total'] = test_total

    test_total_correct = sum([val['correct'] for val in test_results.values()])
    test_total_samples = sum([val['total'] for val in test_results.values()])
    test_avg_loss = np.mean([val['loss'] for val in test_results.values()])
    test_avg_acc = test_total_correct / test_total_samples
    test_all_acc = [val['correct'] / val['total'] for val in test_results.values()]
    if args.train_acc_pre:
        train_total_correct = sum([val['correct'] for val in train_results.values()])
        train_total_samples = sum([val['total'] for val in train_results.values()])
        train_avg_loss = np.mean([val['loss'] for val in train_results.values()])
        train_acc_pre = train_total_correct / train_total_samples

        train_all_acc = [val['correct'] / val['total'] for val in train_results.values()]
        return train_results, train_avg_loss, train_acc_pre, train_all_acc, test_results, test_avg_loss, test_avg_acc, test_all_acc
    else:
        return 0, 0, 0, 0, test_results, test_avg_loss, test_avg_acc, test_all_acc


def compute_accuracy_percnn_client(hyper, global_model, args, net_dataidx_map_train, net_dataidx_map_test, device="cpu"):

    hyper.eval()
    if args.train_acc_pre:
        train_results = defaultdict(lambda: defaultdict(list))
    test_results = defaultdict(lambda: defaultdict(list))

    for net_id in range(args.n_parties):
        dataidxs_train = net_dataidx_map_train[net_id]
        dataidxs_test = net_dataidx_map_test[net_id]

        node_weights = hyper(torch.tensor([net_id], dtype=torch.long).to(device))
        local_model = copy.deepcopy(global_model)
        local_model.load_state_dict(node_weights, strict=False)
        local_model.eval()

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, net_id, args.n_parties-1)
        elif args.noise_type == 'increasing':
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, apply_noise=True)
        else:
            noise_level = 0
            train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level)

        test_correct, test_total, test_avg_loss = compute_accuracy_loss(local_model, test_dl_local, device=device)
        if args.train_acc_pre:
            train_correct, train_total, train_avg_loss = compute_accuracy_loss(local_model, train_dl_local, device=device)

            train_results[net_id]['loss'] = train_avg_loss 
            train_results[net_id]['correct'] = train_correct
            train_results[net_id]['total'] = train_total

        test_results[net_id]['loss'] = test_avg_loss 
        test_results[net_id]['correct'] = test_correct
        test_results[net_id]['total'] = test_total

    test_total_correct = sum([val['correct'] for val in test_results.values()])
    test_total_samples = sum([val['total'] for val in test_results.values()])
    test_avg_loss = np.mean([val['loss'] for val in test_results.values()])
    test_avg_acc = test_total_correct / test_total_samples

    test_all_acc = [val['correct'] / val['total'] for val in test_results.values()]
    if args.train_acc_pre:
        train_total_correct = sum([val['correct'] for val in train_results.values()])
        train_total_samples = sum([val['total'] for val in train_results.values()])
        train_avg_loss = np.mean([val['loss'] for val in train_results.values()])
        train_acc_pre = train_total_correct / train_total_samples

        train_all_acc = [val['correct'] / val['total'] for val in train_results.values()]
        return train_results, train_avg_loss, train_acc_pre, train_all_acc, test_results, test_avg_loss, test_avg_acc, test_all_acc
    else:
        return 0, 0, 0, 0, test_results, test_avg_loss, test_avg_acc, test_all_acc


def compute_accuracy_personally(personal_qkv_list, global_model, args, net_dataidx_map_train, net_dataidx_map_test, nets, round=0, device="cpu"):

    if args.train_acc_pre:
        train_results = defaultdict(lambda: defaultdict(list))
    test_results = defaultdict(lambda: defaultdict(list))
    for net_id in range(args.n_parties):

        node_weights = personal_qkv_list[net_id]
        local_model = copy.deepcopy(global_model)
        local_model.load_state_dict(node_weights, strict=False)
        local_model.eval()

        if args.calibrated:
            calibrated_model = copy.deepcopy(nets[net_id])
            calibrated_model.eval()
        else:
            calibrated_model = None

        if args.dataset == "shakespeare":
            train_dl_local = net_dataidx_map_train[net_id]
            test_dl_local = net_dataidx_map_test[net_id]

            test_correct, test_total, test_avg_loss = compute_accuracy_shakes(local_model, test_dl_local, device=device)
            if args.train_acc_pre:
                train_correct, train_total, train_avg_loss = compute_accuracy_shakes(local_model, train_dl_local, device=device)
        else:
            dataidxs_train = net_dataidx_map_train[net_id]
            dataidxs_test = net_dataidx_map_test[net_id]
            if args.noise_type == 'space':
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, net_id, args.n_parties-1)
            elif args.noise_type == 'increasing':
                noise_level = args.noise / (args.n_parties - 1) * net_id
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, apply_noise=True)
            else:
                noise_level = 0
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level)
           
            if args.calibrated:
                test_correct, test_total, test_avg_loss = compute_accuracy_loss_cal(local_model, train_dl_local, test_dl_local, args, calibrated_model, round, device=device)
                if args.train_acc_pre:
                    train_correct, train_total, train_avg_loss = compute_accuracy_loss_cal(local_model, train_dl_local, train_dl_local, args, calibrated_model, round, device=device)
            
            elif args.k_neighbor:
                n_train_samples = len(train_dl_local.dataset)
                capacity = int(args.capacity_ratio * n_train_samples)
                rng = np.random.default_rng(seed=args.init_seed)
                # vec_dim = 128*65
                vec_dim = 128
                datastore = DataStore(capacity, "random", vec_dim, rng)
                
                test_correct, test_total, test_avg_loss = compute_accuracy_loss_knn(local_model, train_dl_local, test_dl_local, datastore, vec_dim, args, device=device)
                if args.train_acc_pre:
                    train_correct, train_total, train_avg_loss = compute_accuracy_loss_knn(local_model, train_dl_local, train_dl_local, datastore, vec_dim, args, device=device)
            
            else:
                test_correct, test_total, test_avg_loss = compute_accuracy_loss(local_model, test_dl_local, device=device)
                if args.train_acc_pre:
                    train_correct, train_total, train_avg_loss = compute_accuracy_loss(local_model, train_dl_local, device=device)

        if args.train_acc_pre:
            train_results[net_id]['loss'] = train_avg_loss 
            train_results[net_id]['correct'] = train_correct
            train_results[net_id]['total'] = train_total

        test_results[net_id]['loss'] = test_avg_loss 
        test_results[net_id]['correct'] = test_correct
        test_results[net_id]['total'] = test_total

    test_total_correct = sum([val['correct'] for val in test_results.values()])
    test_total_samples = sum([val['total'] for val in test_results.values()])
    test_avg_loss = np.mean([val['loss'] for val in test_results.values()])
    test_avg_acc = test_total_correct / test_total_samples

    test_all_acc = [val['correct'] / val['total'] for val in test_results.values()]

    if args.train_acc_pre:
        train_total_correct = sum([val['correct'] for val in train_results.values()])
        train_total_samples = sum([val['total'] for val in train_results.values()])
        train_avg_loss = np.mean([val['loss'] for val in train_results.values()])
        train_acc_pre = train_total_correct / train_total_samples

        train_all_acc = [val['correct'] / val['total'] for val in train_results.values()]
        return train_results, train_avg_loss, train_acc_pre, train_all_acc, test_results, test_avg_loss, test_avg_acc, test_all_acc
    else:
        return 0, 0, 0, 0, test_results, test_avg_loss, test_avg_acc, test_all_acc


def compute_accuracy_perplus_client(hyper, prototype_dict, class_id_dict, global_model, args, net_dataidx_map_train, net_dataidx_map_test, nets, round, device="cpu"):
    
    hyper.eval()
    if args.train_acc_pre:
        train_results = defaultdict(lambda: defaultdict(list))
    test_results = defaultdict(lambda: defaultdict(list))

    for net_id in range(args.n_parties):
        dataidxs_train = net_dataidx_map_train[net_id]
        dataidxs_test = net_dataidx_map_test[net_id]

        net_id_embedding = prototype_dict[net_id].to(device)
        if args.partition == 'noniid-labeluni' and args.position_embedding:
            class_id_list = class_id_dict[net_id]
            net_id_embedding = hyper.pos_embedding(net_id_embedding, class_id_list, device)

        node_weights = hyper(net_id_embedding, True)
        local_model = copy.deepcopy(global_model)
        local_model.load_state_dict(node_weights, strict=False)
        local_model.eval()

        if args.calibrated:
            calibrated_model = copy.deepcopy(nets[net_id])
            calibrated_model.eval()
        else:
            calibrated_model = None

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level)

        if args.calibrated:
            test_correct, test_total, test_avg_loss = compute_accuracy_loss_cal(local_model, train_dl_local, test_dl_local, args, calibrated_model, round, device=device)
            if args.train_acc_pre:
                train_correct, train_total, train_avg_loss = compute_accuracy_loss_cal(local_model, train_dl_local, train_dl_local, args, calibrated_model, round, device=device)
        
        elif args.k_neighbor:
            n_train_samples = len(train_dl_local.dataset)
            capacity = int(args.capacity_ratio * n_train_samples)
            rng = np.random.default_rng(seed=args.init_seed)
            # vec_dim = 128*65
            vec_dim = 128
            datastore = DataStore(capacity, "random", vec_dim, rng)
            
            test_correct, test_total, test_avg_loss = compute_accuracy_loss_knn(local_model, train_dl_local, test_dl_local, datastore, vec_dim, args, device=device)
            if args.train_acc_pre:
                train_correct, train_total, train_avg_loss = compute_accuracy_loss_knn(local_model, train_dl_local, train_dl_local, datastore, vec_dim, args, device=device)

        else:
            test_correct, test_total, test_avg_loss = compute_accuracy_loss(local_model, test_dl_local, device=device)
            if args.train_acc_pre:
                train_correct, train_total, train_avg_loss = compute_accuracy_loss(local_model, train_dl_local, device=device)

        if args.train_acc_pre:
            train_results[net_id]['loss'] = train_avg_loss 
            train_results[net_id]['correct'] = train_correct
            train_results[net_id]['total'] = train_total

        test_results[net_id]['loss'] = test_avg_loss 
        test_results[net_id]['correct'] = test_correct
        test_results[net_id]['total'] = test_total

    test_total_correct = sum([val['correct'] for val in test_results.values()])
    test_total_samples = sum([val['total'] for val in test_results.values()])
    test_avg_loss = np.mean([val['loss'] for val in test_results.values()])
    test_avg_acc = test_total_correct / test_total_samples

    test_all_acc = [val['correct'] / val['total'] for val in test_results.values()]
    if args.train_acc_pre:
        train_total_correct = sum([val['correct'] for val in train_results.values()])
        train_total_samples = sum([val['total'] for val in train_results.values()])
        train_avg_loss = np.mean([val['loss'] for val in train_results.values()])
        train_acc_pre = train_total_correct / train_total_samples

        train_all_acc = [val['correct'] / val['total'] for val in train_results.values()]
        return train_results, train_avg_loss, train_acc_pre, train_all_acc, test_results, test_avg_loss, test_avg_acc, test_all_acc
    else:
        return 0, 0, 0, 0, test_results, test_avg_loss, test_avg_acc, test_all_acc


def compute_accuracy_perRod(personal_head_list, global_model, args, net_dataidx_map_train, net_dataidx_map_test, alpha_dict, device="cpu", hyper=None):
    if hyper != None:
        hyper.eval()
    if args.train_acc_pre:
        train_results = defaultdict(lambda: defaultdict(list))
    test_results = defaultdict(lambda: defaultdict(list))

    for net_id in range(args.n_parties):

        local_model = copy.deepcopy(global_model)
        if hyper!=None:
            node_weights = hyper(torch.tensor([net_id], dtype=torch.long).to(device), True)
            local_model.load_state_dict(node_weights, strict=False)
        local_model.eval()

        p_head = copy.deepcopy(personal_head_list[net_id])
        p_head.eval()

        sample_per_class = alpha_dict[net_id]
        if args.dataset == "shakespeare":
            train_dl_local = net_dataidx_map_train[net_id]
            test_dl_local = net_dataidx_map_test[net_id]

            test_correct, test_total, test_avg_loss = compute_accuracy_shakes(local_model, test_dl_local, device=device)
            if args.train_acc_pre:
                train_correct, train_total, train_avg_loss = compute_accuracy_shakes(local_model, train_dl_local, device=device)
        else:
            dataidxs_train = net_dataidx_map_train[net_id]
            dataidxs_test = net_dataidx_map_test[net_id]
            if args.noise_type == 'space':
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, net_id, args.n_parties-1)
            else:
                noise_level = args.noise / (args.n_parties - 1) * net_id
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level)          
            
            test_correct, test_total, test_avg_loss = compute_accuracy_fedRod(local_model, p_head, test_dl_local, sample_per_class, args, device=device)
            if args.train_acc_pre:
                train_correct, train_total, train_avg_loss = compute_accuracy_fedRod(local_model, p_head, train_dl_local, sample_per_class, args, device=device)

        if args.train_acc_pre:
            train_results[net_id]['loss'] = train_avg_loss 
            train_results[net_id]['correct'] = train_correct
            train_results[net_id]['total'] = train_total

        test_results[net_id]['loss'] = test_avg_loss 
        test_results[net_id]['correct'] = test_correct
        test_results[net_id]['total'] = test_total

    test_total_correct = sum([val['correct'] for val in test_results.values()])
    test_total_samples = sum([val['total'] for val in test_results.values()])
    test_avg_loss = np.mean([val['loss'] for val in test_results.values()])
    test_avg_acc = test_total_correct / test_total_samples

    test_all_acc = [val['correct'] / val['total'] for val in test_results.values()]

    if args.train_acc_pre:
        train_total_correct = sum([val['correct'] for val in train_results.values()])
        train_total_samples = sum([val['total'] for val in train_results.values()])
        train_avg_loss = np.mean([val['loss'] for val in train_results.values()])
        train_acc_pre = train_total_correct / train_total_samples

        train_all_acc = [val['correct'] / val['total'] for val in train_results.values()]
        return train_results, train_avg_loss, train_acc_pre, train_all_acc, test_results, test_avg_loss, test_avg_acc, test_all_acc
    else:
        return 0, 0, 0, 0, test_results, test_avg_loss, test_avg_acc, test_all_acc


def compute_accuracy_perProto(nets, global_protos, args, net_dataidx_map_train, net_dataidx_map_test, device="cpu"):
    if args.train_acc_pre:
        train_results = defaultdict(lambda: defaultdict(list))
    test_results = defaultdict(lambda: defaultdict(list))

    for net_id in range(args.n_parties):

        local_model = copy.deepcopy(nets[net_id])
        local_model.eval()

        dataidxs_train = net_dataidx_map_train[net_id]
        dataidxs_test = net_dataidx_map_test[net_id]
        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level)          
        
        test_correct, test_total, test_avg_loss = compute_accuracy_fedproto(local_model, 
            global_protos, test_dl_local, args, device=device)
        
        if args.train_acc_pre:
            train_correct, train_total, train_avg_loss = compute_accuracy_fedproto(local_model, 
            global_protos, train_dl_local, args, device=device)

            train_results[net_id]['loss'] = train_avg_loss 
            train_results[net_id]['correct'] = train_correct
            train_results[net_id]['total'] = train_total

        test_results[net_id]['loss'] = test_avg_loss 
        test_results[net_id]['correct'] = test_correct
        test_results[net_id]['total'] = test_total

    test_total_correct = sum([val['correct'] for val in test_results.values()])
    test_total_samples = sum([val['total'] for val in test_results.values()])
    test_avg_loss = np.mean([val['loss'] for val in test_results.values()])
    test_avg_acc = test_total_correct / test_total_samples

    test_all_acc = [val['correct'] / val['total'] for val in test_results.values()]

    if args.train_acc_pre:
        train_total_correct = sum([val['correct'] for val in train_results.values()])
        train_total_samples = sum([val['total'] for val in train_results.values()])
        train_avg_loss = np.mean([val['loss'] for val in train_results.values()])
        train_acc_pre = train_total_correct / train_total_samples

        train_all_acc = [val['correct'] / val['total'] for val in train_results.values()]
        return train_results, train_avg_loss, train_acc_pre, train_all_acc, test_results, test_avg_loss, test_avg_acc, test_all_acc
    else:
        return 0, 0, 0, 0, test_results, test_avg_loss, test_avg_acc, test_all_acc


def compute_accuracy_hphead_client(hyper, personal_qkv_list, global_model, args, net_dataidx_map_train, net_dataidx_map_test, device="cpu"):

    hyper.eval()
    if args.train_acc_pre:
        train_results = defaultdict(lambda: defaultdict(list))
    test_results = defaultdict(lambda: defaultdict(list))

    for net_id in range(args.n_parties):
        dataidxs_train = net_dataidx_map_train[net_id]
        dataidxs_test = net_dataidx_map_test[net_id]

        node_weights = hyper(torch.tensor([net_id], dtype=torch.long).to(device), True)
        p_head = personal_qkv_list[net_id]
        local_model = copy.deepcopy(global_model)
        local_model.load_state_dict(node_weights, strict=False)
        local_model.load_state_dict(p_head, strict=False)
        local_model.eval()

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level)

        test_correct, test_total, test_avg_loss = compute_accuracy_loss(local_model, test_dl_local, device=device)
        if args.train_acc_pre:
            train_correct, train_total, train_avg_loss = compute_accuracy_loss(local_model, train_dl_local, device=device)

            train_results[net_id]['loss'] = train_avg_loss 
            train_results[net_id]['correct'] = train_correct
            train_results[net_id]['total'] = train_total

        test_results[net_id]['loss'] = test_avg_loss 
        test_results[net_id]['correct'] = test_correct
        test_results[net_id]['total'] = test_total

    test_total_correct = sum([val['correct'] for val in test_results.values()])
    test_total_samples = sum([val['total'] for val in test_results.values()])
    test_avg_loss = np.mean([val['loss'] for val in test_results.values()])
    test_avg_acc = test_total_correct / test_total_samples

    test_all_acc = [val['correct'] / val['total'] for val in test_results.values()]
    if args.train_acc_pre:
        train_total_correct = sum([val['correct'] for val in train_results.values()])
        train_total_samples = sum([val['total'] for val in train_results.values()])
        train_avg_loss = np.mean([val['loss'] for val in train_results.values()])
        train_acc_pre = train_total_correct / train_total_samples

        train_all_acc = [val['correct'] / val['total'] for val in train_results.values()]
        return train_results, train_avg_loss, train_acc_pre, train_all_acc, test_results, test_avg_loss, test_avg_acc, test_all_acc
    else:
        return 0, 0, 0, 0, test_results, test_avg_loss, test_avg_acc, test_all_acc


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., net_id=None, total=0):
        self.std = std
        self.mean = mean
        self.net_id = net_id
        self.num = int(sqrt(total))
        if self.num * self.num < total:
            self.num = self.num + 1

    def __call__(self, tensor):
        if self.net_id is None:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        else:
            tmp = torch.randn(tensor.size())
            filt = torch.zeros(tensor.size())
            size = int(28 / self.num)
            row = int(self.net_id / size)
            col = self.net_id % size
            for i in range(size):
                for j in range(size):
                    filt[:,row*size+i,col*size+j] = 1
            tmp = tmp * filt
            return tensor + tmp * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class GaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):

        return torch.clamp((tensor + torch.randn(tensor.size()) * self.std + self.mean), 0, 255)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, noise_level=0, net_id=None, total=0, apply_noise=False):
    if dataset in ('mnist', 'femnist', 'fmnist', 'cifar10','cifar100', 'svhn', 'generated', 'covtype', 'a9a', 'rcv1', 'SUSY'):
        if dataset == 'mnist':
            dl_obj = MNIST_truncated

            transform_train = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)])

        elif dataset == 'femnist':
            dl_obj = FEMNIST
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)])

        elif dataset == 'fmnist':
            dl_obj = FashionMNIST_truncated
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)])

        elif dataset == 'svhn':
            dl_obj = SVHN_custom
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)])


        elif dataset == 'cifar10':
            dl_obj = CIFAR10_truncated

            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                # transforms.Lambda(lambda x: F.pad(
                #     Variable(x.unsqueeze(0), requires_grad=False),
                #     (4, 4, 4, 4), mode='reflect').data.squeeze()),
                # transforms.ToPILImage(),
                # transforms.RandomCrop(32),
                # transforms.RandomHorizontalFlip(),
                # transforms.ToTensor(),
                # AddGaussianNoise(0., noise_level, net_id, total)
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                # AddGaussianNoise(0., noise_level, net_id, total)
                ])

        elif dataset == 'cifar100':
            dl_obj = CIFAR100_truncated

            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
                # transforms.Lambda(lambda x: F.pad(
                #     Variable(x.unsqueeze(0), requires_grad=False),
                #     (4, 4, 4, 4), mode='reflect').data.squeeze()),
                # transforms.ToPILImage(),
                # transforms.RandomCrop(32),
                # transforms.RandomHorizontalFlip(),
                # transforms.ToTensor(),
                # AddGaussianNoise(0., noise_level, net_id, total)
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
                # AddGaussianNoise(0., noise_level, net_id, total)
                ])


        else:
            dl_obj = Generated
            transform_train = None
            transform_test = None


        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=False)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=False)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)

    return train_dl, test_dl, train_ds, test_ds


def get_divided_dataloader(dataset, datadir, train_bs, test_bs, dataidxs_train, dataidxs_test, noise_level=0, net_id=None, total=0, drop_last=False, apply_noise=False):
    if dataset in ('mnist', 'femnist', 'fmnist', 'cifar10', 'cifar100', 'svhn', 'generated', 'covtype', 'a9a', 'rcv1', 'SUSY'):
        if dataset == 'mnist':
            dl_obj = MNIST_truncated

            transform_train = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)])

        elif dataset == 'femnist':
            dl_obj = FEMNIST
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)])

        elif dataset == 'fmnist':
            dl_obj = FashionMNIST_truncated
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)])

        elif dataset == 'svhn':
            dl_obj = SVHN_custom
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)])


        elif dataset == 'cifar10':
            dl_obj = CIFAR10_truncated
            if apply_noise:
                transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    GaussianNoise(0., noise_level)
                ])
                # data prep for test set
                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    GaussianNoise(0., noise_level)
                    ])
            else:
                transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
   
                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        elif dataset == 'cifar100':
            dl_obj = CIFAR100_truncated
            if apply_noise:
                transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
                    GaussianNoise(0., noise_level)
                ])
                # data prep for test set
                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
                    GaussianNoise(0., noise_level)
                    ])
            else:
                transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])

                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])

        else:
            dl_obj = Generated
            transform_train = None
            transform_test = None


        train_ds = dl_obj(datadir, dataidxs=dataidxs_train, train=True, transform=transform_train, download=False)
        test_ds = dl_obj(datadir, dataidxs= dataidxs_test ,train=False, transform=transform_test, download=False)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=drop_last)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)

    return train_dl, test_dl, train_ds, test_ds


def get_spe_dataloaders(dataset, data_dir, batch_size, chunk_len, is_validation=False):

    inputs, targets = None, None

    train_iterators, val_iterators, test_iterators = [], [], []

    for task_id, task_dir in enumerate(os.listdir(data_dir)):
        task_data_path = os.path.join(data_dir, task_dir)

        train_iterator = get_spe_loader(dataset=dataset,
        path=os.path.join(task_data_path, f"train{EXTENSIONS[dataset]}"),
        batch_size=batch_size, chunk_len=chunk_len, inputs=inputs, targets=targets, train=True)

        val_iterator = get_spe_loader(dataset=dataset,
        path=os.path.join(task_data_path, f"train{EXTENSIONS[dataset]}"),
        batch_size=batch_size, chunk_len=chunk_len, inputs=inputs, targets=targets, train=False)

        if is_validation:
            test_set = "val"
        else:
            test_set = "test"

        test_iterator =get_spe_loader(dataset=dataset,
        path=os.path.join(task_data_path, f"{test_set}{EXTENSIONS[dataset]}"),
        batch_size=batch_size, chunk_len=chunk_len, inputs=inputs, targets=targets, train=False)

        if test_iterator!=None:
            train_iterators.append(train_iterator)
            val_iterators.append(val_iterator)
            test_iterators.append(test_iterator)

    original_client_num = task_id + 1

    return train_iterators, val_iterators, test_iterators, original_client_num


def get_spe_loader(dataset, path, batch_size, train, chunk_len=5, inputs=None, targets=None):

    if dataset == "femnist":
        dataset = SubFEMNIST(path)
    elif dataset == "shakespeare":
        dataset = CharacterDataset(path, chunk_len=chunk_len)
    else:
        raise NotImplementedError(f"{dataset} not recognized type; possible are {list(LOADER_TYPE.keys())}")

    if len(dataset) == 0:
        return

    # drop last batch, because of BatchNorm layer used in mobilenet_v2
    drop_last = (len(dataset) > batch_size) and train

    return data.DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=drop_last, num_workers=NUM_WORKERS)


def weights_init(m):
    """
    Initialise weights of the model.
    """
    if(type(m) == nn.ConvTranspose2d or type(m) == nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif(type(m) == nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class NormalNLLLoss:
    """
    Calculate the negative log likelihood
    of normal distribution.
    This needs to be minimised.

    Treating Q(cj | x) as a factored Gaussian.
    """
    def __call__(self, x, mu, var):

        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
        nll = -(logli.sum(1).mean())

        return nll


def noise_sample(choice, n_dis_c, dis_c_dim, n_con_c, n_z, batch_size, device):
    """
    Sample random noise vector for training.

    INPUT
    --------
    n_dis_c : Number of discrete latent code.
    dis_c_dim : Dimension of discrete latent code.
    n_con_c : Number of continuous latent code.
    n_z : Dimension of iicompressible noise.
    batch_size : Batch Size
    device : GPU/CPU
    """

    z = torch.randn(batch_size, n_z, 1, 1, device=device)
    idx = np.zeros((n_dis_c, batch_size))
    if(n_dis_c != 0):
        dis_c = torch.zeros(batch_size, n_dis_c, dis_c_dim, device=device)

        c_tmp = np.array(choice)

        for i in range(n_dis_c):
            idx[i] = np.random.randint(len(choice), size=batch_size)
            for j in range(batch_size):
                idx[i][j] = c_tmp[int(idx[i][j])]

            dis_c[torch.arange(0, batch_size), i, idx[i]] = 1.0

        dis_c = dis_c.view(batch_size, -1, 1, 1)

    if(n_con_c != 0):
        # Random uniform between -1 and 1.
        con_c = torch.rand(batch_size, n_con_c, 1, 1, device=device) * 2 - 1

    noise = z
    if(n_dis_c != 0):
        noise = torch.cat((z, dis_c), dim=1)
    if(n_con_c != 0):
        noise = torch.cat((noise, con_c), dim=1)

    return noise, idx


