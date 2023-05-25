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

from datasets import CIFAR10_truncated, CIFAR100_truncated
from math import sqrt

import torch.nn as nn
from einops import repeat
import torch.optim as optim
import torchvision.utils as vutils
import time
import random

from config import params
import sklearn.datasets as sk

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass


def get_form(model):
    tmpm = []
    tmpv = []
    for name in model.state_dict().keys():
        if 'running_mean' in name:
            tmpm.append(model.state_dict()[name].detach().to('cpu').numpy())
        if 'running_var' in name:
            tmpv.append(model.state_dict()[name].detach().to('cpu').numpy())
    return tmpm, tmpv


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


def partition_data(dataset, datadir, partition, n_parties, beta=0.3, logdir=None):
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

    if partition == "noniid-labeldir":
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
            K = 10
        elif dataset == "cifar100":
            num = 10
            K = 100
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


def compute_accuracy_co2(net, p_net, dataloader, device="cpu"):
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
                out_g = model.fc3(rep)
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


def compute_accuracy_local(nets, args, net_dataidx_map_train, net_dataidx_map_test, device="cpu"):
    if args.train_acc_pre:
        train_results = defaultdict(lambda: defaultdict(list))
    test_results = defaultdict(lambda: defaultdict(list))
    for net_id in range(args.n_parties):
        # print("net_id: ", net_id)
        local_model = copy.deepcopy(nets[net_id])
        local_model.eval()
        dataidxs_train = net_dataidx_map_train[net_id]
        dataidxs_test = net_dataidx_map_test[net_id]
        train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test)
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
        train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test)
        test_correct, test_total, test_avg_loss = compute_accuracy_co2(g_net, p_net, test_dl_local, device=device)
        
        if args.train_acc_pre:
            train_correct, train_total, train_avg_loss = compute_accuracy_co2(g_net, p_net, train_dl_local, device=device)
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
        dataidxs_train = net_dataidx_map_train[net_id]
        dataidxs_test = net_dataidx_map_test[net_id]
        train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test)
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

        dataidxs_train = net_dataidx_map_train[net_id]
        dataidxs_test = net_dataidx_map_test[net_id]
        train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test)
        
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


def compute_accuracy_perRod(personal_head_list, global_model, args, net_dataidx_map_train, net_dataidx_map_test, alpha_dict, device="cpu"):
    if args.train_acc_pre:
        train_results = defaultdict(lambda: defaultdict(list))
    test_results = defaultdict(lambda: defaultdict(list))

    for net_id in range(args.n_parties):

        local_model = copy.deepcopy(global_model)
        local_model.eval()
        p_head = copy.deepcopy(personal_head_list[net_id])
        p_head.eval()

        sample_per_class = alpha_dict[net_id]      
        dataidxs_train = net_dataidx_map_train[net_id]
        dataidxs_test = net_dataidx_map_test[net_id]
        train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test)          
        
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


def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, net_id=None, total=0):
    if dataset in ('cifar10','cifar100'):
        if dataset == 'cifar10':
            dl_obj = CIFAR10_truncated
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])

        elif dataset == 'cifar100':
            dl_obj = CIFAR100_truncated
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
                ])

        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=False)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=False)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)

    return train_dl, test_dl, train_ds, test_ds


def get_divided_dataloader(dataset, datadir, train_bs, test_bs, dataidxs_train, dataidxs_test, net_id=None, total=0, drop_last=False):
    if dataset in ('cifar10', 'cifar100'):
        if dataset == 'cifar10':
            dl_obj = CIFAR10_truncated
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        elif dataset == 'cifar100':
            dl_obj = CIFAR100_truncated
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])

        train_ds = dl_obj(datadir, dataidxs=dataidxs_train, train=True, transform=transform_train, download=False)
        test_ds = dl_obj(datadir, dataidxs= dataidxs_test ,train=False, transform=transform_test, download=False)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=drop_last)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)

    return train_dl, test_dl, train_ds, test_ds


def weights_init(m):
    """
    Initialise weights of the model.
    """
    if(type(m) == nn.ConvTranspose2d or type(m) == nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif(type(m) == nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

