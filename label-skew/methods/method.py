import numpy as np
import json
import torch
import torch.optim as optim
from collections import OrderedDict, defaultdict
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
import argparse
import logging
import os
import copy
from math import *
import random
from utils import *


def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device="cpu", global_net=None, fedprox=False):
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD([p for p in net.parameters() if p.requires_grad], lr=lr, momentum=args.rho, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    net.train()
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    if fedprox:
        mu = args.mu
        global_weight_collector = list(global_net.to(device).parameters())

    if args.dataset == "cifar100":
        num_class = 100
    elif args.dataset == "cifar10":
        num_class = 10

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            sample_per_class = torch.zeros(num_class)
            for batch_idx, (x, target) in enumerate(tmp):

                x, target = x.to(device), target.to(device)
                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)

                if fedprox:
                    fed_prox_reg = 0.0
                    for param_index, param in enumerate(net.parameters()):
                        fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index]))**2)
                    loss += fed_prox_reg

                loss.backward()
                optimizer.step()

                cnt += 1
                epoch_loss_collector.append(loss.item())

        if len(epoch_loss_collector) == 0:
            assert args.model == 'cnn-b'
            epoch_loss = 0
        else:
            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)

    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    if args.train_acc_pre:
        train_acc = compute_accuracy(net, train_dataloader, device=device)
        return train_acc, test_acc
    else:
        return None, test_acc


def train_net_fedRod(net_id, net, p_head, sample_per_class, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device="cpu"):
    
    if args.dataset == "cifar10":
        class_number = 10
        criterion = nn.CrossEntropyLoss().to(device)
    elif args.dataset == "cifar100":
        class_number = 100
        criterion = nn.CrossEntropyLoss().to(device)

    criterion_ba = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
    opt_pred = optim.SGD(p_head.parameters(), lr=lr)

    cnt = 0
    global_loss = 0
    global_metric = 0
    n_samples = 0
    net.train()
    p_head.train()

    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    # print("net_id: ", net_id)
    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:        
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                rep = net.produce_feature(x)
                out_g = net.fc3(rep)
                loss_bsm = criterion_ba(out_g, target)
                optimizer.zero_grad()
                loss_bsm.backward()
                optimizer.step()

                out_p = p_head(rep.detach())
                loss = criterion(out_g.detach() + out_p, target)
                opt_pred.zero_grad()
                loss.backward()
                opt_pred.step()
                
                cnt += 1
                epoch_loss_collector.append(loss.item())
                epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)

    correct, total, _ = compute_accuracy_fedRod(net, p_head, test_dataloader, sample_per_class, args, device=device)
    test_acc = correct/float(total)

    if args.train_acc_pre:
        correct, total, _ = compute_accuracy_fedRod(net, p_head, train_dataloader, sample_per_class, args, device=device)
        train_acc = correct/float(total)
        return train_acc, test_acc 
    else:
        return None, test_acc 


def train_net_2branch(net_id, net, p_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device="cpu"):
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
        p_optimizer = optim.Adam(filter(lambda p: p.requires_grad, p_net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg, amsgrad=True)
        p_optimizer = optim.Adam(filter(lambda p: p.requires_grad, p_net.parameters()), lr=lr, weight_decay=args.reg, amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD([p for p in net.parameters() if p.requires_grad], lr=lr, momentum=args.rho, weight_decay=args.reg)
        p_optimizer = optim.SGD([p for p in p_net.parameters() if p.requires_grad], lr=lr, momentum=args.rho, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    net.train()
    p_net.train()
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    for epoch in range(epochs):
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()
                output_g = net(x)
                output_p = p_net(x)

                optimizer.zero_grad()
                loss = criterion(output_g, target)
                loss.backward()
                optimizer.step()

                p_optimizer.zero_grad()
                loss_p = criterion(output_p, target)
                loss_p.backward()
                p_optimizer.step()

    correct, total, _ = compute_accuracy_co2(net, p_net, test_dataloader, device=device)
    test_acc = correct[0]/float(total)

    if args.train_acc_pre:
        correct, total, _ = compute_accuracy_co2(net, p_net, train_dataloader, device=device)
        train_acc = correct[0]/float(total)
        return train_acc, test_acc
    else:
        return None, test_acc


def local_train_net(nets, selected, args, net_dataidx_map_train, net_dataidx_map_test, logger, device="cpu"):
    avg_acc = 0.0
    results_dict = defaultdict(list)
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs_train = net_dataidx_map_train[net_id]
        dataidxs_test = net_dataidx_map_test[net_id]

        if args.log_flag:
            logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs_train)))
        net.to(device)

        train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test)
        n_epoch = args.epochs
        trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl_local, n_epoch, args.lr, args.optimizer, args, device=device)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
        
    avg_acc /= len(selected)
    nets_list = list(nets.values())
    return nets_list


def local_train_net_per(nets, selected, args, net_dataidx_map_train, net_dataidx_map_test, logger=None, device="cpu"):
    avg_acc = 0.0
    n_epoch = args.epochs
    for net_id, net in nets.items():
        if net_id not in selected:
            continue

        net.to(device)
        dataidxs_train = net_dataidx_map_train[net_id]
        dataidxs_test = net_dataidx_map_test[net_id]
        if args.model == 'cnn-b':
            train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 2*args.batch_size, dataidxs_train, dataidxs_test, drop_last=True)
        else:
            train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test)
        
        trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl_local, n_epoch, 
            args.lr, args.optimizer, args, device=device)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc

    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list


def local_train_net_per_2branch(nets, p_nets, selected, args, net_dataidx_map_train, net_dataidx_map_test, a_iter, logger=None, device="cpu"):
    n_epoch = args.epochs
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        net.to(device)
        p_net = p_nets[net_id]
        p_net.to(device)

        dataidxs_train = net_dataidx_map_train[net_id]
        dataidxs_test = net_dataidx_map_test[net_id]
        train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 2*args.batch_size, dataidxs_train, dataidxs_test, drop_last=True)
        _, testacc = train_net_2branch(net_id, net, p_net, train_dl_local, test_dl_local, n_epoch, 
        args.lr, args.optimizer, args, device=device)   
        logger.info("net %d final test acc %f" % (net_id, testacc))
    nets_list = list(nets.values())
    return nets_list


def local_train_net_fedprox(nets, selected, global_model, args, net_dataidx_map_train, net_dataidx_map_test, logger=None, device="cpu"):
    avg_acc = 0.0
    n_epoch = args.epochs

    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        net.to(device)
        dataidxs_train = net_dataidx_map_train[net_id]
        dataidxs_test = net_dataidx_map_test[net_id]
        noise_level = args.noise
        train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test)
        
        trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl_local, 
        n_epoch, args.lr, args.optimizer, args, device=device, global_net=global_model, fedprox=True)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list


def local_train_net_fedrod(nets, selected, p_head, args, net_dataidx_map_train, net_dataidx_map_test, logger=None, device="cpu", alpha=None):
    avg_acc = 0.0
    n_epoch = args.epochs
    for net_id, net in nets.items():
        if net_id not in selected:
            continue

        net.to(device)
        personal_head = p_head[net_id]
        sample_per_class = []
   
        sample_per_class = alpha[net_id]
        dataidxs_train = net_dataidx_map_train[net_id]
        dataidxs_test = net_dataidx_map_test[net_id]
        train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test)

        trainacc, testacc = train_net_fedRod(net_id, net, personal_head, sample_per_class,
        train_dl_local, test_dl_local, n_epoch, args.lr, args.optimizer, args, device=device)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
    avg_acc /= len(selected)

    return p_head


def local_train_net_moon(nets, args, net_dataidx_map_train, net_dataidx_map_test, global_model, prev_model_pool, device="cpu"):
    global_model.to(device)
    for net_id, net in nets.items():
        net.to(device)
        dataidxs_train = net_dataidx_map_train[net_id]
        dataidxs_test = net_dataidx_map_test[net_id]
        noise_level = 0
        if args.model == 'cnn-b':
            train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 2*args.batch_size, dataidxs_train, dataidxs_test, drop_last=True)
        else:
            train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test)
        n_epoch = args.epochs

        prev_model = prev_model_pool[net_id].to(device)
        trainacc, testacc = train_net_fedcon(net_id, net, global_model, prev_model, train_dl_local, test_dl_local, n_epoch, args.lr,
                                              args.optimizer, args.mu, args.temperature, args, device=device)
        logger.info("net %d final test acc %f" % (net_id, testacc))
    return nets


def train_net_fedcon(net_id, net, global_net, previous_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, temperature, args, device="cpu"):
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().to(device)
    previous_net.to(device)
    previous_net.eval()
    global_net.eval()
    net.train()

    cnt = 0
    cos=torch.nn.CosineSimilarity(dim=-1)

    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            pro1 = net.produce_feature(x)
            out = net.fc3(pro1)
            pro2 = global_net.produce_feature(x)

            posi = cos(pro1, pro2)
            logits = posi.reshape(-1,1)

            pro3 = previous_net.produce_feature(x)
            nega = cos(pro1, pro3)
            logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)

            logits /= temperature
            labels = torch.zeros(x.size(0)).cuda().long()

            loss2 = mu * criterion(logits, labels)

            loss1 = criterion(out, target)
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(loss1.item())
            epoch_loss2_collector.append(loss2.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
    test_acc  = compute_accuracy(net, test_dataloader, device=device)
    if args.train_acc_pre:
        train_acc = compute_accuracy(net, train_dataloader, device=device)
        return train_acc, test_acc
    else:
        return None, test_acc
