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
from optimizer.optimizer import pFedMeOptimizer


def balanced_softmax_loss(labels, logits, sample_per_class, reduction="mean"):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss


def agg_func(protos):
    """
    Returns the average of the weights.
    """
    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos


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

                # print("batch_idx: ", batch_idx)
                # print("actual batch_size: ", x.shape[0]) 
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                # balanced softmax
                if args.alg == "hyperVit" and args.balanced_soft_max:
                    for k in range(num_class):
                        sample_per_class[k] = (target == k).sum()
                    out = out + torch.log(sample_per_class).to(device)
                loss = criterion(out, target)

                # fedprox
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
            assert args.model in ["mobilent", 'cnn-b']
            epoch_loss = 0
        else:
            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)

    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    if args.train_acc_pre:
        train_acc = compute_accuracy(net, train_dataloader, device=device)
        return train_acc, test_acc
    else:
        return None, test_acc


def train_net_pfedMe(net_id, net, regularized_local, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device="cpu"):
    
    local_params = copy.deepcopy(list(regularized_local.parameters()))
    personalized_params= copy.deepcopy(list(regularized_local.parameters()))
    if args.dataset=="shakespeare":
        all_characters = string.printable
        labels_weight = torch.ones(len(all_characters), device=device)
        for character in CHARACTERS_WEIGHTS:
            labels_weight[all_characters.index(character)] = CHARACTERS_WEIGHTS[character]
        labels_weight = labels_weight * 8
        criterion = nn.CrossEntropyLoss(reduction="none", weight=labels_weight).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    optimizer = pFedMeOptimizer(net.parameters(), 
        lr=lr, lamda=args.pfedMe_lambda, mu=args.pfedMe_mu)

    cnt = 0
    global_loss = 0
    global_metric = 0
    n_samples = 0
    net.train()
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    # print("net_id: ", net_id)
    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            if args.dataset == "shakespeare":
                for x, y, indices in tmp:
                    x = x.to(device)
                    y = y.to(device)

                    n_samples += y.size(0)
                    chunk_len = y.size(1)

                    # in pfedMe args.pfedMe_k is set to 5 by default
                    for ikik in range(args.pfedMe_k):
                        optimizer.zero_grad()
                        y_pred, _ = net(x)
                        loss_vec = criterion(y_pred, y)
                        loss = loss_vec.mean()
                        loss.backward()
                        personalized_params = optimizer.step(local_params, device)
                    global_loss += loss.item() * loss_vec.size(0) / chunk_len
                    _, predicted = torch.max(y_pred, 1)
                    correct = (predicted == y).float()
                    acc = correct.sum()
                    global_metric += acc.item() / chunk_len
                    for new_param, localweight in zip(personalized_params, local_params):
                        localweight = localweight.to(device)
                        localweight.data = localweight.data - args.pfedMe_lambda * lr * (localweight.data - new_param.data)

            else:
                for batch_idx, (x, target) in enumerate(tmp):
                    x, target = x.to(device), target.to(device)

                    # in pfedMe args.pfedMe_k is set to 5 by default
                    for ikik in range(args.pfedMe_k):
                        target = target.long()
                        optimizer.zero_grad()
                        out = net(x)
                        loss = criterion(out, target)
                        loss.backward()
                        personalized_params = optimizer.step(local_params, device)

                    for new_param, localweight in zip(personalized_params, local_params):
                        localweight = localweight.to(device)
                        localweight.data = localweight.data - args.pfedMe_lambda * lr * (localweight.data - new_param.data)   

                    cnt += 1
                    epoch_loss_collector.append(loss.item())
                epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)

    for param, new_param in zip(net.parameters(), personalized_params):
    # for param, new_param in zip(net.parameters(), local):
        param.data = new_param.data.clone()

    if args.dataset == "shakespeare":
        te_metric, te_samples, _ = compute_accuracy_shakes(net, test_dataloader, device=device)
        test_acc = te_metric/te_samples
    else:
        test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    if args.train_acc_pre:
        if args.dataset == "shakespeare":
            tr_metric, tr_samples, _ = compute_accuracy_shakes(net, train_dataloader, device=device)
            train_acc = tr_metric/tr_samples
        else:
            train_acc = compute_accuracy(net, train_dataloader, device=device)
        return train_acc, test_acc, local_params
    else:
        return None, test_acc, local_params


def train_net_fedRod(net_id, net, p_head, sample_per_class, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device="cpu"):
    
    if args.dataset=="shakespeare":
        all_characters = string.printable
        labels_weight = torch.ones(len(all_characters), device=device)
        for character in CHARACTERS_WEIGHTS:
            labels_weight[all_characters.index(character)] = CHARACTERS_WEIGHTS[character]
        labels_weight = labels_weight * 8
        criterion = nn.CrossEntropyLoss(reduction="none", weight=labels_weight).to(device)
    elif args.dataset == "cifar10":
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
            if args.dataset == "shakespeare":
                for x, y, indices in tmp:
 
                    x = x.to(device)
                    y = y.to(device)

                    n_samples += y.size(0)
                    chunk_len = y.size(1)

                    rep = net.produce_feature(x)
                    if args.model == 'lstm':
                        out_g = net.decoder(rep)
                        out_g = out_g.permute(0, 2, 1) 
                    elif args.model == 'transformer':
                        seq_logit = net.trg_word_prj(rep)
                        if net.scale_prj:
                            seq_logit *= net.d_model ** -0.5
        
                        out_g = seq_logit.permute(0, 2, 1)
                    
                    loss_bsm_vec = criterion_ba(out_g, target)
                    loss_bsm = loss_bsm_vec.mean()
                    optimizer.zero_grad()
                    loss_bsm.backward()
                    optimizer.step()

                    out_p = p_head(rep.detach())
                    if args.model == 'transformer' and net.scale_prj:
                        out_p *= net.d_model ** -0.5
                    loss_vec = criterion(out_g.detach() + out_p, target)
                    loss = loss_vec.mean()
                    opt_pred.zero_grad()
                    loss.backward()
                    opr_pred.step()

                    global_loss += loss.item() * loss_vec.size(0) / chunk_len
                    _, predicted = torch.max(y_pred, 1)
                    correct = (predicted == y).float()
                    acc = correct.sum()
                    global_metric += acc.item() / chunk_len
            else:
                for batch_idx, (x, target) in enumerate(tmp):
                    x, target = x.to(device), target.to(device)

                    rep = net.produce_feature(x)
                    if args.model == 'cnn':
                        out_g = net.fc3(rep)
                    elif args.model == 'vit':
                        out_g = net.mlp_head(rep)
                    if args.balanced_soft_max:
                        loss_bsm = balanced_softmax_loss(target, out_g, sample_per_class)
                    else:
                        loss_bsm = criterion_ba(out_g, target)
                    optimizer.zero_grad()
                    loss_bsm.backward()
                    optimizer.step()

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
                    loss = criterion(out_g.detach() + out_p, target)
                    opt_pred.zero_grad()
                    loss.backward()
                    opt_pred.step()
                    
                    cnt += 1
                    epoch_loss_collector.append(loss.item())
                epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)

    if args.dataset == "shakespeare":
        te_metric, te_samples, _ = compute_accuracy_shakes(net, test_dataloader, device=device)
        test_acc = te_metric/te_samples
    else:
        correct, total, _ = compute_accuracy_fedRod(net, p_head, test_dataloader, sample_per_class, args, device=device)
        test_acc = correct/float(total)

    if args.train_acc_pre:
        if args.dataset == "shakespeare":
            tr_metric, tr_samples, _ = compute_accuracy_shakes(net, train_dataloader, device=device)
            train_acc = tr_metric/tr_samples
        else:
            correct, total, _ = compute_accuracy_fedRod(net, p_head, train_dataloader, sample_per_class, args, device=device)
            train_acc = correct/float(total)
        return train_acc, test_acc 
    else:
        return None, test_acc 


def train_net_shakes(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device="cpu", global_net=None, fedprox=False):

    all_characters = string.printable
    labels_weight = torch.ones(len(all_characters), device=device)
    for character in CHARACTERS_WEIGHTS:
        labels_weight[all_characters.index(character)] = CHARACTERS_WEIGHTS[character]
    labels_weight = labels_weight * 8
    criterion = nn.CrossEntropyLoss(reduction="none", weight=labels_weight).to(device)

    optimizer = optim.SGD(
            [param for param in net.parameters() if param.requires_grad],
            lr=lr, momentum=0., weight_decay=5e-4)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    if fedprox:
        # print("fedprox!")
        mu = args.mu
        global_weight_collector = list(global_net.to(device).parameters())

    net.train()
    global_loss = 0.
    global_metric = 0.
    n_samples = 0
    for epoch in range(epochs):
        for tmp in train_dataloader:
            for x, y, indices in tmp:
                # print('y: ', y)
                # print("indices: ", indices)
                x = x.to(device)
                y = y.to(device)

                n_samples += y.size(0)
                chunk_len = y.size(1)
                optimizer.zero_grad()

                y_pred, _ = net(x)
                loss_vec = criterion(y_pred, y)
                loss = loss_vec.mean()

                if fedprox:
                    fed_prox_reg = 0.0
                    for param_index, param in enumerate(net.parameters()):
                        fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index]))**2)
                    loss += fed_prox_reg

                loss.backward()
                optimizer.step()
                global_loss += loss.item() * loss_vec.size(0) / chunk_len
                _, predicted = torch.max(y_pred, 1)
                correct = (predicted == y).float()
                acc = correct.sum()
                global_metric += acc.item() / chunk_len

    te_metric, te_samples, _ = compute_accuracy_shakes(net, test_dataloader, device=device)
    test_acc = te_metric/te_samples

    if args.train_acc_pre:
        tr_metric, tr_samples, _ = compute_accuracy_shakes(net, train_dataloader, device=device)
        train_acc = tr_metric/tr_samples
        return train_acc, test_acc
    else:
        return None, test_acc


def train_net_pfedKL_full(net_id, net, p_net, train_dataloader, test_dataloader, epochs, kl_epochs, lr, args_optimizer, args, a_iter, device="cpu"):
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
    kl_loss = nn.KLDivLoss(reduction='batchmean')

    net.train()
    p_net.train()
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    if a_iter != 0:
        back_g_model = copy.deepcopy(net)
        back_g_model.eval()
        back_p_model = copy.deepcopy(p_net)
        back_p_model.eval()
        for epoch in range(kl_epochs):
            for tmp in train_dataloader:
                for batch_idx, (x, target) in enumerate(tmp):
                    x, target = x.to(device), target.to(device)
                    x.requires_grad = True
                    target.requires_grad = False
                    target = target.long()
                    
                    output_g = net(x)
                    output_p = p_net(x)
                    last_g = back_g_model(x)
                    last_p = back_p_model(x)

                    optimizer.zero_grad()
                    loss = kl_loss(F.log_softmax(output_g, dim=1), F.softmax(last_p.detach(), dim=1))
                    loss.backward()
                    optimizer.step()

                    p_optimizer.zero_grad()
                    loss_p = kl_loss(F.log_softmax(output_p, dim=1), F.softmax(last_g.detach(), dim=1))
                    loss_p.backward()
                    p_optimizer.step()

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

    correct, total, _ = compute_accuracy_pfedKL(net, p_net, test_dataloader, device=device)
    test_acc = correct[0]/float(total)

    if args.train_acc_pre:
        correct, total, _ = compute_accuracy_pfedKL(net, p_net, train_dataloader, device=device)
        train_acc = correct[0]/float(total)
        return train_acc, test_acc
    else:
        return None, test_acc


def train_net_pfedKL_p(net_id, net, p_net, train_dataloader, test_dataloader, epochs, kl_epochs, lr, args_optimizer, args, a_iter, device="cpu"):
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
    kl_loss = nn.KLDivLoss(reduction='batchmean')

    net.train()
    p_net.train()
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    if a_iter != 0:
        back_g_model = copy.deepcopy(net)
        back_g_model.eval()
        for epoch in range(kl_epochs):
            for tmp in train_dataloader:
                for batch_idx, (x, target) in enumerate(tmp):
                    x, target = x.to(device), target.to(device)
                    x.requires_grad = True
                    target.requires_grad = False
                    target = target.long()
                    output_p = p_net(x)
                    last_g = back_g_model(x)

                    p_optimizer.zero_grad()
                    loss_p = kl_loss(F.log_softmax(output_p, dim=1), F.softmax(last_g.detach(), dim=1))
                    loss_p.backward()
                    p_optimizer.step()

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

    correct, total, _ = compute_accuracy_pfedKL(net, p_net, test_dataloader, device=device)
    test_acc = correct[0]/float(total)

    if args.train_acc_pre:
        correct, total, _ = compute_accuracy_pfedKL(net, p_net, train_dataloader, device=device)
        train_acc = correct[0]/float(total)
        return train_acc, test_acc
    else:
        return None, test_acc


def train_net_pfedKL_g(net_id, net, p_net, train_dataloader, test_dataloader, epochs, kl_epochs, lr, args_optimizer, args, a_iter, device="cpu"):
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
    kl_loss = nn.KLDivLoss(reduction='batchmean')

    net.train()
    p_net.train()
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    if a_iter != 0:
        back_p_model = copy.deepcopy(p_net)
        back_p_model.eval()
        for epoch in range(kl_epochs):
            for tmp in train_dataloader:
                for batch_idx, (x, target) in enumerate(tmp):
                    x, target = x.to(device), target.to(device)
                    x.requires_grad = True
                    target.requires_grad = False
                    target = target.long()
                    output_g = net(x)
                    last_p = back_p_model(x)

                    optimizer.zero_grad()
                    loss = kl_loss(F.log_softmax(output_g, dim=1), F.softmax(last_p.detach(), dim=1))
                    loss.backward()
                    optimizer.step()

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

    correct, total, _ = compute_accuracy_pfedKL(net, p_net, test_dataloader, device=device)
    test_acc = correct[0]/float(total)

    if args.train_acc_pre:
        correct, total, _ = compute_accuracy_pfedKL(net, p_net, train_dataloader, device=device)
        train_acc = correct[0]/float(total)
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

    correct, total, _ = compute_accuracy_pfedKL(net, p_net, test_dataloader, device=device)
    test_acc = correct[0]/float(total)

    if args.train_acc_pre:
        correct, total, _ = compute_accuracy_pfedKL(net, p_net, train_dataloader, device=device)
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

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level)

        n_epoch = args.epochs
        trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl_local, n_epoch, args.lr, args.optimizer, args, device=device)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
        # saving the trained models here
        # save_model(net, net_id, args)
        # else:
        #     load_model(net, net_id, device=device)
    avg_acc /= len(selected)
    # if args.alg == 'local_training':
    #     logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list


def local_train_net_per(nets, selected, args, net_dataidx_map_train, net_dataidx_map_test, logger=None, device="cpu"):
    avg_acc = 0.0
    n_epoch = args.epochs
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        # if args.log_flag:
        #     logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs_train)))
        # print("Training network %s. n_training: %d" % (str(net_id), len(dataidxs_train)))
        # move the model to cuda device:
        net.to(device)

        if args.dataset == "shakespeare":
            train_dl_local = net_dataidx_map_train[net_id]
            test_dl_local = net_dataidx_map_test[net_id]
            trainacc, testacc = train_net_shakes(net_id, net, train_dl_local, test_dl_local, 
                n_epoch, args.lr, args.optimizer, args, device=device)
        else:
            dataidxs_train = net_dataidx_map_train[net_id]
            dataidxs_test = net_dataidx_map_test[net_id]
            noise_level = args.noise
            if net_id == args.n_parties - 1:
                noise_level = 0

            if args.model in ['mobilent', 'cnn-b']:
                if args.noise_type == 'space':
                    train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 2*args.batch_size, dataidxs_train, dataidxs_test, noise_level, net_id, args.n_parties-1, drop_last=True)
                elif args.noise_type == 'increasing':
                    noise_level = args.noise / (args.n_parties - 1) * net_id
                    train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 2*args.batch_size, dataidxs_train, dataidxs_test, noise_level, drop_last=True, apply_noise=True)
                else:
                    noise_level = 0
                    train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 2*args.batch_size, dataidxs_train, dataidxs_test, noise_level, drop_last=True)
        
            else:
                if args.noise_type == 'space':
                    train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, net_id, args.n_parties-1)
                elif args.noise_type == 'increasing':
                    noise_level = args.noise / (args.n_parties - 1) * net_id
                    train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, apply_noise=True)
                else:
                    noise_level = 0
                    train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level)
            
            trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl_local, n_epoch, 
                args.lr, args.optimizer, args, device=device)
        
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc

    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list


def local_train_net_per_kl(nets, p_nets, selected, args, net_dataidx_map_train, net_dataidx_map_test, a_iter, logger=None, device="cpu"):
    n_epoch = args.epochs
    kl_epoch = args.kl_epochs
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        # if args.log_flag:
        #     logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs_train)))
        # print("Training network %s. n_training: %d" % (str(net_id), len(dataidxs_train)))
        # move the model to cuda device:
        net.to(device)
        p_net = p_nets[net_id]
        p_net.to(device)

        dataidxs_train = net_dataidx_map_train[net_id]
        dataidxs_test = net_dataidx_map_test[net_id]
        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.model in ['mobilent', 'cnn-b']:
            if args.noise_type == 'space':
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 2*args.batch_size, dataidxs_train, dataidxs_test, noise_level, net_id, args.n_parties-1, drop_last=True)
            elif args.noise_type == 'increasing':
                noise_level = args.noise / (args.n_parties - 1) * net_id
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 2*args.batch_size, dataidxs_train, dataidxs_test, noise_level, drop_last=True, apply_noise=True)
            else:
                noise_level = 0
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 2*args.batch_size, dataidxs_train, dataidxs_test, noise_level, drop_last=True)

        if args.version == 1:
            _, testacc = train_net_pfedKL_full(net_id, net, p_net, train_dl_local, test_dl_local, n_epoch, kl_epoch,
            args.lr, args.optimizer, args, a_iter, device=device)
        elif args.version == 2:
            _, testacc = train_net_pfedKL_p(net_id, net, p_net, train_dl_local, test_dl_local, n_epoch, kl_epoch,
            args.lr, args.optimizer, args, a_iter, device=device)
        elif args.version == 3:
            _, testacc = train_net_pfedKL_g(net_id, net, p_net, train_dl_local, test_dl_local, n_epoch, kl_epoch,
            args.lr, args.optimizer, args, a_iter, device=device)
        
        logger.info("net %d final test acc %f" % (net_id, testacc))

    nets_list = list(nets.values())
    return nets_list


def local_train_net_per_2branch(nets, p_nets, selected, args, net_dataidx_map_train, net_dataidx_map_test, a_iter, logger=None, device="cpu"):
    n_epoch = args.epochs
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        # if args.log_flag:
        #     logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs_train)))
        # print("Training network %s. n_training: %d" % (str(net_id), len(dataidxs_train)))
        # move the model to cuda device:
        net.to(device)
        p_net = p_nets[net_id]
        p_net.to(device)

        dataidxs_train = net_dataidx_map_train[net_id]
        dataidxs_test = net_dataidx_map_test[net_id]
        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.model in ['mobilent', 'cnn-b']:
            if args.noise_type == 'space':
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 2*args.batch_size, dataidxs_train, dataidxs_test, noise_level, net_id, args.n_parties-1, drop_last=True)
            elif args.noise_type == 'increasing':
                noise_level = args.noise / (args.n_parties - 1) * net_id
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 2*args.batch_size, dataidxs_train, dataidxs_test, noise_level, drop_last=True, apply_noise=True)
            else:
                noise_level = 0
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 2*args.batch_size, dataidxs_train, dataidxs_test, noise_level, drop_last=True)

        _, testacc = train_net_2branch(net_id, net, p_net, train_dl_local, test_dl_local, n_epoch, 
        args.lr, args.optimizer, args, device=device)
        
        logger.info("net %d final test acc %f" % (net_id, testacc))

    nets_list = list(nets.values())
    return nets_list


def local_train_net_perplus(nets, selected, args, net_dataidx_map_train, net_dataidx_map_test, round, prototype_dict, class_id_dict, logger=None, device="cpu"):
    avg_acc = 0.0

    update_flag = False
    if round == args.beginning_round-1:
        update_flag = True
    if round % args.update_round == 0 and round >= args.beginning_round:
        update_flag = True

    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs_train = net_dataidx_map_train[net_id]
        dataidxs_test = net_dataidx_map_test[net_id]

        net.to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.k_neighbor:
            if args.noise_type == 'space':
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 2*args.batch_size, dataidxs_train, dataidxs_test, noise_level, net_id, args.n_parties-1)
            else:
                noise_level = args.noise / (args.n_parties - 1) * net_id
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 2*args.batch_size, dataidxs_train, dataidxs_test, noise_level)

        else:
            if args.noise_type == 'space':
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, net_id, args.n_parties-1)
            else:
                noise_level = args.noise / (args.n_parties - 1) * net_id
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level)
        

        n_epoch = args.epochs
        trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl_local, n_epoch, args.lr, args.optimizer, args, device=device)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc

        if update_flag: 
            if args.dataset == "cifar10":
                if args.partition == 'noniid-labeluni' and args.position_embedding:
                    class_id = {}
                else:
                    class_id = {idd:[torch.zeros(1,128).to(device), 1] for idd in range(10)}
            elif args.dataset == "cifar100":
                if args.partition == 'noniid-labeluni' and args.position_embedding:
                    class_id = {}
                else:
                    class_id = {idd:[torch.zeros(1,128).to(device), 1] for idd in range(100)}

            class_id_list = {}

            with torch.no_grad():
                for i, data in enumerate(train_dl_local):
                    img, label = tuple(t.to(device) for t in data)
                    feature = net.produce_feature(img).detach()
                    for ins in range(len(label)):
                        c_id = int(label[ins])
                        if c_id in class_id.keys():
                            class_id[c_id][0] += feature[ins]
                            class_id[c_id][1] += 1
                            class_id_list[c_id] = 1
                        else:
                            class_id[c_id] = [feature[ins].view(1,-1),1]

                if args.similarity:
                    if args.dataset == "cifar10":
                        new_client_p = torch.eye(10).to(device)
                        c_number = 10
                    elif args.dataset == "cifar100":
                        new_client_p = torch.eye(100).to(device)
                        c_number = 100

                    for ik in range(c_number):
                        if ik not in class_id_list.keys():
                            continue
                        proto_c1 = class_id[ik][0]/class_id[ik][1]
                        for jk in range(ik+1,c_number):
                            if jk not in class_id_list.keys():
                                continue
                            proto_c2 = class_id[jk][0]/class_id[jk][1]
                            # print(proto_c1.shape)
                            cos_sim = torch.cosine_similarity(proto_c1, proto_c2, dim=1)
                            # print(cos_sim)
                            new_client_p[ik][jk] = cos_sim.data
                            new_client_p[jk][ik] = cos_sim.data
                    # print("reshape before: ", new_client_p)
                    new_client_p = new_client_p.view(1,-1)
                    # print("reshape after: ", new_client_p)
                    # assert False

                else:
                    new_client_p = 0
                    ff = 0
                    if args.partition == 'noniid-labeluni' and args.position_embedding:
                        for cc in class_id.keys():
                            class_id[cc][0] = (class_id[cc][0]/class_id[cc][1])
                            if ff == 0:
                                new_client_p = class_id[cc][0]
                                ff = 1
                            else:
                                new_client_p = torch.cat((new_client_p, class_id[cc][0]), 1)
                    else:
                        for cc in range(len(class_id.keys())):
                            class_id[cc][0] = class_id[cc][0]/class_id[cc][1]
                            if ff == 0:
                                new_client_p = class_id[cc][0]
                                ff = 1
                            else:
                                new_client_p = torch.cat((new_client_p, class_id[cc][0]), 1)

                    # elif args.partition == 'noniid-labeluni':
                    #     for cc in class_id.keys():
                    #         class_id[cc][0] = class_id[cc][0]/class_id[cc][1]
                    #         if ff == 0:
                    #             new_client_p = class_id[cc][0]
                    #             ff = 1
                    #         else:
                    #             new_client_p = torch.cat((new_client_p, class_id[cc][0]), 1)
            # print("<><><>", new_client_p)
            prototype_dict[net_id] = new_client_p
            if round == args.beginning_round-1:
                class_id_dict[net_id] = list(class_id_list.keys())

    if update_flag:
        for net_id, net in nets.items():
            if net_id in selected:
                continue

            dataidxs_train = net_dataidx_map_train[net_id]
            dataidxs_test = net_dataidx_map_test[net_id]
            net.to(device)

            noise_level = args.noise
            if net_id == args.n_parties - 1:
                noise_level = 0

            if args.noise_type == 'space':
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, net_id, args.n_parties-1)
            else:
                noise_level = args.noise / (args.n_parties - 1) * net_id
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level)


            if args.dataset == "cifar10":
                if args.partition == 'noniid-labeluni' and args.position_embedding:
                    class_id = {}
                else:
                    class_id = {idd:[torch.zeros(1,128).to(device), 1] for idd in range(10)}
            elif args.dataset == "cifar100":
                if args.partition == 'noniid-labeluni' and args.position_embedding:
                    class_id = {}
                else:
                    class_id = {idd:[torch.zeros(1,128).to(device), 1] for idd in range(100)}

            class_id_list = {}

            with torch.no_grad():
                for i, data in enumerate(train_dl_local):
                    img, label = tuple(t.to(device) for t in data)
                    feature = net.produce_feature(img).detach()
                    for ins in range(len(label)):
                        c_id = int(label[ins])
                        if c_id in class_id.keys():
                            class_id[c_id][0] += feature[ins]
                            class_id[c_id][1] += 1
                        else:
                            class_id[c_id] = [feature[ins].view(1,-1),1]
                            class_id_list[c_id] = 1
                            
                if args.similarity:
                    if args.dataset == "cifar10":
                        new_client_p = torch.eye(10).to(device)
                        c_number = 10
                    elif args.dataset == "cifar100":
                        new_client_p = torch.eye(100).to(device)
                        c_number = 100

                    for ik in range(c_number):
                        if ik not in class_id_list.keys():
                            continue
                        proto_c1 = class_id[ik][0]/class_id[ik][1]
                        for jk in range(ik+1,c_number):
                            if jk not in class_id_list.keys():
                                continue
                            proto_c2 = class_id[jk][0]/class_id[jk][1]
                            # print(proto_c1.shape)
                            cos_sim = torch.cosine_similarity(proto_c1, proto_c2, dim=1)
                            # print(cos_sim)
                            new_client_p[ik][jk] = cos_sim.data
                            new_client_p[jk][ik] = cos_sim.data
                    # print("reshape before: ", new_client_p)
                    new_client_p = new_client_p.view(1,-1)
                    # print("reshape after: ", new_client_p)
                    # assert False

                else:
                    new_client_p = 0
                    ff = 0               
                    # print(class_id_list.keys())
                    if args.partition == 'noniid-labeluni' and args.position_embedding:
                        for cc in class_id.keys():
                            class_id[cc][0] = (class_id[cc][0]/class_id[cc][1])
                            if ff == 0:
                                new_client_p = class_id[cc][0]
                                ff = 1
                            else:
                                new_client_p = torch.cat((new_client_p, class_id[cc][0]), 1)
                    else:
                        for cc in range(len(class_id.keys())):
                            class_id[cc][0] = class_id[cc][0]/class_id[cc][1]
                            if ff == 0:
                                new_client_p = class_id[cc][0]
                                ff = 1
                            else:
                                new_client_p = torch.cat((new_client_p, class_id[cc][0]), 1)

                    # elif args.partition == 'noniid-labeluni':
                    #     for cc in class_id.keys():
                    #         class_id[cc][0] = class_id[cc][0]/class_id[cc][1]
                    #         if ff == 0:
                    #             new_client_p = class_id[cc][0]
                    #             ff = 1
                    #         else:
                    #             new_client_p = torch.cat((new_client_p, class_id[cc][0]), 1)
            prototype_dict[net_id] = new_client_p
            if round == args.beginning_round-1:
                class_id_dict[net_id] = list(class_id_list.keys())

    avg_acc /= len(selected)
    nets_list = list(nets.values())
    return nets_list


def local_train_net_fedprox(nets, selected, global_model, args, net_dataidx_map_train, net_dataidx_map_test, logger=None, device="cpu"):
    avg_acc = 0.0
    n_epoch = args.epochs

    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        net.to(device)

        if args.dataset == "shakespeare":
            train_dl_local = net_dataidx_map_train[net_id]
            test_dl_local = net_dataidx_map_test[net_id]
            trainacc, testacc = train_net_shakes(net_id, net, train_dl_local, test_dl_local, 
            n_epoch, args.lr, args.optimizer, args, device=device, global_net=global_model, fedprox=True)
        else:
            dataidxs_train = net_dataidx_map_train[net_id]
            dataidxs_test = net_dataidx_map_test[net_id]
            noise_level = args.noise
            if net_id == args.n_parties - 1:
                noise_level = 0
            if args.noise_type == 'space':
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, net_id, args.n_parties-1)
            else:
                noise_level = args.noise / (args.n_parties - 1) * net_id
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level)
            
            trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl_local, 
            n_epoch, args.lr, args.optimizer, args, device=device, global_net=global_model, fedprox=True)
        
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list


def local_train_net_pfedMe(nets, selected, global_model, args, net_dataidx_map_train, net_dataidx_map_test, logger=None, device="cpu"):
    avg_acc = 0.0
    n_epoch = args.epochs
    update_dict = {}
    global_para = copy.deepcopy(global_model.state_dict())
    for net_id, net in nets.items():
        if net_id not in selected:
            continue

        regularized_local = copy.deepcopy(global_model)
        net.load_state_dict(global_para)
        # if args.log_flag:
        #     logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs_train)))
        # print("Training network %s. n_training: %d" % (str(net_id), len(dataidxs_train)))
        # move the model to cuda device:
        net.to(device)

        if args.dataset == "shakespeare":
            train_dl_local = net_dataidx_map_train[net_id]
            test_dl_local = net_dataidx_map_test[net_id]
        
        else:
            dataidxs_train = net_dataidx_map_train[net_id]
            dataidxs_test = net_dataidx_map_test[net_id]
            noise_level = args.noise
            if net_id == args.n_parties - 1:
                noise_level = 0
        
            if args.noise_type == 'space':
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, net_id, args.n_parties-1)
            else:
                noise_level = args.noise / (args.n_parties - 1) * net_id
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level)

        trainacc, testacc, local_params = train_net_pfedMe(net_id, net, regularized_local, train_dl_local, test_dl_local, n_epoch, 
            args.lr, args.optimizer, args, device=device)
        
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
        update_dict[net_id] = local_params
    avg_acc /= len(selected)

    # nets_list = list(nets.values())
    return update_dict


def local_train_net_fedRod(nets, selected, p_head, args, net_dataidx_map_train, net_dataidx_map_test, logger=None, device="cpu", alpha=None):
    avg_acc = 0.0
    n_epoch = args.epochs
    for net_id, net in nets.items():
        if net_id not in selected:
            continue

        net.to(device)
        personal_head = p_head[net_id]
        sample_per_class = []

        if args.dataset == "shakespeare":
            train_dl_local = net_dataidx_map_train[net_id]
            test_dl_local = net_dataidx_map_test[net_id]
        
        else:
            sample_per_class = alpha[net_id]
            dataidxs_train = net_dataidx_map_train[net_id]
            dataidxs_test = net_dataidx_map_test[net_id]
            noise_level = args.noise
            if net_id == args.n_parties - 1:
                noise_level = 0
        
            if args.noise_type == 'space':
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, net_id, args.n_parties-1)
            else:
                noise_level = args.noise / (args.n_parties - 1) * net_id
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level)

        trainacc, testacc = train_net_fedRod(net_id, net, personal_head, sample_per_class,
        train_dl_local, test_dl_local, n_epoch, args.lr, args.optimizer, args, device=device)
        
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
    avg_acc /= len(selected)

    return p_head


def local_train_net_cluster(nets, selected, args, net_dataidx_map_train, net_dataidx_map_test, prototype_dict=None, class_id_dict=None, logger=None, device="cpu"):
    avg_acc = 0.0
    prototype_dict = {}

    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs_train = net_dataidx_map_train[net_id]
        dataidxs_test = net_dataidx_map_test[net_id]

        net.to(device)
        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level)
        
        n_epoch = args.epochs
        trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl_local, n_epoch, args.lr, args.optimizer, args, device=device)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc

        # class_id = {}
        if args.dataset == "cifar10":
            if args.model == 'cnn':
                class_id = {idd:[torch.zeros(1,84).to(device), 0] for idd in range(10)}
            elif args.model == 'vit':
                class_id = {idd:[torch.zeros(1,128).to(device), 0] for idd in range(10)}
        elif args.dataset == "cifar100":
            if args.model == 'cnn':
                class_id = {idd:[torch.zeros(1,84).to(device), 0] for idd in range(100)}
            elif args.model == 'vit':
                class_id = {idd:[torch.zeros(1,128).to(device), 0] for idd in range(100)}

        with torch.no_grad():
            for i, data in enumerate(train_dl_local):
                img, label = tuple(t.to(device) for t in data)
                feature = net.produce_feature(img).detach()
                for ins in range(len(label)):
                    c_id = int(label[ins])
                    if c_id in class_id.keys():
                        class_id[c_id][0] += feature[ins].view(1,-1)
                        class_id[c_id][1] += 1 
                    else:
                        class_id[c_id] = [feature[ins].view(1,-1),1]

            new_client_p = 0
            ff = 0
            for cc in range(len(class_id.keys())):
                if class_id[cc][1] == 0:
                    class_id[cc][0] = class_id[cc][0]
                else:
                    class_id[cc][0] = class_id[cc][0]/class_id[cc][1]
                if ff == 0:
                    new_client_p = class_id[cc][0]
                    ff = 1
                else:
                    new_client_p = torch.cat((new_client_p, class_id[cc][0]), 0)
        prototype_dict[net_id] = new_client_p
        # print(prototype_dict[0].shape)
        # print(prototype_dict[0])
        # assert False
    avg_acc /= len(selected)
    nets_list = list(nets.values())
    return prototype_dict


def local_train_net_moon(nets, args, net_dataidx_map_train, net_dataidx_map_test, global_model, prev_model_pool, device="cpu"):
    global_model.to(device)
    for net_id, net in nets.items():
        net.to(device)
        dataidxs_train = net_dataidx_map_train[net_id]
        dataidxs_test = net_dataidx_map_test[net_id]

        noise_level = 0
        train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level)
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
            # _, pro1, out = net(x)
            # _, pro2, _ = global_net(x)

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
        # logger.info('Epoch: %d Loss: %f Loss1: %f Loss2: %f' % (epoch, epoch_loss, epoch_loss1, epoch_loss2))

    test_acc  = compute_accuracy(net, test_dataloader, device=device)
    if args.train_acc_pre:
        train_acc = compute_accuracy(net, train_dataloader, device=device)
        return train_acc, test_acc
    else:
        return None, test_acc

