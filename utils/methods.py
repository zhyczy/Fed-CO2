import sys, os
import numpy as np
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict, defaultdict


def local_training(models, personalized_models, paggregation_models, server_model, Extra_modules, args, train_loaders, optimizers, loss_fun, device, a_iter=0):
    client_num = len(models)
    Specific_head = {}

    if args.mode == 'fed-co2':
        for client_idx in range(client_num):
            Specific_head[client_idx] = copy.deepcopy(personalized_models[client_idx].classifier)

    elif args.mode == 'copa':
        for client_idx in range(client_num):
            Specific_head[client_idx] = copy.deepcopy(models[client_idx].head)

    for client_idx, model in enumerate(models):
        if args.mode == 'fedprox':
            # skip the first server model(random initialized)
            if a_iter > 0:
                train_loss, train_acc = train_prox(args, model, server_model, train_loaders[client_idx], optimizers[client_idx], loss_fun, device)
            else:
                train_loss, train_acc = train(model, train_loaders[client_idx], optimizers[client_idx], loss_fun, device)
        
        elif args.mode == 'fed-co2':
            p_optimizer = optim.SGD(params=personalized_models[client_idx].parameters(), lr=args.lr)
            criterion_ba = nn.CrossEntropyLoss().to(device)
            train_loss, train_acc = train_CO2(model, personalized_models[client_idx], train_loaders[client_idx], optimizers[client_idx], 
                                                  p_optimizer, loss_fun, criterion_ba, Specific_head, client_idx, a_iter, device)

        elif args.mode == 'fedper':
            private_params = personalized_models[client_idx].state_dict()
            model.load_state_dict(private_params, strict=False)
            train_loss, train_acc = train(model, train_loaders[client_idx], optimizers[client_idx], loss_fun, device)

        elif args.mode == 'fedrod':
            p_optimizer = optim.SGD(params=personalized_models[client_idx].parameters(), lr=args.lr)
            criterion_ba = nn.CrossEntropyLoss().to(device)
            train_loss, train_acc = train_rod(model, personalized_models[client_idx], train_loaders[client_idx], optimizers[client_idx], p_optimizer, loss_fun, criterion_ba, device)

        elif args.mode == 'moon':
            train_loss, train_acc = train_moon(model, paggregation_models[client_idx], train_loaders[client_idx], optimizers[client_idx], loss_fun, device)

        elif args.mode == 'copa':
            train_loss, train_acc = train_COPA(model, Specific_head, train_loaders[client_idx], optimizers[client_idx], loss_fun, client_idx, device)

        else:
            train_loss, train_acc = train(model, train_loaders[client_idx], optimizers[client_idx], loss_fun, device)

    return train_loss, train_acc


def train_CO2(model, p_model, data_loader, optimizer, p_optimizer, loss_fun, loss_g, Specific_heads, client_idx, a_iter, device):
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    client_num = len(Specific_heads.keys())
    assert client_num != 0
    model.train()
    p_model.train()

    loss_all = 0
    total = 0
    correct = 0

    if a_iter != 0:
        back_g_model = copy.deepcopy(model)
        back_g_model.eval()
        back_p_model = copy.deepcopy(p_model)
        back_p_model.eval()
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            output_g = model(data)
            output_p = p_model(data)

            last_g = back_g_model(data)
            last_p = back_p_model(data)

            optimizer.zero_grad()
            loss = kl_loss(F.log_softmax(output_g, dim=1), F.softmax(last_p.detach(), dim=1))
            loss.backward()
            optimizer.step()

            p_optimizer.zero_grad()
            loss_p = kl_loss(F.log_softmax(output_p, dim=1), F.softmax(last_g.detach(), dim=1))
            loss_p.backward()
            p_optimizer.step()

    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)
        feature_g = model.produce_feature(data)
        output_g = model.classifier(feature_g)
        feature_p = p_model.produce_feature(data)
        output_p = p_model.classifier(feature_p)

        optimizer.zero_grad()
        part1 = loss_g(output_g, target)

        part2 = 0
        part3 = 0
        for idxx in range(client_num):
            if idxx != client_idx:
                Spe_classifier = Specific_heads[idxx]
                Spe_classifier.eval()
                output_gen = Spe_classifier(feature_g)
                output_per = Spe_classifier(feature_p)
                part2 += loss_g(output_gen, target)
                part3 += loss_g(output_per, target)
        loss = part1 + part2
        loss.backward()
        optimizer.step()

        p_optimizer.zero_grad()
        loss_p = loss_fun(output_p, target) + part3
        loss_p.backward()
        p_optimizer.step()

        loss_all += loss_p.item()
        total += target.size(0)
        pred = (output_g.detach()+output_p).data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()
    return loss_all / len(data_loader), correct/total


def train_rod(model, p_model, data_loader, optimizer, p_optimizer, loss_fun, loss_g, device):
    model.train()
    p_model.train()
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        optimizer.zero_grad()

        data = data.to(device)
        target = target.to(device)
        mid_features = model.produce_feature(data)
        output_g = model.classifier(mid_features)
        loss = loss_g(output_g, target)
        loss.backward()
        optimizer.step()

        output_p = p_model(mid_features.detach())
        loss_p = loss_fun(output_g.detach()+output_p, target)
        p_optimizer.zero_grad()
        loss_p.backward()
        p_optimizer.step()

        loss_all += loss_p.item()
        total += target.size(0)
        pred = (output_g.detach()+output_p).data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    return loss_all / len(data_loader), correct/total


def train(model, data_loader, optimizer, loss_fun, device):
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        optimizer.zero_grad()

        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        loss.backward()
        optimizer.step()

    return loss_all / len(data_loader), correct/total


def train_prox(args, model, server_model, data_loader, optimizer, loss_fun, device):
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    for step, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()

        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = loss_fun(output, target)
        if step>0:
            w_diff = torch.tensor(0., device=device)
            for w, w_t in zip(server_model.parameters(), model.parameters()):
                w_diff += torch.pow(torch.norm(w - w_t), 2)
                
            w_diff = torch.sqrt(w_diff)
            loss += args.mu / 2. * w_diff
                        
        loss.backward()
        optimizer.step()

        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    return loss_all / len(data_loader), correct/total


def train_COPA(model, Specific_heads, data_loader, optimizer, loss_fun, client_idx, device):
    client_num = len(Specific_heads.keys())
    assert client_num != 0
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        optimizer.zero_grad()

        data = data.to(device)
        target = target.to(device)
        local_feature = model.produce_feature(data)
        output = model.head(local_feature)
        part1 = loss_fun(output, target)

        part2 = 0
        for idxx in range(client_num):
            if idxx != client_idx:
                Spe_classifier = Specific_heads[idxx]
                Spe_classifier.eval()
                output_gen = Spe_classifier(local_feature)
                part2 += loss_fun(output_gen, target)
        loss = part1 + part2

        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        loss.backward()
        optimizer.step()

    return loss_all / len(data_loader), correct/total


def train_moon(model, previous_model, data_loader, optimizer, loss_fun, device, temperature=0.5, mu=1):
    cos=torch.nn.CosineSimilarity(dim=-1)
    global_model = copy.deepcopy(model)
    global_model.eval()
    for param in global_model.parameters():
        param.requires_grad = False
    previous_model.eval()
    for param in previous_model.parameters():
        param.requires_grad = False
    model.train()

    loss_all = 0
    total = 0
    correct = 0

    for data, target in data_loader:
        optimizer.zero_grad()

        data = data.to(device)
        target = target.to(device)

        pro1 = model.produce_feature(data)
        output = model.classifier(pro1)
        pro2 = global_model.produce_feature(data)
        posi = cos(pro1, pro2)
        logits = posi.reshape(-1,1)

        pro3 = previous_model.produce_feature(data)
        nega = cos(pro1, pro3)
        logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)
        logits /= temperature
        labels = torch.zeros(data.size(0)).cuda().long()

        loss2 = mu * loss_fun(logits, labels)
        loss1 = loss_fun(output, target)
        loss = loss1 + loss2

        loss.backward()
        optimizer.step()

        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    return loss_all / len(data_loader), correct/total

