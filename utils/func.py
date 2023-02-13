import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict, defaultdict


def others_train(version, model, p_model, Extra_modules, train_loader, test_loader, optimizer, p_optimizer, loss_fun, criterion_ba, Specific_head, Specific_adaptor, valid_value, client_idx, device, args, a_iter):
    if version == 46:
        train_loss, train_acc = train_tt(model, p_model, train_loader, test_loader, 
                                optimizer, p_optimizer, loss_fun, criterion_ba, Specific_head, client_idx, device, a_iter)

    elif version == 47:
        train_loss, train_acc = train_tt_kl(model, p_model, train_loader, test_loader, 
                                optimizer, p_optimizer, loss_fun, criterion_ba, Specific_head, valid_value, client_idx, device, a_iter)

    elif version == 48:
        train_loss, train_acc = train_tt_kl1(model, p_model, train_loader, test_loader, 
                                optimizer, p_optimizer, loss_fun, criterion_ba, Specific_head, valid_value, client_idx, device, a_iter)

    elif version == 49:
        train_loss, train_acc = train_tt_kll(model, p_model, train_loader, test_loader, 
                                optimizer, p_optimizer, loss_fun, criterion_ba, Specific_head, valid_value, client_idx, device, a_iter)

    elif version == 1:
        optimizer = optim.SGD(params=[{'params':model.parameters()},
                              {'params':p_model.c_adaptor.parameters()}], lr=args.lr, momentum=args.momentum)
        p_optimizer = optim.SGD(params=[{'params':p_model.features.parameters()},
                             {'params':p_model.classifier.parameters()},
                             {'params':p_model.f_adaptor.parameters()}], lr=args.lr)
        train_loss, train_acc = train_ada0(model, p_model, train_loader, optimizer, 
                                          p_optimizer, loss_fun, criterion_ba, Specific_head, client_idx, device)

    elif version == 2:
        a_otimizer = optim.SGD(params=[{'params':p_model.f_adaptor.parameters()},
                              {'params':p_model.c_adaptor.parameters()}], lr=args.lr, momentum=args.momentum)
        p_optimizer = optim.SGD(params=[{'params':p_model.features.parameters()},
                             {'params':p_model.classifier.parameters()}], lr=args.lr)
        train_loss, train_acc = train_ada1(model, p_model, train_loader, optimizer, 
                                          p_optimizer, a_otimizer, loss_fun, criterion_ba, Specific_head, client_idx, device)

    elif version == 3:
        optimizer = optim.SGD(params=[{'params':model.conv1.parameters()},
                                    {'params':model.bn1.parameters()},
                                    {'params':model.conv2.parameters()},
                                    {'params':model.bn2.parameters()},
                                    {'params':model.conv3.parameters()},
                                    {'params':model.bn3.parameters()},
                                    {'params':model.conv4.parameters()},
                                    {'params':model.bn4.parameters()},
                                    {'params':model.conv5.parameters()},
                                    {'params':model.bn5.parameters()},
                                    {'params':model.classifier.parameters()},
                                    {'params':p_model.c_adaptor.parameters()}], lr=args.lr, momentum=args.momentum)

        p_optimizer = optim.SGD(params=[{'params':p_model.features.parameters()},
                                       {'params':p_model.classifier.parameters()},
                                       {'params':model.adap3.parameters()},
                                       {'params':model.adap4.parameters()},
                                       {'params':model.adap5.parameters()}], lr=args.lr)
        train_loss, train_acc = train_ada2(model, p_model, train_loader, optimizer, 
                                          p_optimizer, loss_fun, criterion_ba, Specific_head, client_idx, device)

    elif version == 4:
        optimizer = optim.SGD(params=[{'params':model.conv1.parameters()},
                                    {'params':model.bn1.parameters()},
                                    {'params':model.conv2.parameters()},
                                    {'params':model.bn2.parameters()},
                                    {'params':model.conv3.parameters()},
                                    {'params':model.bn3.parameters()},
                                    {'params':model.conv4.parameters()},
                                    {'params':model.bn4.parameters()},
                                    {'params':model.conv5.parameters()},
                                    {'params':model.bn5.parameters()},
                                    {'params':model.classifier.parameters()}], lr=args.lr, momentum=args.momentum)
        a_otimizer = optim.SGD(params=[{'params':model.adap3.parameters()},
                                       {'params':model.adap4.parameters()},
                                       {'params':model.adap5.parameters()},
                                       {'params':p_model.c_adaptor.parameters()}], lr=args.lr, momentum=args.momentum)
        p_optimizer = optim.SGD(params=[{'params':p_model.features.parameters()},
                             {'params':p_model.classifier.parameters()}], lr=args.lr)
        train_loss, train_acc = train_ada3(model, p_model, train_loader, optimizer, 
                                          p_optimizer, a_otimizer, loss_fun, criterion_ba, Specific_head, client_idx, device)

    elif args.version == 23:
        a_otimizer = optim.SGD(params=Extra_modules[client_idx].parameters(), lr=args.lr, momentum=args.momentum)
        train_loss, train_acc = train_ada_shabby1(model, p_model, Extra_modules[client_idx], train_loader, optimizer, 
                                          p_optimizer, a_otimizer, loss_fun, criterion_ba, Specific_head, client_idx, device)

    elif args.version == 26:
        a_otimizer = optim.SGD(params=Extra_modules[client_idx].parameters(), lr=args.lr, momentum=args.momentum)
        train_loss, train_acc = train_ada_shabby3(model, p_model, Extra_modules[client_idx], train_loader, optimizer, 
                                          p_optimizer, a_otimizer, loss_fun, criterion_ba, Specific_head, client_idx, device)

    return train_loss, train_acc


def train_tt(model, p_model, data_loader, unlabeled_loader, optimizer, p_optimizer, loss_fun, loss_g, Specific_heads, client_idx, device, a_iter):
    clinet_num = len(Specific_heads.keys())
    assert clinet_num != 0
    l_lambda = 1/(clinet_num-1)
    model.train()
    p_model.train()
    loss_all = 0
    total = 0
    correct = 0
    g_pseudo_dict = {}
    p_pseudo_dict = {}
    if a_iter >= 100:
        with torch.no_grad():
            inner_count = 0
            for data, _ in unlabeled_loader:
                if data.shape[0] != 32:
                    continue
                data = data.to(device)
                # print(data.shape)
                # assert False
                g_logits = model(data)
                p_logits = p_model(data)
                g_pseudo_dict[inner_count] = g_logits.data.max(1)[1]
                p_pseudo_dict[inner_count] = p_logits.data.max(1)[1]
                inner_count += 1

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
        for idxx in range(clinet_num):
            if idxx != client_idx:
                Spe_classifier = Specific_heads[idxx]
                output_gen = Spe_classifier(feature_g)
                # part2 += l_lambda * loss_g(output_gen, target)
                part2 += loss_g(output_gen, target)
        loss = part1 + part2
        loss.backward()
        optimizer.step()

        p_optimizer.zero_grad()
        loss_p = loss_fun(output_p, target)
        loss_p.backward()
        p_optimizer.step()

        loss_all += loss_p.item()
        total += target.size(0)
        pred = (output_g.detach()+output_p).data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    if a_iter >= 100:
        inner_count = 0
        for data, _ in unlabeled_loader:
            if data.shape[0] != 32:
                continue
            data = data.to(device)
            g_pseudo = g_pseudo_dict[inner_count].to(device)
            p_pseudo = p_pseudo_dict[inner_count].to(device)
            inner_count += 1

            output_g = model(data)
            output_p = p_model(data)

            optimizer.zero_grad()
            loss = loss_g(output_g, g_pseudo)
            loss.backward()
            optimizer.step()

            p_optimizer.zero_grad()
            loss_p = loss_fun(output_p, p_pseudo)
            loss_p.backward()
            p_optimizer.step()

            # loss_all += loss_p.item()
            # total += target.size(0)
            # pred = (output_g.detach()+output_p).data.max(1)[1]
            # correct += pred.eq(target.view(-1)).sum().item()

    return loss_all / len(data_loader), correct/total


def train_tt_kl(model, p_model, data_loader, unlabeled_loader, optimizer, p_optimizer, loss_fun, loss_g, Specific_heads, onehot_value, client_idx, device, a_iter):
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    clinet_num = len(Specific_heads.keys())
    assert clinet_num != 0
    l_lambda = 1/(clinet_num-1)
    model.train()
    p_model.train()
    loss_all = 0
    total = 0
    correct = 0
    g_pseudo_dict = {}
    p_pseudo_dict = {}
    if a_iter >= 100:
        with torch.no_grad():
            inner_count = 0
            for data, _ in unlabeled_loader:
                if data.shape[0] != 32:
                    continue
                data = data.to(device)
                # print(data.shape)
                # assert False
                g_logits = model(data)
                p_logits = p_model(data)
                g_pseudo_dict[inner_count] = g_logits.data.max(1)[1]
                p_pseudo_dict[inner_count] = p_logits.data.max(1)[1]
                inner_count += 1

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
        for idxx in range(clinet_num):
            if idxx != client_idx:
                Spe_classifier = Specific_heads[idxx]
                output_gen = Spe_classifier(feature_g)
                # part2 += l_lambda * loss_g(output_gen, target)
                part2 += loss_g(output_gen, target)
        loss = part1 + part2
        loss.backward()
        optimizer.step()

        p_optimizer.zero_grad()
        loss_p = loss_fun(output_p, target)
        loss_p.backward()
        p_optimizer.step()

        loss_all += loss_p.item()
        total += target.size(0)
        pred = (output_g.detach()+output_p).data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    if a_iter >= 100:
        inner_count = 0
        for data, _ in unlabeled_loader:
            if data.shape[0] != 32:
                continue
            data = data.to(device)
            g_pseudo = g_pseudo_dict[inner_count].to(device)
            p_pseudo = p_pseudo_dict[inner_count].to(device)
            inner_count += 1

            feature_g = model.produce_feature(data)
            output_g = model.classifier(feature_g)
            feature_p = p_model.produce_feature(data)
            output_p = p_model.classifier(feature_p)

            optimizer.zero_grad()
            loss = loss_g(output_g, g_pseudo)
            if onehot_value == 0:
                part3 = kl_loss(F.log_softmax(feature_g, dim=1), F.softmax(feature_p.detach(), dim=1))
                loss += part3
            loss.backward()
            optimizer.step()

            p_optimizer.zero_grad()
            loss_p = loss_fun(output_p, p_pseudo)
            if onehot_value == 1:
                part3 = kl_loss(F.log_softmax(feature_p, dim=1), F.softmax(feature_g.detach(), dim=1))
                loss_p += part3
            loss_p.backward()
            p_optimizer.step()

    return loss_all / len(data_loader), correct/total


def train_tt_kl1(model, p_model, data_loader, unlabeled_loader, optimizer, p_optimizer, loss_fun, loss_g, Specific_heads, onehot_value, client_idx, device, a_iter):
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    clinet_num = len(Specific_heads.keys())
    assert clinet_num != 0
    l_lambda = 1/(clinet_num-1)
    model.train()
    p_model.train()
    loss_all = 0
    total = 0
    correct = 0

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
        for idxx in range(clinet_num):
            if idxx != client_idx:
                Spe_classifier = Specific_heads[idxx]
                output_gen = Spe_classifier(feature_g)
                # part2 += l_lambda * loss_g(output_gen, target)
                part2 += loss_g(output_gen, target)
        loss = part1 + part2
        loss.backward()
        optimizer.step()

        p_optimizer.zero_grad()
        loss_p = loss_fun(output_p, target)
        loss_p.backward()
        p_optimizer.step()

        loss_all += loss_p.item()
        total += target.size(0)
        pred = (output_g.detach()+output_p).data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    if a_iter >= 100:
        inner_count = 0
        for data, _ in unlabeled_loader:
            if data.shape[0] != 32:
                continue
            data = data.to(device)
            feature_g = model.produce_feature(data)
            feature_p = p_model.produce_feature(data)

            if onehot_value == 0:
                optimizer.zero_grad()
                loss = kl_loss(F.log_softmax(feature_g, dim=1), F.softmax(feature_p.detach(), dim=1))
                loss.backward()
                optimizer.step()

            elif onehot_value == 1:
                p_optimizer.zero_grad()
                loss_p = kl_loss(F.log_softmax(feature_p, dim=1), F.softmax(feature_g.detach(), dim=1))
                loss_p.backward()
                p_optimizer.step()

    return loss_all / len(data_loader), correct/total


def train_tt_kll(model, p_model, data_loader, unlabeled_loader, optimizer, p_optimizer, loss_fun, loss_g, Specific_heads, onehot_value, client_idx, device, a_iter):
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    clinet_num = len(Specific_heads.keys())
    assert clinet_num != 0
    l_lambda = 1/(clinet_num-1)
    model.train()
    p_model.train()
    loss_all = 0
    total = 0
    correct = 0
    g_pseudo_dict = {}
    p_pseudo_dict = {}
    if a_iter >= 100:
        with torch.no_grad():
            inner_count = 0
            for data, _ in unlabeled_loader:
                if data.shape[0] != 32:
                    continue
                data = data.to(device)
                # print(data.shape)
                # assert False
                g_logits = model(data)
                p_logits = p_model(data)
                g_pseudo_dict[inner_count] = g_logits.data.max(1)[1]
                p_pseudo_dict[inner_count] = p_logits.data.max(1)[1]
                inner_count += 1

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
        for idxx in range(clinet_num):
            if idxx != client_idx:
                Spe_classifier = Specific_heads[idxx]
                output_gen = Spe_classifier(feature_g)
                # part2 += l_lambda * loss_g(output_gen, target)
                part2 += loss_g(output_gen, target)
        loss = part1 + part2
        loss.backward()
        optimizer.step()

        p_optimizer.zero_grad()
        loss_p = loss_fun(output_p, target)
        loss_p.backward()
        p_optimizer.step()

        loss_all += loss_p.item()
        total += target.size(0)
        pred = (output_g.detach()+output_p).data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    if a_iter >= 100:
        inner_count = 0
        for data, _ in unlabeled_loader:
            if data.shape[0] != 32:
                continue
            data = data.to(device)
            g_pseudo = g_pseudo_dict[inner_count].to(device)
            p_pseudo = p_pseudo_dict[inner_count].to(device)
            inner_count += 1

            output_g = model(data)
            output_p = p_model(data)

            optimizer.zero_grad()
            loss = loss_g(output_g, g_pseudo)
            if onehot_value == 0:
                part3 = kl_loss(F.log_softmax(output_g, dim=1), F.softmax(output_p.detach(), dim=1))
                loss += part3
            loss.backward()
            optimizer.step()

            p_optimizer.zero_grad()
            loss_p = loss_fun(output_p, p_pseudo)
            if onehot_value == 1:
                part3 = kl_loss(F.log_softmax(output_p, dim=1), F.softmax(output_g.detach(), dim=1))
                loss_p += part3
            loss_p.backward()
            p_optimizer.step()

    return loss_all / len(data_loader), correct/total


def train_ada0(model, p_model, data_loader, optimizer, p_optimizer, loss_fun, loss_g, Specific_heads, client_idx, device):
    clinet_num = len(Specific_heads.keys())
    assert clinet_num != 0
    l_lambda = 1/(clinet_num-1)
    model.train()
    p_model.train()
    loss_all = 0
    total = 0
    correct = 0
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
        for idxx in range(clinet_num):
            if idxx != client_idx:
                Spe_classifier = Specific_heads[idxx]
                output_gen = Spe_classifier(feature_g)
                # part2 += l_lambda * loss_g(output_gen, target)
                part2 += loss_g(output_gen, target)
        adapt_logits = p_model.c_adaptor(output_p.detach())
        part3 = loss_g(adapt_logits, target)
        loss = part1 + part2 + part3
        loss.backward()
        optimizer.step()

        p_optimizer.zero_grad()
        loss_p = loss_fun(output_p, target)
        adapt_feature = p_model.f_adaptor(feature_g.detach())
        part2 = loss_fun(p_model.classifier(adapt_feature), target)
        loss_p += part2
        loss_p.backward()
        p_optimizer.step()

        loss_all += loss_p.item()
        total += target.size(0)
        pred = (output_g.detach()+output_p).data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()
    return loss_all / len(data_loader), correct/total


def train_ada1(model, p_model, data_loader, optimizer, p_optimizer, a_otimizer, loss_fun, loss_g, Specific_heads, client_idx, device):
    clinet_num = len(Specific_heads.keys())
    assert clinet_num != 0
    l_lambda = 1/(clinet_num-1)
    model.train()
    p_model.train()
    loss_all = 0
    total = 0
    correct = 0
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
        for idxx in range(clinet_num):
            if idxx != client_idx:
                Spe_classifier = Specific_heads[idxx]
                output_gen = Spe_classifier(feature_g)
                # part2 += l_lambda * loss_g(output_gen, target)
                part2 += loss_g(output_gen, target)
        loss = part1 + part2 
        loss.backward()
        optimizer.step()

        p_optimizer.zero_grad()
        loss_p = loss_fun(output_p, target)
        loss_p.backward()
        p_optimizer.step()

        a_otimizer.zero_grad()
        adapt_logits = p_model.c_adaptor(output_p.detach())
        adapt_loss = loss_g(adapt_logits, target)
        adapt_feature = p_model.f_adaptor(feature_g.detach())
        adapt_loss += loss_fun(p_model.classifier(adapt_feature), target)
        adapt_loss.backward()
        a_otimizer.step()

        loss_all += loss_p.item()
        total += target.size(0)
        pred = (output_g.detach()+output_p).data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()
    return loss_all / len(data_loader), correct/total


def train_ada2(model, p_model, data_loader, optimizer, p_optimizer, loss_fun, loss_g, Specific_heads, client_idx, device):
    clinet_num = len(Specific_heads.keys())
    assert clinet_num != 0
    l_lambda = 1/(clinet_num-1)
    model.train()
    p_model.train()
    loss_all = 0
    total = 0
    correct = 0
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
        for idxx in range(clinet_num):
            if idxx != client_idx:
                Spe_classifier = Specific_heads[idxx]
                output_gen = Spe_classifier(feature_g)
                # part2 += l_lambda * loss_g(output_gen, target)
                part2 += loss_g(output_gen, target)
        adapt_logits = p_model.c_adaptor(output_p.detach())
        part3 = loss_g(adapt_logits, target)
        loss = part1 + part2 + part3
        loss.backward()
        optimizer.step()

        p_optimizer.zero_grad()
        loss_p = loss_fun(output_p, target)
        adapt_feature = model.produce_adapt_feature(data)
        part2 = loss_fun(p_model.classifier(adapt_feature), target)
        loss_p += part2
        loss_p.backward()
        p_optimizer.step()

        loss_all += loss_p.item()
        total += target.size(0)
        pred = (output_g.detach()+output_p).data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()
    return loss_all / len(data_loader), correct/total


def train_ada3(model, p_model, data_loader, optimizer, p_optimizer, a_otimizer, loss_fun, loss_g, Specific_heads, client_idx, device):
    clinet_num = len(Specific_heads.keys())
    assert clinet_num != 0
    l_lambda = 1/(clinet_num-1)
    model.train()
    p_model.train()
    loss_all = 0
    total = 0
    correct = 0
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
        for idxx in range(clinet_num):
            if idxx != client_idx:
                Spe_classifier = Specific_heads[idxx]
                output_gen = Spe_classifier(feature_g)
                # part2 += l_lambda * loss_g(output_gen, target)
                part2 += loss_g(output_gen, target)
        loss = part1 + part2 
        loss.backward()
        optimizer.step()

        p_optimizer.zero_grad()
        loss_p = loss_fun(output_p, target)
        loss_p.backward()
        p_optimizer.step()

        a_otimizer.zero_grad()
        adapt_logits = p_model.c_adaptor(output_p.detach())
        adapt_loss = loss_g(adapt_logits, target)
        adapt_feature = model.produce_adapt_feature(data)
        adapt_loss += loss_fun(p_model.classifier(adapt_feature), target)
        adapt_loss.backward()
        a_otimizer.step()

        loss_all += loss_p.item()
        total += target.size(0)
        pred = (output_g.detach()+output_p).data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()
    return loss_all / len(data_loader), correct/total


def train_gen_plus0(model, p_model, data_loader, optimizer, p_optimizer, a_otimizer, loss_fun, loss_g, Specific_heads, Specific_adaptors, client_idx, device):
    clinet_num = len(Specific_heads.keys())
    assert clinet_num != 0
    l_lambda = 1/(clinet_num-1)
    model.train()
    p_model.train()
    loss_all = 0
    total = 0
    correct = 0
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
        for idxx in range(clinet_num):
            if idxx != client_idx:
                Spe_classifier = Specific_heads[idxx]
                Specific_adaptor = Specific_adaptors[idxx]
                feature_adapt = Specific_adaptor(feature_g)
                output_adapt = Spe_classifier(feature_adapt)
                # part2 += l_lambda * loss_g(output_adapt, target)
                part2 += loss_g(output_adapt, target)
        loss = part1 + part2
        loss.backward()
        optimizer.step()

        p_optimizer.zero_grad()
        loss_p = loss_fun(output_p, target)
        loss_p.backward()
        p_optimizer.step()

        a_otimizer.zero_grad()
        adapt_feature = p_model.f_adaptor(feature_g.detach())
        adapt_classifier = copy.deepcopy(p_model.classifier)
        adapt_loss = loss_fun(adapt_classifier(adapt_feature), target)
        adapt_loss.backward()
        a_otimizer.step()

        loss_all += loss_p.item()
        total += target.size(0)
        pred = (output_g.detach()+output_p).data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()
    return loss_all / len(data_loader), correct/total


def train_gen_plus1(model, p_model, data_loader, optimizer, p_optimizer, a_otimizer, loss_fun, loss_g, Specific_heads, Specific_adaptors, client_idx, device):
    clinet_num = len(Specific_heads.keys())
    assert clinet_num != 0
    l_lambda = 1/(clinet_num-1)
    model.train()
    p_model.train()
    loss_all = 0
    total = 0
    correct = 0
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
        for idxx in range(clinet_num):
            if idxx != client_idx:
                Spe_classifier = Specific_heads[idxx]
                Specific_adaptor = Specific_adaptors[idxx]
                feature_adapt = Specific_adaptor(feature_g)
                output_adapt = Spe_classifier(feature_adapt)

                output_gen = Spe_classifier(feature_g)
                # part2 += l_lambda * loss_g(output_gen, target)
                part2 += loss_g(output_adapt, target)
                part2 += loss_g(output_gen, target)
        loss = part1 + part2
        loss.backward()
        optimizer.step()

        p_optimizer.zero_grad()
        loss_p = loss_fun(output_p, target)
        loss_p.backward()
        p_optimizer.step()

        a_otimizer.zero_grad()
        adapt_feature = p_model.f_adaptor(feature_g.detach())
        adapt_classifier = copy.deepcopy(p_model.classifier)
        adapt_loss = loss_fun(adapt_classifier(adapt_feature), target)
        adapt_loss.backward()
        a_otimizer.step()

        loss_all += loss_p.item()
        total += target.size(0)
        pred = (output_g.detach()+output_p).data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()
    return loss_all / len(data_loader), correct/total


def train_gen_plus2(model, p_model, copy_model, data_loader, optimizer, p_optimizer, a_otimizer, loss_fun, loss_g, Specific_heads, Specific_adaptors, client_idx, device):
    clinet_num = len(Specific_heads.keys())
    assert clinet_num != 0
    l_lambda = 1/(clinet_num-1)
    model.train()
    p_model.train()
    copy_model.train()
    loss_all = 0
    total = 0
    correct = 0
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
        for idxx in range(clinet_num):
            if idxx != client_idx:
                Spe_classifier = Specific_heads[idxx]
                Specific_up1, Specific_up2, Specific_up3 = Specific_adaptors[idxx]
                model.load_state_dict(Specific_up1, strict=False)
                model.load_state_dict(Specific_up2, strict=False)
                model.load_state_dict(Specific_up3, strict=False)
                feature_adapt = model.produce_adapt_feature(data)
                output_adapt = Spe_classifier(feature_adapt)
                # part2 += l_lambda * loss_g(output_adapt, target)
                part2 += loss_g(output_adapt, target)
        loss = part1 + part2
        loss.backward()
        optimizer.step()

        p_optimizer.zero_grad()
        loss_p = loss_fun(output_p, target)
        loss_p.backward()
        p_optimizer.step()

        for kkk in model.state_dict().keys():
            if 'adap' not in kkk:
                if 'num_batches_tracked' not in kkk:
                    copy_model.state_dict()[kkk].data.copy_(model.state_dict()[kkk])

        a_otimizer.zero_grad()
        adapt_feature = copy_model.produce_adapt_feature(data)
        adapt_classifier = copy.deepcopy(p_model.classifier)
        adapt_loss = loss_fun(adapt_classifier(adapt_feature), target)
        adapt_loss.backward()
        a_otimizer.step()

        loss_all += loss_p.item()
        total += target.size(0)
        pred = (output_g.detach()+output_p).data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    for kkk in copy_model.state_dict().keys():
        if 'adap' in kkk:
            model.state_dict()[kkk].data.copy_(copy_model.state_dict()[kkk])

    return loss_all / len(data_loader), correct/total


def train_gen_plus3(model, p_model, copy_model, data_loader, optimizer, p_optimizer, a_otimizer, loss_fun, loss_g, Specific_heads, Specific_adaptors, client_idx, device):
    clinet_num = len(Specific_heads.keys())
    assert clinet_num != 0
    l_lambda = 1/(clinet_num-1)
    model.train()
    p_model.train()
    copy_model.train()
    loss_all = 0
    total = 0
    correct = 0
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
        for idxx in range(clinet_num):
            if idxx != client_idx:
                Spe_classifier = Specific_heads[idxx]
                Specific_up1, Specific_up2, Specific_up3 = Specific_adaptors[idxx]
                model.load_state_dict(Specific_up1, strict=False)
                model.load_state_dict(Specific_up2, strict=False)
                model.load_state_dict(Specific_up3, strict=False)
                feature_adapt = model.produce_adapt_feature(data)
                output_adapt = Spe_classifier(feature_adapt)

                output_gen = Spe_classifier(feature_g)
                # part2 += l_lambda * loss_g(output_gen, target)
                part2 += loss_g(output_adapt, target)
                part2 += loss_g(output_gen, target)
        loss = part1 + part2
        loss.backward()
        optimizer.step()

        p_optimizer.zero_grad()
        loss_p = loss_fun(output_p, target)
        loss_p.backward()
        p_optimizer.step()

        for kkk in model.state_dict().keys():
            if 'adap' not in kkk:
                if 'num_batches_tracked' not in kkk:
                    copy_model.state_dict()[kkk].data.copy_(model.state_dict()[kkk])

        a_otimizer.zero_grad()
        adapt_feature = copy_model.produce_adapt_feature(data)
        adapt_classifier = copy.deepcopy(p_model.classifier)
        adapt_loss = loss_fun(adapt_classifier(adapt_feature), target)
        adapt_loss.backward()
        a_otimizer.step()

        loss_all += loss_p.item()
        total += target.size(0)
        pred = (output_g.detach()+output_p).data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    for kkk in copy_model.state_dict().keys():
        if 'adap' in kkk:
            model.state_dict()[kkk].data.copy_(copy_model.state_dict()[kkk])

    return loss_all / len(data_loader), correct/total


def train_ada_shabby1(model, p_model, adaptor, data_loader, optimizer, p_optimizer, a_otimizer, loss_fun, loss_g, Specific_heads, client_idx, device):
    clinet_num = len(Specific_heads.keys())
    assert clinet_num != 0
    l_lambda = 1/(clinet_num-1)
    model.train()
    p_model.train()
    loss_all = 0
    total = 0
    correct = 0
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
        for idxx in range(clinet_num):
            if idxx != client_idx:
                Spe_classifier = Specific_heads[idxx]
                output_gen = Spe_classifier(feature_g)
                # part2 += l_lambda * loss_g(output_gen, target)
                part2 += loss_g(output_gen, target)
        loss = part1 + part2 
        loss.backward()
        optimizer.step()

        p_optimizer.zero_grad()
        loss_p = loss_fun(output_p, target)
        loss_p.backward()
        p_optimizer.step()

        a_otimizer.zero_grad()
        adapt_feature = adaptor(feature_g.detach())
        adapt_loss = loss_fun(p_model.classifier(adapt_feature), target)
        adapt_loss.backward()
        a_otimizer.step()

        loss_all += loss_p.item()
        total += target.size(0)
        pred = (output_g.detach()+output_p).data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    return loss_all / len(data_loader), correct/total


def train_ada_shabby2(model, p_model, data_loader, optimizer, p_optimizer, a_otimizer, loss_fun, loss_g, Specific_heads, client_idx, device):
    clinet_num = len(Specific_heads.keys())
    assert clinet_num != 0
    l_lambda = 1/(clinet_num-1)
    model.train()
    p_model.train()
    loss_all = 0
    total = 0
    correct = 0
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
        for idxx in range(clinet_num):
            if idxx != client_idx:
                Spe_classifier = Specific_heads[idxx]
                output_gen = Spe_classifier(feature_g)
                # part2 += l_lambda * loss_g(output_gen, target)
                part2 += loss_g(output_gen, target)
        loss = part1 + part2 
        loss.backward()
        optimizer.step()

        p_optimizer.zero_grad()
        loss_p = loss_fun(output_p, target)
        loss_p.backward()
        p_optimizer.step()

        a_otimizer.zero_grad()
        adapt_feature = p_model.f_adaptor(feature_g.detach())
        adapt_classifier = copy.deepcopy(p_model.classifier)
        adapt_loss = loss_fun(adapt_classifier(adapt_feature), target)
        adapt_loss.backward()
        a_otimizer.step()

        loss_all += loss_p.item()
        total += target.size(0)
        pred = (output_g.detach()+output_p).data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    return loss_all / len(data_loader), correct/total


def train_ada_shabby3(model, p_model, adaptor, data_loader, optimizer, p_optimizer, a_otimizer, loss_fun, loss_g, Specific_heads, client_idx, device):
    clinet_num = len(Specific_heads.keys())
    assert clinet_num != 0
    l_lambda = 1/(clinet_num-1)
    model.train()
    p_model.train()
    loss_all = 0
    total = 0
    correct = 0
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
        for idxx in range(clinet_num):
            if idxx != client_idx:
                Spe_classifier = Specific_heads[idxx]
                output_gen = Spe_classifier(feature_g)
                # part2 += l_lambda * loss_g(output_gen, target)
                part2 += loss_g(output_gen, target)
        loss = part1 + part2 
        loss.backward()
        optimizer.step()

        p_optimizer.zero_grad()
        loss_p = loss_fun(output_p, target)
        loss_p.backward()
        p_optimizer.step()

        a_otimizer.zero_grad()
        adapt_feature = adaptor(feature_g.detach())
        adapt_classifier = copy.deepcopy(p_model.classifier)
        adapt_loss = loss_fun(adapt_classifier(adapt_feature), target)
        adapt_loss.backward()
        a_otimizer.step()

        loss_all += loss_p.item()
        total += target.size(0)
        pred = (output_g.detach()+output_p).data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    return loss_all / len(data_loader), correct/total


def train_ada_shabby_kl(model, p_model, data_loader, optimizer, p_optimizer, a_otimizer, loss_fun, loss_g, Specific_heads, client_idx, device):
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    clinet_num = len(Specific_heads.keys())
    assert clinet_num != 0
    l_lambda = 1/(clinet_num-1)
    model.train()
    p_model.train()
    loss_all = 0
    total = 0
    correct = 0
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
        for idxx in range(clinet_num):
            if idxx != client_idx:
                Spe_classifier = Specific_heads[idxx]
                output_gen = Spe_classifier(feature_g)
                # part2 += l_lambda * loss_g(output_gen, target)
                part2 += loss_g(output_gen, target)
        loss = part1 + part2 
        loss.backward()
        optimizer.step()

        p_optimizer.zero_grad()
        loss_p = loss_fun(output_p, target)
        loss_p.backward()
        p_optimizer.step()

        a_otimizer.zero_grad()
        adapt_feature = p_model.f_adaptor(feature_g.detach())
        adapt_classifier = copy.deepcopy(p_model.classifier)
        adapt_loss = kl_loss(F.log_softmax(adapt_classifier(adapt_feature), dim=1), F.softmax(output_p.detach(), dim=1))
        adapt_loss.backward()
        a_otimizer.step()

        loss_all += loss_p.item()
        total += target.size(0)
        pred = (output_g.detach()+output_p).data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    return loss_all / len(data_loader), correct/total


def train_ada_shabby_klcr(model, p_model, data_loader, optimizer, p_optimizer, a_otimizer, loss_fun, loss_g, Specific_heads, client_idx, device):
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    clinet_num = len(Specific_heads.keys())
    assert clinet_num != 0
    l_lambda = 1/(clinet_num-1)
    model.train()
    p_model.train()
    loss_all = 0
    total = 0
    correct = 0
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
        for idxx in range(clinet_num):
            if idxx != client_idx:
                Spe_classifier = Specific_heads[idxx]
                output_gen = Spe_classifier(feature_g)
                # part2 += l_lambda * loss_g(output_gen, target)
                part2 += loss_g(output_gen, target)
        loss = part1 + part2 
        loss.backward()
        optimizer.step()

        p_optimizer.zero_grad()
        loss_p = loss_fun(output_p, target)
        loss_p.backward()
        p_optimizer.step()

        a_otimizer.zero_grad()
        adapt_feature = p_model.f_adaptor(feature_g.detach())
        adapt_classifier = copy.deepcopy(p_model.classifier)
        adapt_loss = kl_loss(F.log_softmax(adapt_classifier(adapt_feature), dim=1), F.softmax(output_p.detach(), dim=1))
        adapt_loss += loss_fun(adapt_classifier(adapt_feature), target)
        adapt_loss.backward()
        a_otimizer.step()

        loss_all += loss_p.item()
        total += target.size(0)
        pred = (output_g.detach()+output_p).data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    return loss_all / len(data_loader), correct/total


def train_ada_residual_kl(model, p_model, copy_model, data_loader, optimizer, p_optimizer, a_otimizer, loss_fun, loss_g, Specific_heads, client_idx, device):
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    clinet_num = len(Specific_heads.keys())
    assert clinet_num != 0
    l_lambda = 1/(clinet_num-1)
    model.train()
    p_model.train()
    copy_model.train()
    loss_all = 0
    total = 0
    correct = 0
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
        for idxx in range(clinet_num):
            if idxx != client_idx:
                Spe_classifier = Specific_heads[idxx]
                output_gen = Spe_classifier(feature_g)
                # part2 += l_lambda * loss_g(output_gen, target)
                part2 += loss_g(output_gen, target)
        loss = part1 + part2 
        loss.backward()
        optimizer.step()

        p_optimizer.zero_grad()
        loss_p = loss_fun(output_p, target)
        loss_p.backward()
        p_optimizer.step()

        for kkk in model.state_dict().keys():
            if 'adap' not in kkk:
                if 'num_batches_tracked' not in kkk:
                    copy_model.state_dict()[kkk].data.copy_(model.state_dict()[kkk])

        a_otimizer.zero_grad()
        adapt_feature = copy_model.produce_adapt_feature(data)
        adapt_classifier = copy.deepcopy(p_model.classifier)
        adapt_loss = kl_loss(F.log_softmax(adapt_classifier(adapt_feature), dim=1), F.softmax(output_p.detach(), dim=1))
        adapt_loss.backward()
        a_otimizer.step()

        loss_all += loss_p.item()
        total += target.size(0)
        pred = (output_g.detach()+output_p).data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    for kkk in copy_model.state_dict().keys():
        if 'adap' in kkk:
            model.state_dict()[kkk].data.copy_(copy_model.state_dict()[kkk])

    return loss_all / len(data_loader), correct/total


def train_ada_residual_klcr(model, p_model, copy_model, data_loader, optimizer, p_optimizer, a_otimizer, loss_fun, loss_g, Specific_heads, client_idx, device):
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    clinet_num = len(Specific_heads.keys())
    assert clinet_num != 0
    l_lambda = 1/(clinet_num-1)
    model.train()
    p_model.train()
    copy_model.train()
    loss_all = 0
    total = 0
    correct = 0
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
        for idxx in range(clinet_num):
            if idxx != client_idx:
                Spe_classifier = Specific_heads[idxx]
                output_gen = Spe_classifier(feature_g)
                # part2 += l_lambda * loss_g(output_gen, target)
                part2 += loss_g(output_gen, target)
        loss = part1 + part2 
        loss.backward()
        optimizer.step()

        p_optimizer.zero_grad()
        loss_p = loss_fun(output_p, target)
        loss_p.backward()
        p_optimizer.step()

        for kkk in model.state_dict().keys():
            if 'adap' not in kkk:
                if 'num_batches_tracked' not in kkk:
                    copy_model.state_dict()[kkk].data.copy_(model.state_dict()[kkk])

        a_otimizer.zero_grad()
        adapt_feature = copy_model.produce_adapt_feature(data)
        adapt_classifier = copy.deepcopy(p_model.classifier)
        adapt_loss = kl_loss(F.log_softmax(adapt_classifier(adapt_feature), dim=1), F.softmax(output_p.detach(), dim=1))
        adapt_loss += loss_fun(adapt_classifier(adapt_feature), target)
        adapt_loss.backward()
        a_otimizer.step()

        loss_all += loss_p.item()
        total += target.size(0)
        pred = (output_g.detach()+output_p).data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    for kkk in copy_model.state_dict().keys():
        if 'adap' in kkk:
            model.state_dict()[kkk].data.copy_(copy_model.state_dict()[kkk])

    return loss_all / len(data_loader), correct/total

