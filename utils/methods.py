import sys, os
from utils.utils import euclidean_dist
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict, defaultdict
from utils.func import *


def local_training(models, personalized_models, paggregation_models, hnet, server_model, global_prototypes, Extra_modules, valid_onehot, his_weight, args, train_loaders, test_loaders, optimizers, loss_fun, device, a_iter=0):
    clinet_num = len(models)
    proto_dict = {}
    client_weight = {x:[] for x in range(clinet_num)}
    average_proto = 0
    # if a_iter == 2:
    #     assert False
    if args.mode == 'fedtp':
        hnet.train()
        h_optimizer = optim.SGD(params=hnet.parameters(), lr=args.lr)
        arr = np.arange(clinet_num)
        weights = hnet(torch.tensor([arr], dtype=torch.long).to(device),False)
        for client_idx, model in enumerate(models):
            private_params = weights[client_idx]
            model.load_state_dict(private_params, strict=False)
            train_loss, train_acc = train(model, train_loaders[client_idx], optimizers[client_idx], loss_fun, device)

        for client_idx in range(clinet_num):
            final_state = models[client_idx].state_dict()
            node_weights = weights[client_idx]
            inner_state = OrderedDict({k: tensor.data for k, tensor in node_weights.items()})
            delta_theta = OrderedDict({k: inner_state[k] - final_state[k] for k in node_weights.keys()})
            hnet_grads = torch.autograd.grad(
                list(node_weights.values()), hnet.parameters(), grad_outputs=list(delta_theta.values()), retain_graph=True
            )

            if client_idx == 0:
                grads_update = [1/clinet_num  for x in hnet_grads]
            else:
                for g in range(len(hnet_grads)):
                    grads_update[g] += 1/clinet_num * hnet_grads[g]
        h_optimizer.zero_grad()
        for p, g in zip(hnet.parameters(), grads_update):
            p.grad = g
        h_optimizer.step()
        return train_loss, train_acc

    elif args.mode == 'peer':
        Specific_head = {}
        Specific_adaptor = {}
        for client_idx in range(clinet_num):
            Specific_head[client_idx] = copy.deepcopy(personalized_models[client_idx].classifier)
        if args.version in [7, 8, 19, 20]:
            for client_idx in range(clinet_num):
                Specific_adaptor[client_idx] = copy.deepcopy(personalized_models[client_idx].f_adaptor)
        elif args.version in [9, 10, 21, 22]:
            for client_idx in range(clinet_num):
                Specific_adaptor[client_idx] = [copy.deepcopy(models[client_idx].adap3.state_dict()), copy.deepcopy(models[client_idx].adap4.state_dict()), copy.deepcopy(models[client_idx].adap5.state_dict())]

    for client_idx, model in enumerate(models):
        if args.mode.lower() == 'fedprox':
            # skip the first server model(random initialized)
            if a_iter > 0:
                train_loss, train_acc = train_prox(args, model, server_model, train_loaders[client_idx], optimizers[client_idx], loss_fun, device)
            else:
                train_loss, train_acc = train(model, train_loaders[client_idx], optimizers[client_idx], loss_fun, device)
        
        elif args.mode == 'peer':
            p_optimizer = optim.SGD(params=personalized_models[client_idx].parameters(), lr=args.lr)
            criterion_ba = nn.CrossEntropyLoss().to(device)
    
            if args.version == 17:
                train_loss, train_acc = train_v0(model, personalized_models[client_idx], train_loaders[client_idx], optimizers[client_idx], p_optimizer, loss_fun, criterion_ba, device)

            elif args.version in [18, 27]:
                train_loss, train_acc = train_gen0(model, personalized_models[client_idx], train_loaders[client_idx], optimizers[client_idx], 
                                                  p_optimizer, loss_fun, criterion_ba, Specific_head, client_idx, device)

            elif args.version == 25:
                train_loss, train_acc = train_gen_full(model, personalized_models[client_idx], train_loaders[client_idx], optimizers[client_idx], 
                                                  p_optimizer, loss_fun, criterion_ba, Specific_head, client_idx, device)

            elif args.version == 28:
                train_loss, train_acc = train_gen_full(model, personalized_models[client_idx], train_loaders[client_idx], optimizers[client_idx], 
                                          p_optimizer, loss_fun, criterion_ba, Extra_modules, client_idx, device)

            elif args.version == 50:
                train_loss, train_acc = train_tt_kll1(model, personalized_models[client_idx], train_loaders[client_idx], test_loaders[client_idx], 
                                        optimizers[client_idx], p_optimizer, loss_fun, criterion_ba, Specific_head, valid_onehot[client_idx], client_idx, device, a_iter)

            elif args.version in [5, 15]:
                a_otimizer = optim.SGD(params=personalized_models[client_idx].f_adaptor.parameters(), lr=args.lr, momentum=args.momentum)
                p_optimizer = optim.SGD(params=[{'params':personalized_models[client_idx].features.parameters()},
                                     {'params':personalized_models[client_idx].classifier.parameters()}], lr=args.lr)
                train_loss, train_acc = train_ada_shabby(model, personalized_models[client_idx], train_loaders[client_idx], optimizers[client_idx], 
                                                  p_optimizer, a_otimizer, loss_fun, criterion_ba, Specific_head, client_idx, device)

            elif args.version in [6, 16]:
                copy_model = copy.deepcopy(model)
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
                a_otimizer = optim.SGD(params=[{'params':copy_model.adap3.parameters()},
                                               {'params':copy_model.adap4.parameters()},
                                               {'params':copy_model.adap5.parameters()}], lr=args.lr, momentum=args.momentum)
                train_loss, train_acc = train_ada_residual(model, personalized_models[client_idx], copy_model, train_loaders[client_idx], optimizer, 
                                                  p_optimizer, a_otimizer, loss_fun, criterion_ba, Specific_head, client_idx, device)

            elif args.version in [7, 19]:
                a_otimizer = optim.SGD(params=personalized_models[client_idx].f_adaptor.parameters(), lr=args.lr, momentum=args.momentum)
                p_optimizer = optim.SGD(params=[{'params':personalized_models[client_idx].features.parameters()},
                                     {'params':personalized_models[client_idx].classifier.parameters()}], lr=args.lr)
                train_loss, train_acc = train_gen_plus0(model, personalized_models[client_idx], train_loaders[client_idx], optimizers[client_idx], 
                                                  p_optimizer, a_otimizer, loss_fun, criterion_ba, Specific_head, Specific_adaptor, client_idx, device)

            elif args.version in [8, 20]:
                a_otimizer = optim.SGD(params=personalized_models[client_idx].f_adaptor.parameters(), lr=args.lr, momentum=args.momentum)
                p_optimizer = optim.SGD(params=[{'params':personalized_models[client_idx].features.parameters()},
                                     {'params':personalized_models[client_idx].classifier.parameters()}], lr=args.lr)
                train_loss, train_acc = train_gen_plus1(model, personalized_models[client_idx], train_loaders[client_idx], optimizers[client_idx], 
                                                  p_optimizer, a_otimizer, loss_fun, criterion_ba, Specific_head, Specific_adaptor, client_idx, device)

            elif args.version in [9, 21]:
                copy_model = copy.deepcopy(model)
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
                a_otimizer = optim.SGD(params=[{'params':copy_model.adap3.parameters()},
                                               {'params':copy_model.adap4.parameters()},
                                               {'params':copy_model.adap5.parameters()}], lr=args.lr, momentum=args.momentum)
                train_loss, train_acc = train_gen_plus2(model, personalized_models[client_idx], copy_model, train_loaders[client_idx], optimizer, 
                                                  p_optimizer, a_otimizer, loss_fun, criterion_ba, Specific_head, Specific_adaptor, client_idx, device)

            elif args.version in [10, 22]:
                copy_model = copy.deepcopy(model)
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
                a_otimizer = optim.SGD(params=[{'params':copy_model.adap3.parameters()},
                                               {'params':copy_model.adap4.parameters()},
                                               {'params':copy_model.adap5.parameters()}], lr=args.lr, momentum=args.momentum)
                train_loss, train_acc = train_gen_plus3(model, personalized_models[client_idx], copy_model, train_loaders[client_idx], optimizer, 
                                                  p_optimizer, a_otimizer, loss_fun, criterion_ba, Specific_head, Specific_adaptor, client_idx, device)                

            elif args.version == 11:
                a_otimizer = optim.SGD(params=personalized_models[client_idx].f_adaptor.parameters(), lr=args.lr, momentum=args.momentum)
                p_optimizer = optim.SGD(params=[{'params':personalized_models[client_idx].features.parameters()},
                                     {'params':personalized_models[client_idx].classifier.parameters()}], lr=args.lr)
                train_loss, train_acc = train_ada_shabby_kl(model, personalized_models[client_idx], train_loaders[client_idx], optimizers[client_idx], 
                                                  p_optimizer, a_otimizer, loss_fun, criterion_ba, Specific_head, client_idx, device)

            elif args.version == 12:
                a_otimizer = optim.SGD(params=personalized_models[client_idx].f_adaptor.parameters(), lr=args.lr, momentum=args.momentum)
                p_optimizer = optim.SGD(params=[{'params':personalized_models[client_idx].features.parameters()},
                                     {'params':personalized_models[client_idx].classifier.parameters()}], lr=args.lr)
                train_loss, train_acc = train_ada_shabby_klcr(model, personalized_models[client_idx], train_loaders[client_idx], optimizers[client_idx], 
                                                  p_optimizer, a_otimizer, loss_fun, criterion_ba, Specific_head, client_idx, device)

            elif args.version == 13:
                copy_model = copy.deepcopy(model)
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
                a_otimizer = optim.SGD(params=[{'params':copy_model.adap3.parameters()},
                                               {'params':copy_model.adap4.parameters()},
                                               {'params':copy_model.adap5.parameters()}], lr=args.lr, momentum=args.momentum)
                train_loss, train_acc = train_ada_residual_kl(model, personalized_models[client_idx], copy_model, train_loaders[client_idx], optimizer, 
                                                  p_optimizer, a_otimizer, loss_fun, criterion_ba, Specific_head, client_idx, device)

            elif args.version == 14:
                copy_model = copy.deepcopy(model)
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
                a_otimizer = optim.SGD(params=[{'params':copy_model.adap3.parameters()},
                                               {'params':copy_model.adap4.parameters()},
                                               {'params':copy_model.adap5.parameters()}], lr=args.lr, momentum=args.momentum)
                train_loss, train_acc = train_ada_residual_klcr(model, personalized_models[client_idx], copy_model, train_loaders[client_idx], optimizer, 
                                                  p_optimizer, a_otimizer, loss_fun, criterion_ba, Specific_head, client_idx, device)

            elif args.version == 24:
                a_otimizer = optim.SGD(params=personalized_models[client_idx].f_adaptor.parameters(), lr=args.lr, momentum=args.momentum)
                p_optimizer = optim.SGD(params=[{'params':personalized_models[client_idx].features.parameters()},
                                     {'params':personalized_models[client_idx].classifier.parameters()}], lr=args.lr)
                train_loss, train_acc = train_ada_shabby2(model, personalized_models[client_idx], train_loaders[client_idx], optimizers[client_idx], 
                                                  p_optimizer, a_otimizer, loss_fun, criterion_ba, Specific_head, client_idx, device)

            # elif args.version == 29:


            else:
                train_loss, train_acc = others_train(args.version, model, personalized_models[client_idx], train_loaders[client_idx], test_loaders[client_idx], 
                                        optimizers[client_idx], p_optimizer, loss_fun, criterion_ba, Specific_head, Specific_adaptor, valid_onehot[client_idx], 
                                        client_idx, device, args, a_iter)

        elif args.mode == 'fedper':
            private_params = personalized_models[client_idx].state_dict()
            model.load_state_dict(private_params, strict=False)
            train_loss, train_acc = train(model, train_loaders[client_idx], optimizers[client_idx], loss_fun, device)

        elif args.mode == 'fedrod':
            p_optimizer = optim.SGD(params=personalized_models[client_idx].parameters(), lr=args.lr)
            criterion_ba = nn.CrossEntropyLoss().to(device)
            train_loss, train_acc = train_rod(model, personalized_models[client_idx], train_loaders[client_idx], optimizers[client_idx], p_optimizer, loss_fun, criterion_ba, device)

        else:
            train_loss, train_acc = train(model, train_loaders[client_idx], optimizers[client_idx], loss_fun, device)

    return train_loss, train_acc, proto_dict, client_weight


def train_v0(model, p_model, data_loader, optimizer, p_optimizer, loss_fun, loss_g, device):
    model.train()
    p_model.train()
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        optimizer.zero_grad()

        data = data.to(device)
        target = target.to(device)
        output_g = model(data)
        loss = loss_g(output_g, target)
        loss.backward()
        optimizer.step()

        output_p = p_model(data)
        loss_p = loss_fun(output_p, target)
        p_optimizer.zero_grad()
        loss_p.backward()
        p_optimizer.step()

        loss_all += loss_p.item()
        total += target.size(0)
        pred = (output_g.detach()+output_p).data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    return loss_all / len(data_loader), correct/total


def train_gen0(model, p_model, data_loader, optimizer, p_optimizer, loss_fun, loss_g, Specific_heads, client_idx, device):
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
    return loss_all / len(data_loader), correct/total


def train_gen_full(model, p_model, data_loader, optimizer, p_optimizer, loss_fun, loss_g, Specific_heads, client_idx, device):
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
        part3 = 0
        for idxx in range(clinet_num):
            if idxx != client_idx:
                Spe_classifier = Specific_heads[idxx]
                output_gen = Spe_classifier(feature_g)
                output_per = Spe_classifier(feature_p)
                # part2 += l_lambda * loss_g(output_gen, target)
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


def train_tt_kll1(model, p_model, data_loader, unlabeled_loader, optimizer, p_optimizer, loss_fun, loss_g, Specific_heads, onehot_value, client_idx, device, a_iter):
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
            output_g = model(data)
            output_p = p_model(data)

            if onehot_value == 0:
                optimizer.zero_grad()
                loss = kl_loss(F.log_softmax(output_g, dim=1), F.softmax(output_p.detach(), dim=1))
                loss.backward()
                optimizer.step()

            elif onehot_value == 1:
                p_optimizer.zero_grad()
                loss_p = kl_loss(F.log_softmax(output_p, dim=1), F.softmax(output_g.detach(), dim=1))
                loss_p.backward()
                p_optimizer.step()

    return loss_all / len(data_loader), correct/total


def train_ada_shabby(model, p_model, data_loader, optimizer, p_optimizer, a_otimizer, loss_fun, loss_g, Specific_heads, client_idx, device):
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

        # if client_idx == 3:
        #     print("After adaptor optimization.")
        #     for name, params in p_model.classifier.named_parameters():
        #         # print("name: ", name)
        #         if name == 'fc3.weight':
        #             print("grads:", params.grad)
        # break

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


def train_ada_residual(model, p_model, copy_model, data_loader, optimizer, p_optimizer, a_otimizer, loss_fun, loss_g, Specific_heads, client_idx, device):
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

