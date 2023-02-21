import sys, os
from utils.utils import euclidean_dist
import numpy as np
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict, defaultdict
from utils.func import *


def local_training(models, personalized_models, paggregation_models, hnet, server_model, global_prototypes, Extra_modules, valid_onehot, his_weight, args, train_loaders, test_loaders, optimizers, loss_fun, device, a_iter=0, phase='Train'):
    client_num = len(models)
    proto_dict = {}
    client_weight = {x:[] for x in range(client_num)}
    average_proto = 0

    if args.mode == 'fedtp':
        hnet.train()
        h_optimizer = optim.SGD(params=hnet.parameters(), lr=args.lr)
        arr = np.arange(client_num)
        weights = hnet(torch.tensor([arr], dtype=torch.long).to(device),False)
        for client_idx, model in enumerate(models):
            private_params = weights[client_idx]
            model.load_state_dict(private_params, strict=False)
            train_loss, train_acc = train(model, train_loaders[client_idx], optimizers[client_idx], loss_fun, device)

        for client_idx in range(client_num):
            final_state = models[client_idx].state_dict()
            node_weights = weights[client_idx]
            inner_state = OrderedDict({k: tensor.data for k, tensor in node_weights.items()})
            delta_theta = OrderedDict({k: inner_state[k] - final_state[k] for k in node_weights.keys()})
            hnet_grads = torch.autograd.grad(
                list(node_weights.values()), hnet.parameters(), grad_outputs=list(delta_theta.values()), retain_graph=True
            )

            if client_idx == 0:
                grads_update = [1/client_num  for x in hnet_grads]
            else:
                for g in range(len(hnet_grads)):
                    grads_update[g] += 1/client_num * hnet_grads[g]
        h_optimizer.zero_grad()
        for p, g in zip(hnet.parameters(), grads_update):
            p.grad = g
        h_optimizer.step()
        return train_loss, train_acc

    elif args.mode == 'peer':
        Specific_head = {}
        Specific_adaptor = {}
        for client_idx in range(client_num):
            if args.version in [51, 52, 53, 54, 67, 66, 69]:
                Specific_head[client_idx] = copy.deepcopy(personalized_models[client_idx].head)
            elif args.version == 56:
                Specific_head[client_idx] = copy.deepcopy(paggregation_models[client_idx].head)
            elif args.version == 57:
                Specific_head[client_idx] = [copy.deepcopy(paggregation_models[client_idx].head), copy.deepcopy(personalized_models[client_idx].head)]
            else:
                Specific_head[client_idx] = copy.deepcopy(personalized_models[client_idx].classifier)
        if args.version in [7, 8, 19, 20, 40, 41, 42]:
            for client_idx in range(client_num):
                Specific_adaptor[client_idx] = copy.deepcopy(personalized_models[client_idx].f_adaptor)
        elif args.version in [9, 10, 21, 22]:
            for client_idx in range(client_num):
                Specific_adaptor[client_idx] = [copy.deepcopy(models[client_idx].adap3.state_dict()), copy.deepcopy(models[client_idx].adap4.state_dict()), copy.deepcopy(models[client_idx].adap5.state_dict())]
        elif args.version in [65, 66, 68, 69]:
            for client_idx in range(client_num):
                Specific_adaptor[client_idx] = {}
                for kky in personalized_models[client_idx].state_dict().keys():
                    if 'bn' in kky:
                        if 'num_batches_tracked' not in kky:
                            Specific_adaptor[client_idx][kky] = copy.deepcopy(personalized_models[client_idx].state_dict()[kky])

    elif args.mode == 'COPA':
        Specific_head = {}
        for client_idx in range(client_num):
            Specific_head[client_idx] = copy.deepcopy(models[client_idx].head)

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
    
            if args.version in [17, 61, 64]:
                train_loss, train_acc = train_v0(model, personalized_models[client_idx], train_loaders[client_idx], optimizers[client_idx], p_optimizer, loss_fun, criterion_ba, device)

            elif args.version in [18, 27, 58, 59, 60, 63]:
                train_loss, train_acc = train_gen0(model, personalized_models[client_idx], train_loaders[client_idx], optimizers[client_idx], 
                                                  p_optimizer, loss_fun, criterion_ba, Specific_head, client_idx, device)

            elif args.version == 25:
                train_loss, train_acc = train_gen_full(model, personalized_models[client_idx], train_loaders[client_idx], optimizers[client_idx], 
                                                  p_optimizer, loss_fun, criterion_ba, Specific_head, client_idx, device)

            elif args.version == 28:
                train_loss, train_acc = train_gen_full(model, personalized_models[client_idx], train_loaders[client_idx], optimizers[client_idx], 
                                          p_optimizer, loss_fun, criterion_ba, Extra_modules, client_idx, device)

            elif args.version in [52, 67]:
                train_loss, train_acc = train_gen0_head(model, personalized_models[client_idx], train_loaders[client_idx], optimizers[client_idx], 
                                                  p_optimizer, loss_fun, criterion_ba, Specific_head, client_idx, device)

            elif args.version in [65, 68]:
                train_loss, train_acc = train_bn_adap_head(model, personalized_models[client_idx], train_loaders[client_idx], optimizers[client_idx], 
                                                  p_optimizer, loss_fun, criterion_ba, Specific_adaptor, client_idx, device)

            elif args.version in [66, 69]:
                train_loss, train_acc = train_bn_adap_gen_head(model, personalized_models[client_idx], train_loaders[client_idx], optimizers[client_idx], 
                                                  p_optimizer, loss_fun, criterion_ba, Specific_head, Specific_adaptor, client_idx, device)

            else:
                train_loss, train_acc = others_train(args.version, model, personalized_models[client_idx], Extra_modules, paggregation_models, train_loaders[client_idx], test_loaders[client_idx], 
                                        optimizers[client_idx], p_optimizer, loss_fun, criterion_ba, Specific_head, Specific_adaptor, valid_onehot[client_idx], 
                                        client_idx, device, args, a_iter, phase)

        elif args.mode == 'fedper':
            private_params = personalized_models[client_idx].state_dict()
            model.load_state_dict(private_params, strict=False)
            train_loss, train_acc = train(model, train_loaders[client_idx], optimizers[client_idx], loss_fun, device)

        elif args.mode == 'fedrod':
            p_optimizer = optim.SGD(params=personalized_models[client_idx].parameters(), lr=args.lr)
            criterion_ba = nn.CrossEntropyLoss().to(device)
            train_loss, train_acc = train_rod(model, personalized_models[client_idx], train_loaders[client_idx], optimizers[client_idx], p_optimizer, loss_fun, criterion_ba, device)

        elif args.mode == 'AlignFed':
            if args.version == 1:
                train_loss, train_acc, proto = AlignFed_train(model, train_loaders[client_idx], optimizers[client_idx], global_prototypes, loss_fun, a_iter, device)
                proto_dict[client_idx] = proto
            elif args.version == 2:
                train_loss, train_acc, proto = AlignFed_train1(model, train_loaders[client_idx], optimizers[client_idx], global_prototypes, loss_fun, a_iter, device)
                proto_dict[client_idx] = proto

        elif args.mode == 'COPA':
            train_loss, train_acc = train_COPA(model, Specific_head, train_loaders[client_idx], optimizers[client_idx], loss_fun, client_idx, device)

        else:
            train_loss, train_acc = train(model, train_loaders[client_idx], optimizers[client_idx], loss_fun, device)

    if args.mode == 'peer':
        if args.version in [59, 60, 61]:
            with torch.no_grad():
                for client_idx, model in enumerate(models):
                    client_weight[client_idx] = weight_calculate(models, test_loaders[client_idx], loss_fun, client_num, device)

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
    client_num = len(Specific_heads.keys())
    assert client_num != 0
    l_lambda = 1/(client_num-1)
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
        for idxx in range(client_num):
            if idxx != client_idx:
                Spe_classifier = Specific_heads[idxx]
                Spe_classifier.eval()
                output_gen = Spe_classifier(feature_g)
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


def train_gen0_head(model, p_model, data_loader, optimizer, p_optimizer, loss_fun, loss_g, Specific_heads, client_idx, device):
    client_num = len(Specific_heads.keys())
    assert client_num != 0
    l_lambda = 1/(client_num-1)
    model.train()
    p_model.train()
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)
        feature_g = model.produce_feature(data)
        output_g = model.head(feature_g)
        feature_p = p_model.produce_feature(data)
        output_p = p_model.head(feature_p)

        optimizer.zero_grad()
        part1 = loss_g(output_g, target)

        part2 = 0
        for idxx in range(client_num):
            if idxx != client_idx:
                Spe_classifier = Specific_heads[idxx]
                Spe_classifier.eval()
                output_gen = Spe_classifier(feature_g)
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


def train_bn_adap_head(model, p_model, data_loader, optimizer, p_optimizer, loss_fun, loss_g, Specific_adaptors, client_idx, device):
    client_num = len(Specific_adaptors.keys())
    assert client_num != 0
    l_lambda = 1/(client_num-1)
    back_model = copy.deepcopy(model)
    back_model.eval()
    model.train()
    p_model.train()
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)
        feature_g = model.produce_feature(data)
        output_g = model.head(feature_g)
        feature_p = p_model.produce_feature(data)
        output_p = p_model.head(feature_p)

        optimizer.zero_grad()
        part1 = loss_g(output_g, target)

        part2 = 0
        for idxx in range(client_num):
            if idxx != client_idx:
                BN_list = Specific_adaptors[idxx]
                for kky in back_model.state_dict().keys():
                    if 'bn' in kky:
                        if 'num_batches_tracked' not in kky:
                            back_model.state_dict()[kky].data.copy_(BN_list[kky])
                # print("conv weight: ", back_model.state_dict()["features.conv2.weight"][0])
                # if idxx == 1:
                #     print("bn2 weight: ", back_model.state_dict()["features.bn2.weight"][0])
                #     print("bn2 running_mean: ", back_model.state_dict()["features.bn2.running_mean"][0])

                feature_gen = back_model.produce_feature(data)
                output_gen = model.head(feature_gen.detach())
                part2 += loss_g(output_gen, target)

        loss = part1 + part2
        loss.backward()
        optimizer.step()

        p_optimizer.zero_grad()
        loss_p = loss_fun(output_p, target)
        loss_p.backward()
        p_optimizer.step()

        for key in back_model.state_dict().keys():
            if 'num_batches_tracked' in key:
                continue
            else:
                back_model.state_dict()[key].data.copy_(model.state_dict()[key])

        loss_all += loss_p.item()
        total += target.size(0)
        pred = (output_g.detach()+output_p).data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    return loss_all / len(data_loader), correct/total


def train_bn_adap_gen_head(model, p_model, data_loader, optimizer, p_optimizer, loss_fun, loss_g, Specific_heads, Specific_adaptors, client_idx, device):
    client_num = len(Specific_adaptors.keys())
    assert client_num != 0
    l_lambda = 1/(client_num-1)
    back_model = copy.deepcopy(model)
    back_model.eval()
    model.train()
    p_model.train()
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)
        feature_g = model.produce_feature(data)
        output_g = model.head(feature_g)
        feature_p = p_model.produce_feature(data)
        output_p = p_model.head(feature_p)

        optimizer.zero_grad()
        part1 = loss_g(output_g, target)

        part2, part3 = 0, 0
        for idxx in range(client_num):
            if idxx != client_idx:
                BN_list = Specific_adaptors[idxx]
                Spe_classifier = Specific_heads[idxx]
                Spe_classifier.eval()
                for kky in back_model.state_dict().keys():
                    if 'bn' in kky:
                        if 'num_batches_tracked' not in kky:
                            back_model.state_dict()[kky].data.copy_(BN_list[kky])

                feature_gen = back_model.produce_feature(data)
                output_gen = model.head(feature_gen.detach())
                output_ggg = Spe_classifier(feature_g)
                part2 += loss_g(output_ggg, target)
                part3 += loss_g(output_gen, target)

        loss = part1 + part2 + part3
        loss.backward()
        optimizer.step()

        p_optimizer.zero_grad()
        loss_p = loss_fun(output_p, target)
        loss_p.backward()
        p_optimizer.step()

        for key in back_model.state_dict().keys():
            if 'num_batches_tracked' in key:
                continue
            else:
                back_model.state_dict()[key].data.copy_(model.state_dict()[key])

        loss_all += loss_p.item()
        total += target.size(0)
        pred = (output_g.detach()+output_p).data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    return loss_all / len(data_loader), correct/total


def train_gen_full(model, p_model, data_loader, optimizer, p_optimizer, loss_fun, loss_g, Specific_heads, client_idx, device):
    client_num = len(Specific_heads.keys())
    assert client_num != 0
    l_lambda = 1/(client_num-1)
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
        for idxx in range(client_num):
            if idxx != client_idx:
                Spe_classifier = Specific_heads[idxx]
                Spe_classifier.eval()
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


def AlignFed_train(model, data_loader, optimizer, global_prototypes, loss_fun, a_iter, device):
    model.train()
    if a_iter!=0:
        client_num = len(list(global_prototypes.keys()))
        assert client_num != 0
    loss_all = 0
    total = 0
    correct = 0
    proto = {}

    for data, target in data_loader:
        optimizer.zero_grad()

        data = data.to(device)
        target = target.to(device)
        local_feature = model.produce_feature(data)
        output = model.head(local_feature)
        loss = loss_fun(output, target)

        if a_iter != 0:
            align_loss = 0
            for dix in range(data.shape[0]):
                lab = int(target[dix])
                pc_proto = global_prototypes[lab]
                pos = torch.cosine_similarity(local_feature[dix].view(1, -1), pc_proto)
                neg = 0
                for cix in range(client_num):
                    neg += torch.cosine_similarity(local_feature[dix].view(1, -1), global_prototypes[cix])
                ins_loss = -torch.log(pos/neg)
                align_loss += ins_loss
            loss += align_loss[0]

        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        loss.backward()
        optimizer.step()

    class_id = {idd:[torch.zeros(1,4096).to(device), 0] for idd in range(10)}
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)
        feature = model.produce_feature(data)
        for ins in range(len(target)):
            c_id = int(target[ins])
            class_id[c_id][0] += feature[ins].view(1,-1).detach()
            class_id[c_id][1] += 1 

    class_prototype = 0
    for cc in range(len(class_id.keys())):
        class_prototype = class_id[cc][0]/class_id[cc][1]
        proto[cc] = class_prototype

    return loss_all / len(data_loader), correct/total, proto


def AlignFed_train1(model, data_loader, optimizer, global_prototypes, loss_fun, a_iter, device):
    model.train()
    if a_iter!=0:
        client_num = len(list(global_prototypes.keys()))
        assert client_num != 0
    loss_all = 0
    total = 0
    correct = 0
    proto = {}

    for data, target in data_loader:
        optimizer.zero_grad()

        data = data.to(device)
        target = target.to(device)
        local_feature = model.produce_feature(data)
        output = model.head(local_feature)
        loss = loss_fun(output, target)

        if a_iter >= 100:
            align_loss = 0
            for dix in range(data.shape[0]):
                lab = int(target[dix])
                pc_proto = global_prototypes[lab]
                pos = torch.cosine_similarity(local_feature[dix].view(1, -1), pc_proto)
                neg = 0
                for cix in range(client_num):
                    neg += torch.cosine_similarity(local_feature[dix].view(1, -1), global_prototypes[cix])
                ins_loss = -torch.log(pos/neg)
                align_loss += ins_loss
            loss += align_loss[0]

        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        loss.backward()
        optimizer.step()

    class_id = {idd:[torch.zeros(1,4096).to(device), 0] for idd in range(10)}
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)
        feature = model.produce_feature(data)
        for ins in range(len(target)):
            c_id = int(target[ins])
            class_id[c_id][0] += feature[ins].view(1,-1).detach()
            class_id[c_id][1] += 1 

    class_prototype = 0
    for cc in range(len(class_id.keys())):
        class_prototype = class_id[cc][0]/class_id[cc][1]
        proto[cc] = class_prototype

    return loss_all / len(data_loader), correct/total, proto


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


def weight_calculate(models, test_loader, loss_fun, client_num, device):
    weight_list = []

    for ccix in range(client_num):
        ttt_model = copy.deepcopy(models[ccix])
        ttt_model.eval()
        loss_all = 0
        total = 0
        correct = 0
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output_g = ttt_model(data)
            # loss = loss_g(output_g, target)

            # loss_all += loss.item()
            total += target.size(0)
            pred = output_g.data.max(1)[1]
            correct += pred.eq(target.view(-1)).sum().item()
        weight_list.append(correct/total)

    acc_sum = sum(weight_list)
    w_list = [x/acc_sum for x in weight_list]
    return w_list

