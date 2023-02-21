import torch
import copy
import math
import torch.nn as nn
import torch.optim as optim
import sys
import numpy as np
import torch.nn.functional as F
sys.path.append('../alg/')
from alg.fedap import *
from utils.func_u import *


################# Key Function ########################
def communication(args, server_model, models, p_models, extra_modules, paggre_models, client_weights, class_protos, a_iter):
    client_num = len(client_weights)
    class_number = 10
    global_prototypes = {x:0 for x in range(class_number)}
    with torch.no_grad():
        # aggregate params
        if args.mode.lower() == 'fedbn':
            for key in server_model.state_dict().keys():
                if 'bn' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(client_num):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        
        elif args.mode.lower() == 'fedap':
            for cl in range(client_num):
                for key in server_model.state_dict().keys():
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(client_num):
                        temp += client_weights[cl,client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    if 'bn' not in key:
                        paggre_models[cl].state_dict()[key].data.copy_(server_model.state_dict()[key])

            for client_idx in range(client_num):
                models[client_idx].load_state_dict(copy.deepcopy(paggre_models[client_idx].state_dict()))

        elif args.mode.lower() == 'peer':
            if args.version in [63, 64, 67, 68, 69]:
                for key in server_model.state_dict().keys():
                    if 'bn' not in key:
                        temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                        for client_idx in range(client_num):
                            temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                        server_model.state_dict()[key].data.copy_(temp)
                        for client_idx in range(client_num):
                            models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
            elif args.version in [59, 60, 61]:
                for client_idx in range(client_num):
                    weight_list = client_weights[client_idx]
                    for key in server_model.state_dict().keys():
                        if 'num_batches_tracked' in key:
                            server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                        else:
                            temp = torch.zeros_like(server_model.state_dict()[key])
                            for client_jdx in range(client_num):
                                temp += weight_list[client_jdx] * models[client_jdx].state_dict()[key]
                            server_model.state_dict()[key].data.copy_(temp)
                            paggre_models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
                for key in server_model.state_dict().keys():
                    if 'num_batches_tracked' in key:
                        server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                    else:
                        for client_idx in range(client_num):
                            models[client_idx].state_dict()[key].data.copy_(paggre_models[client_idx].state_dict()[key])
                
            elif args.version in [3, 4, 6, 16, 9, 10, 13, 14, 21, 22]:
                for key in server_model.state_dict().keys():
                    if 'num_batches_tracked' in key:
                        server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                    elif 'adap' in key:
                        continue
                    else:
                        temp = torch.zeros_like(server_model.state_dict()[key])
                        for client_idx in range(client_num):
                            temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                        server_model.state_dict()[key].data.copy_(temp)
                        for client_idx in range(client_num):
                            models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
            else:
                for key in server_model.state_dict().keys():
                    if 'num_batches_tracked' in key:
                        server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                    else:
                        temp = torch.zeros_like(server_model.state_dict()[key])
                        for client_idx in range(client_num):
                            temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                        server_model.state_dict()[key].data.copy_(temp)
                        for client_idx in range(client_num):
                            models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])

        elif args.mode == 'AlignFed':
            if args.version == 1:
                for key in server_model.state_dict().keys():
                    if 'head' in key:                   
                        temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                        for client_idx in range(client_num):
                            temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                        server_model.state_dict()[key].data.copy_(temp)
                        for client_idx in range(client_num):
                            models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
            elif args.version == 2:
                if a_iter>=100:
                    for key in server_model.state_dict().keys():
                        if 'head' in key:                   
                            temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                            for client_idx in range(client_num):
                                temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                            server_model.state_dict()[key].data.copy_(temp)
                            for client_idx in range(client_num):
                                models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
            for cix in range(class_number):
                for nix in range(client_num):
                    global_prototypes[cix] += class_protos[nix][cix]/client_num

        elif args.mode == 'COPA':
            for key in server_model.state_dict().keys():
                if 'head' not in key:                   
                    if 'num_batches_tracked' in key:
                        server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                    else:
                        temp = torch.zeros_like(server_model.state_dict()[key])
                        for client_idx in range(client_num):
                            temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                        server_model.state_dict()[key].data.copy_(temp)
                        for client_idx in range(client_num):
                            models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])

        elif args.mode.lower() == 'local':
            return server_model, models, global_prototypes

        elif args.mode.lower() == 'fedper':
            for client_idx in range(client_num):
                p_models[client_idx].state_dict()["classifier.fc1.weight"].data.copy_(models[client_idx].state_dict()["classifier.fc1.weight"])
                p_models[client_idx].state_dict()["classifier.fc1.bias"].data.copy_(models[client_idx].state_dict()["classifier.fc1.bias"])
                p_models[client_idx].state_dict()["classifier.fc2.weight"].data.copy_(models[client_idx].state_dict()["classifier.fc2.weight"])
                p_models[client_idx].state_dict()["classifier.fc2.bias"].data.copy_(models[client_idx].state_dict()["classifier.fc2.bias"])
                p_models[client_idx].state_dict()["classifier.fc3.weight"].data.copy_(models[client_idx].state_dict()["classifier.fc3.weight"])
                p_models[client_idx].state_dict()["classifier.fc3.bias"].data.copy_(models[client_idx].state_dict()["classifier.fc3.bias"])
                # "classifier.fc1.weight", "classifier.fc1.bias", "classifier.fc2.weight", "classifier.fc2.bias", "classifier.fc3.weight", "classifier.fc3.bias"
            for key in server_model.state_dict().keys():
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(client_num):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])

        else:
            for key in server_model.state_dict().keys():
                # num_batches_tracked is a non trainable LongTensor and
                # num_batches_tracked are the same for all clients for the given datasets
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(client_num):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    return server_model, models, global_prototypes


def test(client_idx, model, p_model, extra_modules, data_loader, loss_fun, device, args, hnet, global_prototype, flog=False):
    if args.mode in ['fedbn', 'fedavg', 'fedprox', 'local', 'fedtp', 'fedap', 'AlignFed', 'COPA']:
        test_loss, test_acc = normal_test(model, data_loader, loss_fun, device)
    elif args.mode == 'peer':
        if args.version == 27:
            test_loss, test_acc = peer_test_uppper_bound(model, p_model, data_loader, loss_fun, client_idx, device)
        elif args.version == 30:
            test_loss, test_acc = peer_test_adapt_here(model, p_model, data_loader, loss_fun, client_idx, device)
        elif args.version == 58:
            test_loss, test_acc = peer_gen_test(model, p_model, extra_modules, data_loader, loss_fun, client_idx, device)
        elif args.version in [1, 2]:
            test_loss, test_acc = peer_test1(model, p_model, data_loader, loss_fun, device)
        elif args.version in [3, 4]:
            test_loss, test_acc = peer_test2(model, p_model, data_loader, loss_fun, device)
        elif args.version in [15, 24, 40, 41, 42, 43]:
            test_loss, test_acc = peer_shabby_validate(model, p_model, data_loader, loss_fun, device)
        elif args.version in [23, 26]:
            test_loss, test_acc = peer_shabby_validate1(model, p_model, extra_modules[client_idx], data_loader, loss_fun, device)
        elif args.version in [16]:
            test_loss, test_acc = peer_residual_validate(model, p_model, data_loader, loss_fun, device)
        elif args.version in [5, 11, 12, 19, 20, 31]:
            test_loss, test_acc = peer_shabby_adaptor(model, p_model, extra_modules, data_loader, loss_fun, client_idx, device)
        elif args.version == 29:
            test_loss, test_acc = peer_shabby_adaptor_validate(model, p_model, extra_modules, data_loader, loss_fun, client_idx, device)
        elif args.version in [6, 13, 14, 21, 22]:
            test_loss, test_acc = peer_residual_adaptor(model, p_model, extra_modules, data_loader, loss_fun, client_idx, device)
        elif args.version in [65, 66, 68, 69]:
            test_loss, test_acc = peer_BN_test(model, p_model, extra_modules, data_loader, loss_fun, client_idx, device)
        else:
            test_loss, test_acc = peer_test(model, p_model, data_loader, loss_fun, device)
    elif args.mode == 'fedper':
        test_loss, test_acc = per_test(model, p_model, data_loader, loss_fun, device)
    elif args.mode == 'fedrod':
        test_loss, test_acc = rod_test(model, p_model, data_loader, loss_fun, device)
    elif args.mode == 'fedtp':
        test_loss, test_acc = fedtp_test(client_idx, model, hnet, data_loader, loss_fun, device)
    return test_loss, test_acc


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def set_client_weight(train_loaders, models, n_clients, device):
    bnmlist1, bnvlist1 = [], []
    for i in range(n_clients):
        model = models[i]
        model.eval()
        avgmeta = metacount(get_form(model)[0])
        with torch.no_grad():
            for data, _ in train_loaders[i]:
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


def get_weight_matrix1(bnmlist, bnvlist):
    # default value is 0.5
    model_momentum = 0.3
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
    

def peer_test(model, p_model, data_loader, loss_fun, device):
    model.eval()
    p_model.eval()
    loss_all, loss_ga, loss_pa = 0, 0, 0
    total = 0
    correct, correct_g, correct_p = 0, 0, 0
    for data, target in data_loader:

        data = data.to(device)
        target = target.to(device)
        output_g = model(data)
        output_p = p_model(data)

        output = output_g.detach()+output_p
        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        loss_g = loss_fun(output_g, target)
        loss_p = loss_fun(output_p, target)
        loss_ga += loss_g
        loss_pa += loss_p

        pred_g = output_g.data.max(1)[1]
        correct_g += pred_g.eq(target.view(-1)).sum().item()
        pred_p = output_p.data.max(1)[1]
        correct_p += pred_p.eq(target.view(-1)).sum().item()

    test_loss = [loss_all/len(data_loader), loss_ga/len(data_loader), loss_pa/len(data_loader)]
    test_acc = [correct/total, correct_g/total, correct_p/total]
    return test_loss, test_acc


def peer_test_uppper_bound(model, p_models, data_loader, loss_fun, client_idx, device):
    model.eval()
    client_num = len(p_models)
    assert client_num != 0
    p_model = p_models[client_idx]
    p_model.eval()
    loss_all, loss_ga, loss_pa = 0, 0, 0
    total = 0
    correct, correct_g, correct_p = 0, 0, 0
    other_loss_dict = {}
    other_acc_dict = {}
    for data, target in data_loader:

        data = data.to(device)
        target = target.to(device)
        output_g = model(data)
        output_p = p_model(data)

        output = output_g.detach()+output_p

        for idxx in range(client_num):
            if idxx != client_idx:
                p_model_other = copy.deepcopy(p_models[idxx])
                p_model_other.eval()
                output_other = p_model_other(data)
                loss = loss_fun(output_other, target)
                pred = output_other.max(1)[1]
                correct_other = pred.eq(target.view(-1)).sum().item()

                if idxx in other_loss_dict.keys():
                    other_loss_dict[idxx] += loss.item()
                    other_acc_dict[idxx] += correct_other
                else:
                    other_loss_dict[idxx] = loss.item()
                    other_acc_dict[idxx] = correct_other
                output += output_other.detach()

        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        loss_g = loss_fun(output_g, target)
        loss_p = loss_fun(output_p, target)
        loss_ga += loss_g
        loss_pa += loss_p

        pred_g = output_g.data.max(1)[1]
        correct_g += pred_g.eq(target.view(-1)).sum().item()
        pred_p = output_p.data.max(1)[1]
        correct_p += pred_p.eq(target.view(-1)).sum().item()

    for idxx in range(client_num):
        if idxx != client_idx:
            other_loss_dict[idxx] = other_loss_dict[idxx]/len(data_loader)
            other_acc_dict[idxx] = other_acc_dict[idxx]/total

    test_loss = [loss_all/len(data_loader), loss_ga/len(data_loader), loss_pa/len(data_loader), other_loss_dict]
    test_acc = [correct/total, correct_g/total, correct_p/total, other_acc_dict]
    return test_loss, test_acc


def peer_test_adapt_here(model, p_models, data_loader, loss_fun, client_idx, device):
    model.eval()
    client_num = len(p_models)
    assert client_num != 0
    p_model = p_models[client_idx]
    p_model.eval()
    loss_all, loss_ga, loss_pa, loss_gaad= 0, 0, 0, 0
    total = 0
    correct, correct_g, correct_p, correct_gad= 0, 0, 0, 0
    other_loss_dict = {}
    other_acc_dict = {}
    for data, target in data_loader:

        data = data.to(device)
        target = target.to(device)
        feature_g = model.produce_feature(data)
        feature_p = p_model.produce_feature(data)
        output_g = model.classifier(feature_g)
        output_p = p_model.classifier(feature_p)

        feature_ga = p_model.f_adaptor(feature_g)
        output_ga = p_model.classifier(feature_ga)

        output = output_g.detach()+output_p

        for idxx in range(client_num):
            if idxx != client_idx:
                p_model_other = copy.deepcopy(p_models[idxx])
                p_model_other.eval()
                feature_other = p_model_other.produce_feature(data)
                output_other = p_model.classifier(p_model.f_adaptor(feature_other))
                loss = loss_fun(output_other, target)
                pred = output_other.max(1)[1]
                correct_other = pred.eq(target.view(-1)).sum().item()

                if idxx in other_loss_dict.keys():
                    other_loss_dict[idxx] += loss.item()
                    other_acc_dict[idxx] += correct_other
                else:
                    other_loss_dict[idxx] = loss.item()
                    other_acc_dict[idxx] = correct_other
                output += output_other.detach()

        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        loss_g = loss_fun(output_g, target)
        loss_p = loss_fun(output_p, target)
        loss_ag = loss_fun(output_ga, target)
        loss_ga += loss_g.item()
        loss_pa += loss_p.item()
        loss_gaad += loss_ag.item()

        pred_g = output_g.data.max(1)[1]
        correct_g += pred_g.eq(target.view(-1)).sum().item()
        pred_p = output_p.data.max(1)[1]
        correct_p += pred_p.eq(target.view(-1)).sum().item()
        pred_ag = output_ga.data.max(1)[1]
        correct_gad += pred_ag.eq(target.view(-1)).sum().item()

    for idxx in range(client_num):
        if idxx != client_idx:
            other_loss_dict[idxx] = other_loss_dict[idxx]/len(data_loader)
            other_acc_dict[idxx] = other_acc_dict[idxx]/total

    test_loss = [loss_all/len(data_loader), loss_ga/len(data_loader), loss_pa/len(data_loader), loss_gaad/len(data_loader), other_loss_dict]
    test_acc = [correct/total, correct_g/total, correct_p/total, correct_gad/total, other_acc_dict]
    return test_loss, test_acc


def per_test(model, p_model, data_loader, loss_fun, device):
    model.eval()
    p_model.eval()
    private_param = p_model.state_dict()
    model.load_state_dict(private_param, strict=False)

    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:

        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    return loss_all / len(data_loader), correct/total


def rod_test(model, p_model, data_loader, loss_fun, device):
    model.eval()
    p_model.eval()
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:

        data = data.to(device)
        target = target.to(device)
        mid_features = model.produce_feature(data)
        output_g = model.classifier(mid_features)
        output_p = p_model(mid_features.detach())

        output = output_g.detach()+output_p
        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    return loss_all / len(data_loader), correct/total


def normal_test(model, data_loader, loss_fun, device):
    model.eval()
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:

        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    return loss_all / len(data_loader), correct/total


def peer_gen_test(model, p_model, extra_modules, data_loader, loss_fun, client_idx, device):
    client_num = len(extra_modules)
    assert client_num != 0
    model.eval()
    p_model.eval()
    loss_all, loss_ga, loss_pa, loss_geng = 0, 0, 0, 0
    total = 0
    correct, correct_g, correct_p, correct_geng = 0, 0, 0, 0
    adapt_loss_dict = {}
    adapt_acc_dict = {}
    for data, target in data_loader:

        data = data.to(device)
        target = target.to(device)
        feature_g = model.produce_feature(data)
        output_g = model.classifier(feature_g)
        output_p = p_model(data)

        output = output_g.detach() + output_p.detach()
        output_geng = copy.deepcopy(output_g.detach())

        for idxx in range(client_num):
            if idxx != client_idx:
                spe_classifier = extra_modules[idxx]
                spe_classifier.eval()

                output_ggg = spe_classifier(feature_g)
                loss = loss_fun(output_ggg, target)
                pred = output_ggg.max(1)[1]
                correct_ada = pred.eq(target.view(-1)).sum().item()

                if idxx in adapt_loss_dict.keys():
                    adapt_loss_dict[idxx] += loss.item()
                    adapt_acc_dict[idxx] += correct_ada
                else:
                    adapt_loss_dict[idxx] = loss.item()
                    adapt_acc_dict[idxx] = correct_ada
                output_geng += output_ggg.detach()

        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        loss_g = loss_fun(output_g, target)
        loss_p = loss_fun(output_p, target)
        loss_gg = loss_fun(output_geng, target)
        loss_ga += loss_g.item()
        loss_pa += loss_p.item()
        loss_geng += loss_gg.item()

        pred_g = output_g.data.max(1)[1]
        correct_g += pred_g.eq(target.view(-1)).sum().item()
        pred_p = output_p.data.max(1)[1]
        correct_p += pred_p.eq(target.view(-1)).sum().item()
        pred_geng = output_geng.data.max(1)[1]
        correct_geng += pred_geng.eq(target.view(-1)).sum().item()

    for idxx in range(client_num):
        if idxx != client_idx:
            adapt_loss_dict[idxx] = adapt_loss_dict[idxx]/len(data_loader)
            adapt_acc_dict[idxx] = adapt_acc_dict[idxx]/total

    test_loss = [loss_all/len(data_loader), loss_ga/len(data_loader), loss_pa/len(data_loader), loss_geng/len(data_loader), adapt_loss_dict]
    test_acc = [correct/total, correct_g/total, correct_p/total, correct_geng/total, adapt_acc_dict]
    return test_loss, test_acc


def peer_BN_test(model, p_model, extra_modules, data_loader, loss_fun, client_idx, device):
    client_num = len(extra_modules)
    assert client_num != 0
    model.eval()
    p_model.eval()
    back_model = copy.deepcopy(model)
    back_model.eval()
    loss_all, loss_ga, loss_pa, loss_geng = 0, 0, 0, 0
    total = 0
    correct, correct_g, correct_p, correct_geng = 0, 0, 0, 0
    adapt_loss_dict = {}
    adapt_acc_dict = {}
    for data, target in data_loader:

        data = data.to(device)
        target = target.to(device)
        feature_g = model.produce_feature(data)
        output_g = model.head(feature_g)
        output_p = p_model(data)

        output = output_g.detach() + output_p.detach()
        output_geng = copy.deepcopy(output_g.detach())

        for idxx in range(client_num):
            if idxx != client_idx:
                BN_list = extra_modules[idxx]
                for kky in back_model.state_dict().keys():
                    if 'bn' in kky:
                        if 'num_batches_tracked' not in kky:
                            back_model.state_dict()[kky].data.copy_(BN_list[kky])
                gg_head = copy.deepcopy(model.head)
                gg_head.eval()
                feature_gen = back_model.produce_feature(data)
                output_ggg = gg_head(feature_gen.detach())
                loss = loss_fun(output_ggg, target)
                pred = output_ggg.max(1)[1]
                correct_ada = pred.eq(target.view(-1)).sum().item()

                if idxx in adapt_loss_dict.keys():
                    adapt_loss_dict[idxx] += loss.item()
                    adapt_acc_dict[idxx] += correct_ada
                else:
                    adapt_loss_dict[idxx] = loss.item()
                    adapt_acc_dict[idxx] = correct_ada
                output_geng += output_ggg.detach()

        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        loss_g = loss_fun(output_g, target)
        loss_p = loss_fun(output_p, target)
        loss_gg = loss_fun(output_geng, target)
        loss_ga += loss_g.item()
        loss_pa += loss_p.item()
        loss_geng += loss_gg.item()

        pred_g = output_g.data.max(1)[1]
        correct_g += pred_g.eq(target.view(-1)).sum().item()
        pred_p = output_p.data.max(1)[1]
        correct_p += pred_p.eq(target.view(-1)).sum().item()
        pred_geng = output_geng.data.max(1)[1]
        correct_geng += pred_geng.eq(target.view(-1)).sum().item()

    #     print("output_geng: ", output_geng)
    #     print("output_g: ", output_g)
    # print(correct_geng)
    # print(correct_g)
    # assert False

    for idxx in range(client_num):
        if idxx != client_idx:
            adapt_loss_dict[idxx] = adapt_loss_dict[idxx]/len(data_loader)
            adapt_acc_dict[idxx] = adapt_acc_dict[idxx]/total

    test_loss = [loss_all/len(data_loader), loss_ga/len(data_loader), loss_pa/len(data_loader), loss_geng/len(data_loader), adapt_loss_dict]
    test_acc = [correct/total, correct_g/total, correct_p/total, correct_geng/total, adapt_acc_dict]
    return test_loss, test_acc


def visualize(client_idx, model, p_model, extra_modules, data_loader, loss_fun, device, args, hnet, global_prototype, flag=2):
    model.eval()
    p_model.eval()

    count0, count1, count2, count3, count4, count5, count6, count7 = 0, 0, 0, 0, 0, 0, 0, 0

    entropy0, entropy1, entropy2, entropy3, entropy4, entropy5, entropy6, entropy7 = [], [], [], [], [], [], [], []
    g_entropy0, g_entropy1, g_entropy2, g_entropy3, g_entropy4, g_entropy5, g_entropy6, g_entropy7 = [], [], [], [], [], [], [], []
    p_entropy0, p_entropy1, p_entropy2, p_entropy3, p_entropy4, p_entropy5, p_entropy6, p_entropy7 = [], [], [], [], [], [], [], []

    top2_dif0, top2_dif1, top2_dif2, top2_dif3, top2_dif4, top2_dif5, top2_dif6, top2_dif7 = [], [], [], [], [], [], [], []
    g_top2_dif0, g_top2_dif1, g_top2_dif2, g_top2_dif3, g_top2_dif4, g_top2_dif5, g_top2_dif6, g_top2_dif7 = [], [], [], [], [], [], [], []
    p_top2_dif0, p_top2_dif1, p_top2_dif2, p_top2_dif3, p_top2_dif4, p_top2_dif5, p_top2_dif6, p_top2_dif7 = [], [], [], [], [], [], [], []

    class_number_dict = {x:0 for x in range(10)}

    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)
        output_g = model(data)
        output_p = p_model(data)

        output = output_g.detach()+output_p
        s_out, idxxx = torch.sort(F.softmax(output, dim=1), descending=True)
        sg_out, g_idxxx = torch.sort(F.softmax(output_g, dim=1), descending=True)
        sp_out, p_idxxx = torch.sort(F.softmax(output_p, dim=1), descending=True)

        prediction = output.data.max(1)[1]
        g_prediction = output_g.data.max(1)[1]
        p_prediction = output_p.data.max(1)[1]
        # for la in range(len(target)):
        #     labbb = int(target[la])
        #     class_number_dict[labbb] += 1

        for la in range(len(target)): 
            pred = prediction[la]
            g_pred = g_prediction[la]
            p_pred = p_prediction[la]

            ent, g_ent, p_ent = 0, 0, 0
            dif, g_dif, p_dif = 0, 0, 0
            for ddd in range(10):
                ent += -s_out[la][ddd]*torch.log(s_out[la][ddd])
                g_ent += -sg_out[la][ddd]*torch.log(sg_out[la][ddd])
                p_ent += -sp_out[la][ddd]*torch.log(sp_out[la][ddd])
            dif += (s_out[la][0]-s_out[la][1])
            g_dif += (sg_out[la][0]-sg_out[la][1])
            p_dif += (sp_out[la][0]-sp_out[la][1])

            if int(pred) == int(target[la]):
                if int(p_pred) == int(target[la]):
                    if int(g_pred) == int(target[la]):
                        count0 += 1
                        entropy0.append(ent.cpu())
                        g_entropy0.append(g_ent.cpu())
                        p_entropy0.append(p_ent.cpu())
                        top2_dif0.append(dif.cpu())
                        g_top2_dif0.append(g_dif.cpu())
                        p_top2_dif0.append(p_dif.cpu())
                    elif int(g_pred) != int(target[la]):
                        count1 += 1
                        entropy1.append(ent.cpu())
                        g_entropy1.append(g_ent.cpu())
                        p_entropy1.append(p_ent.cpu())
                        top2_dif1.append(dif.cpu())
                        g_top2_dif1.append(g_dif.cpu())
                        p_top2_dif1.append(p_dif.cpu())
                elif int(p_pred) != int(target[la]):
                    if int(g_pred) == int(target[la]):
                        count2 += 1
                        entropy2.append(ent.cpu())
                        g_entropy2.append(g_ent.cpu())
                        p_entropy2.append(p_ent.cpu())
                        top2_dif2.append(dif.cpu())
                        g_top2_dif2.append(g_dif.cpu())
                        p_top2_dif2.append(p_dif.cpu())
                    elif int(g_pred) != int(target[la]):
                        count3 += 1
                        entropy3.append(ent.cpu())
                        g_entropy3.append(g_ent.cpu())
                        p_entropy3.append(p_ent.cpu())
                        top2_dif3.append(dif.cpu())
                        g_top2_dif3.append(g_dif.cpu())
                        p_top2_dif3.append(p_dif.cpu())
            elif int(pred) != int(target[la]):
                if int(p_pred) == int(target[la]):
                    if int(g_pred) == int(target[la]):
                        count4 += 1
                        entropy4.append(ent.cpu())
                        g_entropy4.append(g_ent.cpu())
                        p_entropy4.append(p_ent.cpu())
                        top2_dif4.append(dif.cpu())
                        g_top2_dif4.append(g_dif.cpu())
                        p_top2_dif4.append(p_dif.cpu())
                    elif int(g_pred) != int(target[la]):
                        count5 += 1
                        entropy5.append(ent.cpu())
                        g_entropy5.append(g_ent.cpu())
                        p_entropy5.append(p_ent.cpu())
                        top2_dif5.append(dif.cpu())
                        g_top2_dif5.append(g_dif.cpu())
                        p_top2_dif5.append(p_dif.cpu())
                elif int(p_pred) != int(target[la]):
                    if int(g_pred) == int(target[la]):
                        count6 += 1
                        entropy6.append(ent.cpu())
                        g_entropy6.append(g_ent.cpu())
                        p_entropy6.append(p_ent.cpu())
                        top2_dif6.append(dif.cpu())
                        g_top2_dif6.append(g_dif.cpu())
                        p_top2_dif6.append(p_dif.cpu())
                    elif int(g_pred) != int(target[la]):
                        count7 += 1
                        entropy7.append(ent.cpu())
                        g_entropy7.append(g_ent.cpu())
                        p_entropy7.append(p_ent.cpu())
                        top2_dif7.append(dif.cpu())
                        g_top2_dif7.append(g_dif.cpu())
                        p_top2_dif7.append(p_dif.cpu())

            #Wrong prediction
            # if  int(pred)!=int(target[la]):
            #     if flag == 1:
            #         print("Target label: ", int(target[la]), " Predict label: ", int(pred), " Prediction G label: ", int(g_pred), " Prediction P label: ", int(p_pred))
            #         print("Prediction: ", output[la])
            #         print("Prediction G: ", output_g[la])
            #         print("Prediction P: ", output_p[la])
            #         print("  ")

            # #Wong personal prediction
            # if int(p_pred)!= int(target[la]):
            #     if flag == 2:
            #         print("Target label: ", int(target[la]), " Predict label: ", int(pred), " Prediction G label: ", int(g_pred), " Prediction P label: ", int(p_pred))
            #         print("Prediction: ", output[la])
            #         print("Prediction G: ", output_g[la])
            #         print("Prediction P: ", output_p[la])
            #         print("  ")

            # #Wrong global prediction
            # if int(g_pred)!= int(target[la]):
            #     if flag == 3:
            #         print("Target label: ", int(target[la]), " Predict label: ", int(pred), " Prediction G label: ", int(g_pred), " Prediction P label: ", int(p_pred))
            #         print("Prediction: ", output[la])
            #         print("Prediction G: ", output_g[la])
            #         print("Prediction P: ", output_p[la])
            #         print("  ")

    print("Class Distribution: ", class_number_dict)
    print("case1 P right, Pg right, Pp right number:", count0)
    _, var_entropy, avg_entropy, max_entropy, min_entropy = tool_ff(entropy0)
    _, g_var_entropy, g_avg_entropy, g_max_entropy, g_min_entropy = tool_ff(g_entropy0)
    _, p_var_entropy, p_avg_entropy, p_max_entropy, p_min_entropy = tool_ff(p_entropy0)
    _, dif_var, avg_dif, max_dif, min_dif = tool_ff(top2_dif0)
    _, g_dif_var, g_avg_dif, g_max_dif, g_min_dif = tool_ff(g_top2_dif0)
    _, p_dif_var, p_avg_dif, p_max_dif, p_min_dif = tool_ff(p_top2_dif0)
    print("Case1 Detail: ")
    print("Entropy Var: ", var_entropy, " Avg Entropy: ", avg_entropy, " Max Entropy: ", max_entropy, " Min Entropy: ", min_entropy)
    print("G Entropy Var: ", g_var_entropy, " G Avg Entropy: ", g_avg_entropy, " G Max Entropy: ", g_max_entropy, " G Min Entropy: ", g_min_entropy)
    print("P Entropy Var: ", p_var_entropy, " P Avg Entropy: ", p_avg_entropy, " G Max Entropy: ", p_max_entropy, " P Min Entropy: ", p_min_entropy)
    print("Dif Var: ", dif_var, " Avg Dif: ", avg_dif, " Max Dif: ", max_dif, " Min Dif: ", min_dif)
    print("G Dif Var: ", g_dif_var, " G Avg Dif: ", g_avg_dif, " G Max Dif: ", g_max_dif, " G Min Dif: ", g_min_dif)
    print("P Dif Var: ", p_dif_var, " P Avg Dif: ", p_avg_dif, " P Max Dif: ", p_max_dif, " P Min Dif: ", p_min_dif)
    print("     ")

    print("case2 P right, Pg wrong, Pp right number:", count1)
    _, var_entropy, avg_entropy, max_entropy, min_entropy = tool_ff(entropy1)
    _, g_var_entropy, g_avg_entropy, g_max_entropy, g_min_entropy = tool_ff(g_entropy1)
    _, p_var_entropy, p_avg_entropy, p_max_entropy, p_min_entropy = tool_ff(p_entropy1)
    _, dif_var, avg_dif, max_dif, min_dif = tool_ff(top2_dif1)
    _, g_dif_var, g_avg_dif, g_max_dif, g_min_dif = tool_ff(g_top2_dif1)
    _, p_dif_var, p_avg_dif, p_max_dif, p_min_dif = tool_ff(p_top2_dif1)
    print("Case2 Detail: ")
    print("Entropy Var: ", var_entropy, " Avg Entropy: ", avg_entropy, " Max Entropy: ", max_entropy, " Min Entropy: ", min_entropy)
    print("G Entropy Var: ", g_var_entropy, " G Avg Entropy: ", g_avg_entropy, " G Max Entropy: ", g_max_entropy, " G Min Entropy: ", g_min_entropy)
    print("P Entropy Var: ", p_var_entropy, " P Avg Entropy: ", p_avg_entropy, " G Max Entropy: ", p_max_entropy, " P Min Entropy: ", p_min_entropy)
    print("Dif Var: ", dif_var, " Avg Dif: ", avg_dif, " Max Dif: ", max_dif, " Min Dif: ", min_dif)
    print("G Dif Var: ", g_dif_var, " G Avg Dif: ", g_avg_dif, " G Max Dif: ", g_max_dif, " G Min Dif: ", g_min_dif)
    print("P Dif Var: ", p_dif_var, " P Avg Dif: ", p_avg_dif, " P Max Dif: ", p_max_dif, " P Min Dif: ", p_min_dif)
    print("     ")

    print("case3 P right, Pg right, Pp wrong number:", count2)
    _, var_entropy, avg_entropy, max_entropy, min_entropy = tool_ff(entropy2)
    _, g_var_entropy, g_avg_entropy, g_max_entropy, g_min_entropy = tool_ff(g_entropy2)
    _, p_var_entropy, p_avg_entropy, p_max_entropy, p_min_entropy = tool_ff(p_entropy2)
    _, dif_var, avg_dif, max_dif, min_dif = tool_ff(top2_dif2)
    _, g_dif_var, g_avg_dif, g_max_dif, g_min_dif = tool_ff(g_top2_dif2)
    _, p_dif_var, p_avg_dif, p_max_dif, p_min_dif = tool_ff(p_top2_dif2)
    print("Case3 Detail: ")
    print("Entropy Var: ", var_entropy, " Avg Entropy: ", avg_entropy, " Max Entropy: ", max_entropy, " Min Entropy: ", min_entropy)
    print("G Entropy Var: ", g_var_entropy, " G Avg Entropy: ", g_avg_entropy, " G Max Entropy: ", g_max_entropy, " G Min Entropy: ", g_min_entropy)
    print("P Entropy Var: ", p_var_entropy, " P Avg Entropy: ", p_avg_entropy, " G Max Entropy: ", p_max_entropy, " P Min Entropy: ", p_min_entropy)
    print("Dif Var: ", dif_var, " Avg Dif: ", avg_dif, " Max Dif: ", max_dif, " Min Dif: ", min_dif)
    print("G Dif Var: ", g_dif_var, " G Avg Dif: ", g_avg_dif, " G Max Dif: ", g_max_dif, " G Min Dif: ", g_min_dif)
    print("P Dif Var: ", p_dif_var, " P Avg Dif: ", p_avg_dif, " P Max Dif: ", p_max_dif, " P Min Dif: ", p_min_dif)
    print("     ")

    print("case4 P right, Pg wrong, Pp wrong number:", count3)
    _, var_entropy, avg_entropy, max_entropy, min_entropy = tool_ff(entropy3)
    _, g_var_entropy, g_avg_entropy, g_max_entropy, g_min_entropy = tool_ff(g_entropy3)
    _, p_var_entropy, p_avg_entropy, p_max_entropy, p_min_entropy = tool_ff(p_entropy3)
    _, dif_var, avg_dif, max_dif, min_dif = tool_ff(top2_dif3)
    _, g_dif_var, g_avg_dif, g_max_dif, g_min_dif = tool_ff(g_top2_dif3)
    _, p_dif_var, p_avg_dif, p_max_dif, p_min_dif = tool_ff(p_top2_dif3)
    print("Case4 Detail: ")
    print("Entropy Var: ", var_entropy, " Avg Entropy: ", avg_entropy, " Max Entropy: ", max_entropy, " Min Entropy: ", min_entropy)
    print("G Entropy Var: ", g_var_entropy, " G Avg Entropy: ", g_avg_entropy, " G Max Entropy: ", g_max_entropy, " G Min Entropy: ", g_min_entropy)
    print("P Entropy Var: ", p_var_entropy, " P Avg Entropy: ", p_avg_entropy, " G Max Entropy: ", p_max_entropy, " P Min Entropy: ", p_min_entropy)
    print("Dif Var: ", dif_var, " Avg Dif: ", avg_dif, " Max Dif: ", max_dif, " Min Dif: ", min_dif)
    print("G Dif Var: ", g_dif_var, " G Avg Dif: ", g_avg_dif, " G Max Dif: ", g_max_dif, " G Min Dif: ", g_min_dif)
    print("P Dif Var: ", p_dif_var, " P Avg Dif: ", p_avg_dif, " P Max Dif: ", p_max_dif, " P Min Dif: ", p_min_dif)
    print("     ")

    print("case5 P wrong, Pg right, Pp right number:", count4)
    _, var_entropy, avg_entropy, max_entropy, min_entropy = tool_ff(entropy4)
    _, g_var_entropy, g_avg_entropy, g_max_entropy, g_min_entropy = tool_ff(g_entropy4)
    _, p_var_entropy, p_avg_entropy, p_max_entropy, p_min_entropy = tool_ff(p_entropy4)
    _, dif_var, avg_dif, max_dif, min_dif = tool_ff(top2_dif4)
    _, g_dif_var, g_avg_dif, g_max_dif, g_min_dif = tool_ff(g_top2_dif4)
    _, p_dif_var, p_avg_dif, p_max_dif, p_min_dif = tool_ff(p_top2_dif4)
    print("Case5 Detail: ")
    print("Entropy Var: ", var_entropy, " Avg Entropy: ", avg_entropy, " Max Entropy: ", max_entropy, " Min Entropy: ", min_entropy)
    print("G Entropy Var: ", g_var_entropy, " G Avg Entropy: ", g_avg_entropy, " G Max Entropy: ", g_max_entropy, " G Min Entropy: ", g_min_entropy)
    print("P Entropy Var: ", p_var_entropy, " P Avg Entropy: ", p_avg_entropy, " G Max Entropy: ", p_max_entropy, " P Min Entropy: ", p_min_entropy)
    print("Dif Var: ", dif_var, " Avg Dif: ", avg_dif, " Max Dif: ", max_dif, " Min Dif: ", min_dif)
    print("G Dif Var: ", g_dif_var, " G Avg Dif: ", g_avg_dif, " G Max Dif: ", g_max_dif, " G Min Dif: ", g_min_dif)
    print("P Dif Var: ", p_dif_var, " P Avg Dif: ", p_avg_dif, " P Max Dif: ", p_max_dif, " P Min Dif: ", p_min_dif)
    print("     ")

    print("case6 P wrong, Pg wrong, Pp right number:", count5)
    _, var_entropy, avg_entropy, max_entropy, min_entropy = tool_ff(entropy5)
    _, g_var_entropy, g_avg_entropy, g_max_entropy, g_min_entropy = tool_ff(g_entropy5)
    _, p_var_entropy, p_avg_entropy, p_max_entropy, p_min_entropy = tool_ff(p_entropy5)
    _, dif_var, avg_dif, max_dif, min_dif = tool_ff(top2_dif5)
    _, g_dif_var, g_avg_dif, g_max_dif, g_min_dif = tool_ff(g_top2_dif5)
    _, p_dif_var, p_avg_dif, p_max_dif, p_min_dif = tool_ff(p_top2_dif5)
    print("Case6 Detail: ")
    print("Entropy Var: ", var_entropy, " Avg Entropy: ", avg_entropy, " Max Entropy: ", max_entropy, " Min Entropy: ", min_entropy)
    print("G Entropy Var: ", g_var_entropy, " G Avg Entropy: ", g_avg_entropy, " G Max Entropy: ", g_max_entropy, " G Min Entropy: ", g_min_entropy)
    print("P Entropy Var: ", p_var_entropy, " P Avg Entropy: ", p_avg_entropy, " G Max Entropy: ", p_max_entropy, " P Min Entropy: ", p_min_entropy)
    print("Dif Var: ", dif_var, " Avg Dif: ", avg_dif, " Max Dif: ", max_dif, " Min Dif: ", min_dif)
    print("G Dif Var: ", g_dif_var, " G Avg Dif: ", g_avg_dif, " G Max Dif: ", g_max_dif, " G Min Dif: ", g_min_dif)
    print("P Dif Var: ", p_dif_var, " P Avg Dif: ", p_avg_dif, " P Max Dif: ", p_max_dif, " P Min Dif: ", p_min_dif)
    print("     ")

    print("case7 P wrong, Pg right, Pp wrong number:", count6)
    _, var_entropy, avg_entropy, max_entropy, min_entropy = tool_ff(entropy6)
    _, g_var_entropy, g_avg_entropy, g_max_entropy, g_min_entropy = tool_ff(g_entropy6)
    _, p_var_entropy, p_avg_entropy, p_max_entropy, p_min_entropy = tool_ff(p_entropy6)
    _, dif_var, avg_dif, max_dif, min_dif = tool_ff(top2_dif6)
    _, g_dif_var, g_avg_dif, g_max_dif, g_min_dif = tool_ff(g_top2_dif6)
    _, p_dif_var, p_avg_dif, p_max_dif, p_min_dif = tool_ff(p_top2_dif6)
    print("Case7 Detail: ")
    print("Entropy Var: ", var_entropy, " Avg Entropy: ", avg_entropy, " Max Entropy: ", max_entropy, " Min Entropy: ", min_entropy)
    print("G Entropy Var: ", g_var_entropy, " G Avg Entropy: ", g_avg_entropy, " G Max Entropy: ", g_max_entropy, " G Min Entropy: ", g_min_entropy)
    print("P Entropy Var: ", p_var_entropy, " P Avg Entropy: ", p_avg_entropy, " G Max Entropy: ", p_max_entropy, " P Min Entropy: ", p_min_entropy)
    print("Dif Var: ", dif_var, " Avg Dif: ", avg_dif, " Max Dif: ", max_dif, " Min Dif: ", min_dif)
    print("G Dif Var: ", g_dif_var, " G Avg Dif: ", g_avg_dif, " G Max Dif: ", g_max_dif, " G Min Dif: ", g_min_dif)
    print("P Dif Var: ", p_dif_var, " P Avg Dif: ", p_avg_dif, " P Max Dif: ", p_max_dif, " P Min Dif: ", p_min_dif)
    print("     ")

    print("case8 P wrong, Pg wrong, Pp wrong number:", count7)
    _, var_entropy, avg_entropy, max_entropy, min_entropy = tool_ff(entropy7)
    _, g_var_entropy, g_avg_entropy, g_max_entropy, g_min_entropy = tool_ff(g_entropy7)
    _, p_var_entropy, p_avg_entropy, p_max_entropy, p_min_entropy = tool_ff(p_entropy7)
    _, dif_var, avg_dif, max_dif, min_dif = tool_ff(top2_dif7)
    _, g_dif_var, g_avg_dif, g_max_dif, g_min_dif = tool_ff(g_top2_dif7)
    _, p_dif_var, p_avg_dif, p_max_dif, p_min_dif = tool_ff(p_top2_dif7)
    print("Case8 Detail: ")
    print("Entropy Var: ", var_entropy, " Avg Entropy: ", avg_entropy, " Max Entropy: ", max_entropy, " Min Entropy: ", min_entropy)
    print("G Entropy Var: ", g_var_entropy, " G Avg Entropy: ", g_avg_entropy, " G Max Entropy: ", g_max_entropy, " G Min Entropy: ", g_min_entropy)
    print("P Entropy Var: ", p_var_entropy, " P Avg Entropy: ", p_avg_entropy, " G Max Entropy: ", p_max_entropy, " P Min Entropy: ", p_min_entropy)
    print("Dif Var: ", dif_var, " Avg Dif: ", avg_dif, " Max Dif: ", max_dif, " Min Dif: ", min_dif)
    print("G Dif Var: ", g_dif_var, " G Avg Dif: ", g_avg_dif, " G Max Dif: ", g_max_dif, " G Min Dif: ", g_min_dif)
    print("P Dif Var: ", p_dif_var, " P Avg Dif: ", p_avg_dif, " P Max Dif: ", p_max_dif, " P Min Dif: ", p_min_dif)
    print("     ")


def tool_ff(check_list):
    list_sum = sum(check_list)
    list_var = np.var(np.asarray(check_list))
    if len(check_list)==0:
        list_avg = 0
        list_max = 0
        list_min = 0
    else:
        list_avg = list_sum/len(check_list)
        list_max = max(check_list)
        list_min = min(check_list)
    return list_sum, list_var, list_avg, list_max, list_min


def visualize_d(client_idx, model, p_model, extra_modules, data_loader, loss_fun, device, args, hnet, global_prototype, flag=2):
    model.eval()
    p_model.eval()

    count, count_g, count_p, total = 0, 0, 0, 0

    entropy = []
    g_entropy = []
    p_entropy = []

    top2_dif = []
    g_top2_dif = []
    p_top2_dif = []

    class_number_dict = {x:0 for x in range(10)}

    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)
        output_g = model(data)
        output_p = p_model(data)

        output = output_g.detach()+output_p
        total += target.size(0)
        s_out, idxxx = torch.sort(F.softmax(output, dim=1), descending=True)
        sg_out, g_idxxx = torch.sort(F.softmax(output_g, dim=1), descending=True)
        sp_out, p_idxxx = torch.sort(F.softmax(output_p, dim=1), descending=True)

        prediction = output.data.max(1)[1]
        g_prediction = output_g.data.max(1)[1]
        p_prediction = output_p.data.max(1)[1]
        for la in range(len(target)):
            labbb = int(target[la])
            class_number_dict[labbb] += 1

        for la in range(len(target)): 
            pred = prediction[la]
            g_pred = g_prediction[la]
            p_pred = p_prediction[la]

            ent, g_ent, p_ent = 0, 0, 0
            dif, g_dif, p_dif = 0, 0, 0
            for ddd in range(10):
                ent += -s_out[la][ddd]*torch.log(s_out[la][ddd])
                g_ent += -sg_out[la][ddd]*torch.log(sg_out[la][ddd])
                p_ent += -sp_out[la][ddd]*torch.log(sp_out[la][ddd])
            dif += (s_out[la][0]-s_out[la][1])
            g_dif += (sg_out[la][0]-sg_out[la][1])
            p_dif += (sp_out[la][0]-sp_out[la][1])

            if int(pred) == int(target[la]):
                count += 1
            if int(g_pred) == int(target[la]):
                count_g += 1
            if int(p_pred) == int(target[la]):
                count_p += 1

            entropy.append(ent.cpu())
            g_entropy.append(g_ent.cpu())
            p_entropy.append(p_ent.cpu())
            top2_dif.append(dif.cpu())
            g_top2_dif.append(g_dif.cpu())
            p_top2_dif.append(p_dif.cpu())


    print("Class Distribution: ", class_number_dict)
    print("Total right rate:", count/total)
    print("G_branch right rate:", count_g/total)
    print("P_branch right rate:", count_p/total)
    _, var_entropy, avg_entropy, max_entropy, min_entropy = tool_ff(entropy)
    _, g_var_entropy, g_avg_entropy, g_max_entropy, g_min_entropy = tool_ff(g_entropy)
    _, p_var_entropy, p_avg_entropy, p_max_entropy, p_min_entropy = tool_ff(p_entropy)
    _, dif_var, avg_dif, max_dif, min_dif = tool_ff(top2_dif)
    _, g_dif_var, g_avg_dif, g_max_dif, g_min_dif = tool_ff(g_top2_dif)
    _, p_dif_var, p_avg_dif, p_max_dif, p_min_dif = tool_ff(p_top2_dif)
    print("Entropy Var: ", var_entropy, " Avg Entropy: ", avg_entropy, " Max Entropy: ", max_entropy, " Min Entropy: ", min_entropy)
    print("G Entropy Var: ", g_var_entropy, " G Avg Entropy: ", g_avg_entropy, " G Max Entropy: ", g_max_entropy, " G Min Entropy: ", g_min_entropy)
    print("P Entropy Var: ", p_var_entropy, " P Avg Entropy: ", p_avg_entropy, " G Max Entropy: ", p_max_entropy, " P Min Entropy: ", p_min_entropy)
    print("Dif Var: ", dif_var, " Avg Dif: ", avg_dif, " Max Dif: ", max_dif, " Min Dif: ", min_dif)
    print("G Dif Var: ", g_dif_var, " G Avg Dif: ", g_avg_dif, " G Max Dif: ", g_max_dif, " G Min Dif: ", g_min_dif)
    print("P Dif Var: ", p_dif_var, " P Avg Dif: ", p_avg_dif, " P Max Dif: ", p_max_dif, " P Min Dif: ", p_min_dif)


def visualize_combination(client_idx, model, p_models, data_loader, loss_fun, device, args, flag=2, daset='domainnet'):
    model.eval()
    if daset == 'domainnet':
        datasets = ['Clipart', 'Infograph', 'Painting', 'Quickdraw', 'Real', 'Sketch']
    elif daset == 'office_home':
        datasets = ['Amazon', 'Caltech', 'DSLR', 'Webcam']
    client_num = len(p_models)
    for ccdx in range(client_num):
        p_model = p_models[ccdx]
        p_model.eval()
        count, count_g, count_p, total = 0, 0, 0, 0

        entropy = []
        g_entropy = []
        p_entropy = []

        top2_dif = []
        g_top2_dif = []
        p_top2_dif = []

        class_number_dict = {x:0 for x in range(10)}

        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            output_g = model(data)
            output_p = p_model(data)

            output = output_g.detach()+output_p
            total += target.size(0)
            s_out, idxxx = torch.sort(F.softmax(output, dim=1), descending=True)
            sg_out, g_idxxx = torch.sort(F.softmax(output_g, dim=1), descending=True)
            sp_out, p_idxxx = torch.sort(F.softmax(output_p, dim=1), descending=True)

            prediction = output.data.max(1)[1]
            g_prediction = output_g.data.max(1)[1]
            p_prediction = output_p.data.max(1)[1]
            for la in range(len(target)):
                labbb = int(target[la])
                class_number_dict[labbb] += 1

            for la in range(len(target)): 
                pred = prediction[la]
                g_pred = g_prediction[la]
                p_pred = p_prediction[la]

                ent, g_ent, p_ent = 0, 0, 0
                dif, g_dif, p_dif = 0, 0, 0
                for ddd in range(10):
                    ent += -s_out[la][ddd]*torch.log(s_out[la][ddd])
                    g_ent += -sg_out[la][ddd]*torch.log(sg_out[la][ddd])
                    p_ent += -sp_out[la][ddd]*torch.log(sp_out[la][ddd])
                dif += (s_out[la][0]-s_out[la][1])
                g_dif += (sg_out[la][0]-sg_out[la][1])
                p_dif += (sp_out[la][0]-sp_out[la][1])

                if int(pred) == int(target[la]):
                    count += 1
                if int(g_pred) == int(target[la]):
                    count_g += 1
                if int(p_pred) == int(target[la]):
                    count_p += 1

                entropy.append(ent.cpu())
                g_entropy.append(g_ent.cpu())
                p_entropy.append(p_ent.cpu())
                top2_dif.append(dif.cpu())
                g_top2_dif.append(g_dif.cpu())
                p_top2_dif.append(p_dif.cpu())

        print(' Test model from-{:<10s} dataset'.format(datasets[ccdx]))
        print("Total right rate:", count/total)
        print("G_branch right rate:", count_g/total)
        print("P_branch right rate:", count_p/total)
        _, var_entropy, avg_entropy, _, _ = tool_ff(entropy)
        _, g_var_entropy, g_avg_entropy, _, _ = tool_ff(g_entropy)
        _, p_var_entropy, p_avg_entropy, _, _ = tool_ff(p_entropy)
        _, dif_var, avg_dif, _, _ = tool_ff(top2_dif)
        _, g_dif_var, g_avg_dif, _, _ = tool_ff(g_top2_dif)
        _, p_dif_var, p_avg_dif, _, _ = tool_ff(p_top2_dif)
        print("Entropy Var: ", var_entropy, " Avg Entropy: ", avg_entropy)
        print("G Entropy Var: ", g_var_entropy, " G Avg Entropy: ", g_avg_entropy)
        print("P Entropy Var: ", p_var_entropy, " P Avg Entropy: ", p_avg_entropy)
        print("Dif Var: ", dif_var, " Avg Dif: ", avg_dif)
        print("G Dif Var: ", g_dif_var, " G Avg Dif: ", g_avg_dif)
        print("P Dif Var: ", p_dif_var, " P Avg Dif: ", p_avg_dif)
        print(" ")


def adapt_lambda(client_idx, model, p_model, data_loader, test_loader, loss_fun, device, args, flag=2, daset='domainnet'):
    model.eval()
    p_model.eval()
    total = 0
    correct, correct_g, correct_p = 0, 0, 0
    for data, target in data_loader:

        data = data.to(device)
        target = target.to(device)
        output_g = model(data)
        output_p = p_model(data)

        output = output_g.detach()+output_p
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        pred_g = output_g.data.max(1)[1]
        correct_g += pred_g.eq(target.view(-1)).sum().item()
        pred_p = output_p.data.max(1)[1]
        correct_p += pred_p.eq(target.view(-1)).sum().item()

    rate_p = correct_p/total
    rate_g = correct_g/total
    rate_sum = rate_p + rate_g
    print("P acc: ", rate_p," G acc: ", rate_g)
    ratio_p = rate_p/rate_sum
    ratio_g = rate_g/rate_sum
    print("P ratio: ", ratio_p, " G ratio: ", ratio_g)
    print("Oiginal Acc rate: ", correct/total)

    correct = 0
    total = 0
    for data, target in test_loader:

        data = data.to(device)
        target = target.to(device)
        output_g = model(data)
        output_p = p_model(data)
        output = ratio_g * output_g + ratio_p * output_p

        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()
        total += target.size(0)

    print("Combination Acc rate: ", correct/total)