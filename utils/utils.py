import torch
import copy
import math
import torch.nn as nn
import torch.optim as optim
import sys
import numpy as np
import torch.nn.functional as F


################# Key Function ########################
def communication(args, server_model, models, p_models, extra_modules, paggre_models, client_weights):
    client_num = len(client_weights)
    with torch.no_grad():
        # aggregate params 
        if args.mode in ['fedbn', 'fed-co2']:
            for key in server_model.state_dict().keys():
                if 'bn' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(client_num):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])

        elif args.mode == 'copa':
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

        elif args.mode == 'singleset':
            return server_model, models

        elif args.mode == 'fedper':
            for client_idx in range(client_num):
                p_models[client_idx].state_dict()["classifier.fc1.weight"].data.copy_(models[client_idx].state_dict()["classifier.fc1.weight"])
                p_models[client_idx].state_dict()["classifier.fc1.bias"].data.copy_(models[client_idx].state_dict()["classifier.fc1.bias"])
                p_models[client_idx].state_dict()["classifier.fc2.weight"].data.copy_(models[client_idx].state_dict()["classifier.fc2.weight"])
                p_models[client_idx].state_dict()["classifier.fc2.bias"].data.copy_(models[client_idx].state_dict()["classifier.fc2.bias"])
                p_models[client_idx].state_dict()["classifier.fc3.weight"].data.copy_(models[client_idx].state_dict()["classifier.fc3.weight"])
                p_models[client_idx].state_dict()["classifier.fc3.bias"].data.copy_(models[client_idx].state_dict()["classifier.fc3.bias"])
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
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(client_num):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    return server_model, models


def test(client_idx, model, p_model, extra_modules, data_loader, loss_fun, device, args):
    if args.mode in ['fedbn', 'fedavg', 'fedprox', 'singleset', 'moon']:
        test_loss, test_acc = normal_test(model, data_loader, loss_fun, device)
    elif args.mode == 'copa':
        test_loss, test_acc = COPA_test(model, extra_modules, data_loader, loss_fun, client_idx, device)
    elif args.mode == 'fed-co2':
        test_loss, test_acc = co2_test(model, p_model, data_loader, loss_fun, device)
    elif args.mode == 'fedper':
        test_loss, test_acc = per_test(model, p_model, data_loader, loss_fun, device)
    elif args.mode == 'fedrod':
        test_loss, test_acc = rod_test(model, p_model, data_loader, loss_fun, device)
    return test_loss, test_acc


def co2_test(model, p_model, data_loader, loss_fun, device):
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
        loss_ga += loss_g.item()
        loss_pa += loss_p.item()

        pred_g = output_g.data.max(1)[1]
        correct_g += pred_g.eq(target.view(-1)).sum().item()
        pred_p = output_p.data.max(1)[1]
        correct_p += pred_p.eq(target.view(-1)).sum().item()

    test_loss = [loss_all/len(data_loader), loss_ga/len(data_loader), loss_pa/len(data_loader)]
    test_acc = [correct/total, correct_g/total, correct_p/total]
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


def COPA_test(model, extra_modules, data_loader, loss_fun, client_idx, device):
    client_num = len(extra_modules.keys())
    assert client_num != 0
    model.eval()
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)
        u_feature = model.produce_feature(data)
        output = model.head(u_feature)
        for idxx in range(client_num):
            if idxx != client_idx:
                Spe_classifier = extra_modules[idxx]
                Spe_classifier.eval()
                output_gen = Spe_classifier(u_feature)
                output += output_gen
        output = output/client_num
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

