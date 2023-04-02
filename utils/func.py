import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict, defaultdict


def others_train(version, model, p_model, Extra_modules, paggregation_models, train_loader, test_loader, optimizer, p_optimizer, loss_fun, criterion_ba, Specific_head, Specific_adaptor, valid_value, client_idx, device, args, a_iter, phase):
    if version == 2:
        a_optimizer = optim.SGD(params=paggregation_models[client_idx].parameters(), lr=args.lr)
        train_loss, train_acc = train_kl_2phase_0(model, p_model, paggregation_models[client_idx], train_loader, p_optimizer,
        a_optimizer, loss_fun, criterion_ba, a_iter, device)

    elif version == 3:
        a_optimizer = optim.SGD(params=paggregation_models[client_idx].parameters(), lr=args.lr)
        train_loss, train_acc = train_kl_2phase_gen_p(model, p_model, paggregation_models[client_idx], Specific_head, train_loader, 
        p_optimizer, a_optimizer, loss_fun, criterion_ba, a_iter, client_idx, device)

    elif version == 4:
        a_optimizer = optim.SGD(params=paggregation_models[client_idx].parameters(), lr=args.lr)
        train_loss, train_acc = train_kl_2phase_gen_g(model, p_model, paggregation_models[client_idx], Specific_head, train_loader, 
        optimizer, p_optimizer, a_optimizer, loss_fun, criterion_ba, client_idx, device)

    elif version == 5:
        a_optimizer = optim.SGD(params=paggregation_models[client_idx].parameters(), lr=args.lr)
        train_loss, train_acc = train_kl_2phase_gen_full(model, p_model, paggregation_models[client_idx], Specific_head, 
            train_loader, optimizer, p_optimizer, a_optimizer, loss_fun, criterion_ba, client_idx, device)

    elif version == 6:
        a_optimizer = optim.SGD(params=paggregation_models[client_idx].parameters(), lr=args.lr)
        train_loss, train_acc = train_kl_2phase_gtrain(model, p_model, paggregation_models[client_idx], train_loader, 
            optimizer, p_optimizer, a_optimizer, loss_fun, criterion_ba, device)

    elif version == 7:
        a_optimizer = optim.SGD(params=paggregation_models[client_idx].parameters(), lr=args.lr)
        train_loss, train_acc = train_kl_2phase_gen_g_gtrain(model, p_model, paggregation_models[client_idx], Specific_head, 
            train_loader, optimizer, p_optimizer, a_optimizer, loss_fun, criterion_ba, client_idx, device)

    elif version == 8:
        a_optimizer = optim.SGD(params=paggregation_models[client_idx].parameters(), lr=args.lr)
        train_loss, train_acc = train_kl_2phase_gen_full_gtrain(model, p_model, paggregation_models[client_idx], Specific_head, 
            train_loader, optimizer, p_optimizer, a_optimizer, loss_fun, criterion_ba, client_idx, device)

    elif version == 9:
        train_loss, train_acc = train_gen_sep_kl(model, p_model, train_loader, optimizer, 
                                          p_optimizer, loss_fun, criterion_ba, Specific_head, client_idx, a_iter, device)

    elif version == 10:
        train_loss, train_acc = train_gen_sep_kl_indepent(model, personalized_models[client_idx], train_loaders[client_idx], optimizers[client_idx], 
                                          p_optimizer, loss_fun, criterion_ba, Specific_head, client_idx, a_iter, device)
                
    elif version == 11:
        train_loss, train_acc = train_gen_sep_kl_indepent_full(model, p_model, train_loader, optimizer, 
                                          p_optimizer, loss_fun, criterion_ba, Specific_head, client_idx, a_iter, device)

    elif version in [12, 13]:
        train_loss, train_acc = train_softmax(model, train_loader, optimizer, loss_fun, device)

    elif version == 20:
        train_loss, train_acc = train_gen_softmax_nomal(model, p_model, train_loader, optimizer, 
                                          p_optimizer, loss_fun, criterion_ba, Specific_head, client_idx, device)

    elif version == 22:
        a_optimizer = optim.SGD(params=paggregation_models[client_idx].parameters(), lr=args.lr)
        train_loss, train_acc = train_kl_1phase(model, p_model, paggregation_models[client_idx], train_loader, optimizer, 
                                          p_optimizer, a_optimizer, loss_fun, criterion_ba, Specific_head, client_idx, device)
    elif version == 23:
        a_optimizer = optim.SGD(params=paggregation_models[client_idx].parameters(), lr=args.lr)
        train_loss, train_acc = train_kl_1phase_later(model, p_model, paggregation_models[client_idx], train_loader, 
                                          optimizer, p_optimizer, a_optimizer, loss_fun, criterion_ba, Specific_head, client_idx, device)

    elif version == 50:
        train_loss, train_acc = train_learnable_softmax_gen(model, p_model, Extra_modules[client_idx], train_loader, 
                                            optimizer, p_optimizer, loss_fun, criterion_ba, Specific_head, client_idx, device)

    elif version == 51:
        if phase == 'Train':
            train_loss, train_acc = train_softmax_gen(model, p_model, Extra_modules[client_idx], train_loader, 
                                            optimizer, p_optimizer, loss_fun, criterion_ba, Specific_head, client_idx, device)
        elif phase == 'Valid':
            train_loss, train_acc = train_tau(model, p_model, Extra_modules[client_idx], train_loader,
                                            loss_fun, device)

    elif version == 52:
        if phase == 'Train':
            train_loss, train_acc = train_gen0(model, p_model, train_loader, optimizer, 
                                          p_optimizer, loss_fun, criterion_ba, Specific_head, client_idx, device)
        elif phase == 'Valid':
            train_loss, train_acc = train_tau(model, p_model, Extra_modules[client_idx], train_loader,
                                            loss_fun, device)

    elif version == 53:
        if phase == 'Train':
            train_loss, train_acc = train_gen0(model, p_model, train_loader, optimizer, 
                                          p_optimizer, loss_fun, criterion_ba, Specific_head, client_idx, device)
        elif phase == 'Valid':
            train_loss, train_acc = train_tau_tog(model, p_model, Extra_modules[client_idx], train_loader,
                                            loss_fun, device)

    return train_loss, train_acc


def train_tau_tog(model, p_model, a_models, data_loader, loss_fun, device):
    g_tau_emb, p_tau_emb = a_models
    loss_all = 0
    total = 0
    correct = 0

    model.eval()
    p_model.eval()
    a_optimizer_g = optim.SGD(params=g_tau_emb.parameters(), lr=1)
    a_optimizer_p = optim.SGD(params=p_tau_emb.parameters(), lr=1)
    g_tau_emb.train()
    p_tau_emb.train()
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)
        g_tau = torch.relu(g_tau_emb(torch.tensor([0], dtype=torch.long).to(device)))+0.2
        p_tau = torch.relu(p_tau_emb(torch.tensor([0], dtype=torch.long).to(device)))+1
        output_g = softmax_sharp(model(data), tau=g_tau)
        output_p = softmax_sharp(p_model(data), tau=p_tau)

        a_optimizer_g.zero_grad()
        a_optimizer_p.zero_grad()
        loss = loss_fun(output_g+output_p, target)
        loss.backward()
        a_optimizer_g.step()
        a_optimizer_p.step()

        loss_all += loss.item()
        total += target.size(0)
        pred = (output_g.detach()+output_p.detach()).data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()
    return loss_all / len(data_loader), correct/total


def train_tau(model, p_model, a_models, data_loader, loss_fun, device):
    g_tau_emb, p_tau_emb = a_models
    loss_all = 0
    total = 0
    correct = 0

    model.eval()
    p_model.eval()
    a_optimizer_g = optim.SGD(params=g_tau_emb.parameters(), lr=1)
    a_optimizer_p = optim.SGD(params=p_tau_emb.parameters(), lr=1)
    g_tau_emb.train()
    p_tau_emb.train()
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)
        g_tau = torch.relu(g_tau_emb(torch.tensor([0], dtype=torch.long).to(device)))+0.2
        p_tau = torch.relu(p_tau_emb(torch.tensor([0], dtype=torch.long).to(device)))+1
        output_g = softmax_sharp(model(data), tau=g_tau)
        output_p = softmax_sharp(p_model(data), tau=p_tau)

        a_optimizer_g.zero_grad()
        loss = loss_fun(output_g, target)
        loss.backward()
        a_optimizer_g.step()

        a_optimizer_p.zero_grad()
        loss_p = loss_fun(output_p, target)
        loss_p.backward()
        a_optimizer_p.step()

        loss_all += loss_p.item()
        total += target.size(0)
        pred = (output_g.detach()+output_p).data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()
    return loss_all / len(data_loader), correct/total


def train_softmax_gen(model, p_model, a_models, data_loader, optimizer, p_optimizer, loss_fun, loss_g, Specific_heads, client_idx, device):
    client_num = len(Specific_heads.keys())
    assert client_num != 0
    l_lambda = 1/(client_num-1)
    g_tau_emb, p_tau_emb = a_models
    model.train()
    p_model.train()
    g_tau_emb.eval()
    p_tau_emb.eval()

    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)
        g_tau = torch.relu(g_tau_emb(torch.tensor([0], dtype=torch.long).to(device)))+0.2
        p_tau = torch.relu(p_tau_emb(torch.tensor([0], dtype=torch.long).to(device)))+1
        feature_g = model.produce_feature(data)
        output_g = softmax_sharp(model.classifier(feature_g), tau=g_tau)
        feature_p = p_model.produce_feature(data)
        output_p = softmax_sharp(p_model.classifier(feature_p), tau=p_tau)

        optimizer.zero_grad()
        part1 = loss_g(output_g, target)
        part2 = 0
        for idxx in range(client_num):
            if idxx != client_idx:
                Spe_classifier = Specific_heads[idxx]
                Spe_classifier.eval()
                output_gen = F.softmax(Spe_classifier(feature_g), dim=1)
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


def train_gen_softmax_nomal(model, p_model, data_loader, optimizer, p_optimizer, loss_fun, loss_g, Specific_heads, client_idx, device):
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
        output_g = F.softmax(model.classifier(feature_g), dim=1)
        feature_p = p_model.produce_feature(data)
        output_p = F.softmax(p_model.classifier(feature_p), dim=1)

        optimizer.zero_grad()
        part1 = loss_g(output_g, target)

        part2 = 0
        for idxx in range(client_num):
            if idxx != client_idx:
                Spe_classifier = Specific_heads[idxx]
                Spe_classifier.eval()
                output_gen = F.softmax(Spe_classifier(feature_g), dim=1)
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


def train_learnable_softmax_gen(model, p_model, a_models, data_loader, optimizer, p_optimizer, loss_fun, loss_g, Specific_heads, client_idx, device):
    client_num = len(Specific_heads.keys())
    assert client_num != 0
    l_lambda = 1/(client_num-1)
    g_tau_emb, p_tau_emb = a_models
    model.train()
    p_model.train()
    g_tau_emb.eval()
    p_tau_emb.eval()

    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)
        g_tau = torch.relu(g_tau_emb(torch.tensor([0], dtype=torch.long).to(device)))+0.2
        p_tau = torch.relu(p_tau_emb(torch.tensor([0], dtype=torch.long).to(device)))+1
        feature_g = model.produce_feature(data)
        output_g = softmax_sharp(model.classifier(feature_g), tau=g_tau)
        feature_p = p_model.produce_feature(data)
        output_p = softmax_sharp(p_model.classifier(feature_p), tau=p_tau)

        optimizer.zero_grad()
        part1 = loss_g(output_g, target)
        part2 = 0
        for idxx in range(client_num):
            if idxx != client_idx:
                Spe_classifier = Specific_heads[idxx]
                Spe_classifier.eval()
                output_gen = F.softmax(Spe_classifier(feature_g), dim=1)
                part2 += loss_g(output_gen, target)
        loss = part1 + part2
        loss.backward()
        optimizer.step()

        p_optimizer.zero_grad()
        loss_p = loss_fun(output_p, target)
        loss_p.backward()
        p_optimizer.step()

    model.eval()
    p_model.eval()
    a_optimizer_g = optim.SGD(params=g_tau_emb.parameters(), lr=1)
    a_optimizer_p = optim.SGD(params=p_tau_emb.parameters(), lr=1)
    g_tau_emb.train()
    p_tau_emb.train()
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)
        g_tau = torch.relu(g_tau_emb(torch.tensor([0], dtype=torch.long).to(device)))+0.2
        p_tau = torch.relu(p_tau_emb(torch.tensor([0], dtype=torch.long).to(device)))+1
        output_g = softmax_sharp(model(data), tau=g_tau)
        output_p = softmax_sharp(p_model(data), tau=p_tau)

        a_optimizer_g.zero_grad()
        loss = loss_g(output_g, target)
        loss.backward()
        a_optimizer_g.step()

        a_optimizer_p.zero_grad()
        loss_p = loss_fun(output_p, target)
        loss_p.backward()
        a_optimizer_p.step()

        loss_all += loss_p.item()
        total += target.size(0)
        pred = (output_g.detach()+output_p).data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()
    return loss_all / len(data_loader), correct/total


def train_softmax(model, data_loader, optimizer, loss_fun, device):
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        optimizer.zero_grad()

        data = data.to(device)
        target = target.to(device)
        output = F.softmax(model(data), dim=1)
        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        loss.backward()
        optimizer.step()

    return loss_all / len(data_loader), correct/total


def train_kl_1phase_later(model, p_model, a_model, data_loader, optimizer, p_optimizer, a_optimizer, loss_fun, loss_g, Specific_heads, client_idx, device):
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    client_num = len(Specific_heads.keys())
    assert client_num != 0
    l_lambda = 1/(client_num-1)
    model.train()
    p_model.train()
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

    model.eval()
    p_model.eval()
    a_model.train()
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)
        output_g = model(data)
        output_p = p_model(data)
        output_s = a_model(data)

        a_optimizer.zero_grad()
        teacher_logits = output_g + output_p
        loss_a = kl_loss(F.log_softmax(output_s, dim=1), F.softmax(teacher_logits.detach(), dim=1))
        loss_a.backward()
        a_optimizer.step() 

        loss_all += loss_a.item()
        total += target.size(0)
        pred = output_s.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()
    return loss_all / len(data_loader), correct/total


def train_kl_1phase(model, p_model, a_model, data_loader, optimizer, p_optimizer, a_optimizer, loss_fun, loss_g, Specific_heads, client_idx, device):
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    client_num = len(Specific_heads.keys())
    assert client_num != 0
    l_lambda = 1/(client_num-1)
    model.train()
    p_model.train()
    a_model.train()
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
        output_s = a_model(data)

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

        a_optimizer.zero_grad()
        teacher_logits = output_g + output_p
        loss_a = kl_loss(F.log_softmax(output_s, dim=1), F.softmax(teacher_logits.detach(), dim=1))
        loss_a.backward()
        a_optimizer.step() 

        loss_all += loss_a.item()
        total += target.size(0)
        pred = output_s.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()
    return loss_all / len(data_loader), correct/total


def train_gen_sep_kl_indepent_full(model, p_model, data_loader, optimizer, p_optimizer, loss_fun, loss_g, Specific_heads, client_idx, a_iter, device):
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    client_num = len(Specific_heads.keys())
    assert client_num != 0
    l_lambda = 1/(client_num-1)
    loss_all = 0
    total = 0
    correct = 0

    p_model.train()
    model.eval()
    if a_iter == 0:
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            output_p = p_model(data)
            p_optimizer.zero_grad()
            loss = loss_fun(output_p, target)
            loss.backward()
            p_optimizer.step()

            loss_all += loss.item()
            total += target.size(0)
            pred = output_p.data.max(1)[1]
            correct += pred.eq(target.view(-1)).sum().item()
    else:
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            output_g = model(data)
            output_p = p_model(data)
            p_optimizer.zero_grad()
            part1 = kl_loss(F.log_softmax(output_p, dim=1), F.softmax(output_g.detach(), dim=1))
            part2 = loss_fun(output_p, target)
            loss = part1 + part2
            loss.backward()
            p_optimizer.step()

            loss_all += loss.item()
            total += target.size(0)
            pred = output_p.data.max(1)[1]
            correct += pred.eq(target.view(-1)).sum().item()

    p_model.eval()
    model.train()
    if a_iter == 0:
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            feature_g = model.produce_feature(data)
            output_g = model.classifier(feature_g)

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
    else:
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            feature_g = model.produce_feature(data)
            output_g = model.classifier(feature_g)
            output_p = p_model(data)

            optimizer.zero_grad()
            part1 = loss_g(output_g, target)
            part2 = 0
            for idxx in range(client_num):
                if idxx != client_idx:
                    Spe_classifier = Specific_heads[idxx]
                    Spe_classifier.eval()
                    output_gen = Spe_classifier(feature_g)
                    part2 += loss_g(output_gen, target)
            part3 = kl_loss(F.log_softmax(output_g, dim=1), F.softmax(output_p.detach(), dim=1))
            loss = part1 + part2 + part3
            loss.backward()
            optimizer.step()

    return loss_all / len(data_loader), correct/total


def train_gen_sep_kl(model, p_model, data_loader, optimizer, p_optimizer, loss_fun, loss_g, Specific_heads, client_idx, a_iter, device):
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    client_num = len(Specific_heads.keys())
    assert client_num != 0
    l_lambda = 1/(client_num-1)
    loss_all = 0
    total = 0
    correct = 0

    p_model.train()
    if a_iter != 0:
        model.eval()
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            output_g = model(data)
            output_p = p_model(data)
            p_optimizer.zero_grad()
            loss = kl_loss(F.log_softmax(output_p, dim=1), F.softmax(output_g.detach(), dim=1))
            loss.backward()
            p_optimizer.step()

    model.train()
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


def train_kl_2phase_gen_full_gtrain(model, p_model, a_model, Specific_heads, data_loader, optimizer, p_optimizer, a_optimizer, loss_fun, loss_g, client_idx, device):
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    client_num = len(Specific_heads.keys())
    assert client_num != 0
    model.train()
    p_model.train()
    a_model.train()
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
        g_part1 = 0
        p_part1 = 0
        for idxx in range(client_num):
            if idxx != client_idx:
                Spe_classifier = Specific_heads[idxx]
                Spe_classifier.eval()
                output_gen = Spe_classifier(feature_g)
                output_pen = Spe_classifier(feature_p)
                g_part1 += loss_g(output_gen, target)
                p_part1 += loss_g(output_pen, target)
        g_part2 = loss_fun(output_g, target)
        loss = g_part1 + g_part2
        loss.backward()
        optimizer.step()

        p_optimizer.zero_grad()
        p_part2 = loss_fun(output_p, target)
        loss_p = p_part1 + p_part2
        loss_p.backward()
        p_optimizer.step()

    p_model.eval()
    model.eval()
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)

        output = model(data)
        p_output = p_model(data)
        a_output = a_model(data)

        a_optimizer.zero_grad()
        teacher_logits = p_output + output
        a_loss = kl_loss(F.log_softmax(a_output, dim=1), F.softmax(teacher_logits.detach(), dim=1))
        a_loss.backward()
        a_optimizer.step()

        loss_all += a_loss.item()
        total += target.size(0)
        pred = a_output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    return loss_all / len(data_loader), correct/total


def train_kl_2phase_gen_g_gtrain(model, p_model, a_model, Specific_heads, data_loader, optimizer, p_optimizer, a_optimizer, loss_fun, loss_g, client_idx, device):
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    client_num = len(Specific_heads.keys())
    assert client_num != 0
    model.train()
    p_model.train()
    a_model.train()
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
        part1 = loss_fun(output_g, target)
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

    p_model.eval()
    model.eval()
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)

        output = model(data)
        p_output = p_model(data)
        a_output = a_model(data)

        a_optimizer.zero_grad()
        teacher_logits = p_output + output
        a_loss = kl_loss(F.log_softmax(a_output, dim=1), F.softmax(teacher_logits.detach(), dim=1))
        a_loss.backward()
        a_optimizer.step()

        loss_all += a_loss.item()
        total += target.size(0)
        pred = a_output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    return loss_all / len(data_loader), correct/total


def train_kl_2phase_gtrain(model, p_model, a_model, data_loader, optimizer, p_optimizer, a_optimizer, loss_fun, loss_g, device):
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    model.train()
    p_model.train()
    a_model.train()
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)

        output_p = p_model(data)
        output_g = model(data)
        loss_p = loss_fun(output_p, target)
        loss_g = loss_fun(output_g, target)
        p_optimizer.zero_grad()
        optimizer.zero_grad()
        loss_p.backward()
        loss_g.backward()
        p_optimizer.step()
        optimizer.step()

    p_model.eval()
    model.eval()
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)

        output = model(data)
        p_output = p_model(data)
        a_output = a_model(data)

        a_optimizer.zero_grad()
        teacher_logits = p_output + output
        a_loss = kl_loss(F.log_softmax(a_output, dim=1), F.softmax(teacher_logits.detach(), dim=1))
        a_loss.backward()
        a_optimizer.step()

        loss_all += a_loss.item()
        total += target.size(0)
        pred = a_output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    return loss_all / len(data_loader), correct/total


def train_kl_2phase_gen_full(model, p_model, a_model, Specific_heads, data_loader, optimizer, p_optimizer, a_optimizer, loss_fun, loss_g, client_idx, device):
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    client_num = len(Specific_heads.keys())
    model.train()
    p_model.train()
    a_model.train()
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)

        feature_g = model.produce_feature(data)
        feature_p = p_model.produce_feature(data)
        output_p = p_model.classifier(feature_p)

        optimizer.zero_grad()
        loss = 0
        part1 = 0
        for idxx in range(client_num):
            if idxx != client_idx:
                Spe_classifier = Specific_heads[idxx]
                Spe_classifier.eval()
                output_gen = Spe_classifier(feature_g)
                output_pen = Spe_classifier(feature_p)
                loss += loss_g(output_gen, target)
                part1 += loss_g(output_pen, target)
        loss.backward()
        optimizer.step()

        p_optimizer.zero_grad()
        part2 = loss_fun(output_p, target)
        loss_p = part1 + part2
        loss_p.backward()
        p_optimizer.step()

    p_model.eval()
    model.eval()
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)

        output = model(data)
        p_output = p_model(data)
        a_output = a_model(data)

        a_optimizer.zero_grad()
        teacher_logits = p_output + output
        a_loss = kl_loss(F.log_softmax(a_output, dim=1), F.softmax(teacher_logits.detach(), dim=1))
        a_loss.backward()
        a_optimizer.step()

        loss_all += a_loss.item()
        total += target.size(0)
        pred = a_output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    return loss_all / len(data_loader), correct/total


def train_kl_2phase_gen_g(model, p_model, a_model, Specific_heads, data_loader, optimizer, p_optimizer, a_optimizer, loss_fun, loss_g, client_idx, device):
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    client_num = len(Specific_heads.keys())
    model.train()
    p_model.train()
    a_model.train()
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)

        feature_g = model.produce_feature(data)
        feature_p = p_model.produce_feature(data)
        output_p = p_model.classifier(feature_p)

        optimizer.zero_grad()
        loss = 0
        for idxx in range(client_num):
            if idxx != client_idx:
                Spe_classifier = Specific_heads[idxx]
                Spe_classifier.eval()
                output_gen = Spe_classifier(feature_g)
                loss += loss_g(output_gen, target)
        loss.backward()
        optimizer.step()

        p_optimizer.zero_grad()
        loss_p = loss_fun(output_p, target)
        loss_p.backward()
        p_optimizer.step()

    p_model.eval()
    model.eval()
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)

        output = model(data)
        p_output = p_model(data)
        a_output = a_model(data)

        a_optimizer.zero_grad()
        teacher_logits = p_output + output
        a_loss = kl_loss(F.log_softmax(a_output, dim=1), F.softmax(teacher_logits.detach(), dim=1))
        a_loss.backward()
        a_optimizer.step()

        loss_all += a_loss.item()
        total += target.size(0)
        pred = a_output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    return loss_all / len(data_loader), correct/total


def train_kl_2phase_gen_p(model, p_model, a_model, Specific_heads, data_loader, p_optimizer, a_optimizer, loss_fun, loss_g, a_iter, client_idx, device):
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    client_num = len(Specific_heads.keys())
    model.eval()
    p_model.train()
    a_model.train()
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)

        feature_p = p_model.produce_feature(data)
        output_p = p_model.classifier(feature_p)
        part1 = loss_fun(output_p, target)

        part2 = 0
        for idxx in range(client_num):
            if idxx != client_idx:
                Spe_classifier = Specific_heads[idxx]
                Spe_classifier.eval()
                output_gen = Spe_classifier(feature_p)
                part2 += loss_g(output_gen, target)
        loss_p = part1 + part2

        p_optimizer.zero_grad()
        loss_p.backward()
        p_optimizer.step()

        loss_all += loss_p.item()
        total += target.size(0)
        pred = output_p.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    if a_iter == 0:
        a_model = copy.deepcopy(p_model)
    else:
        p_model.eval()
        loss_all = 0
        total = 0
        correct = 0
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            p_output = p_model(data)
            a_output = a_model(data)

            a_optimizer.zero_grad()
            teacher_logits = p_output + output
            a_loss = kl_loss(F.log_softmax(a_output, dim=1), F.softmax(teacher_logits.detach(), dim=1))
            a_loss.backward()
            a_optimizer.step()

            loss_all += a_loss.item()
            total += target.size(0)
            pred = a_output.data.max(1)[1]
            correct += pred.eq(target.view(-1)).sum().item()

    return loss_all / len(data_loader), correct/total


def train_kl_2phase_0(model, p_model, a_model, data_loader, p_optimizer, a_optimizer, loss_fun, loss_g, a_iter, device):
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    model.eval()
    p_model.train()
    a_model.train()
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)

        output_p = p_model(data)
        loss_p = loss_fun(output_p, target)
        p_optimizer.zero_grad()
        loss_p.backward()
        p_optimizer.step()

        loss_all += loss_p.item()
        total += target.size(0)
        pred = output_p.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    if a_iter == 0:
        a_model = copy.deepcopy(p_model)
    else:
        p_model.eval()
        loss_all = 0
        total = 0
        correct = 0
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            p_output = p_model(data)
            a_output = a_model(data)

            a_optimizer.zero_grad()
            teacher_logits = p_output + output
            a_loss = kl_loss(F.log_softmax(a_output, dim=1), F.softmax(teacher_logits.detach(), dim=1))
            a_loss.backward()
            a_optimizer.step()

            loss_all += a_loss.item()
            total += target.size(0)
            pred = a_output.data.max(1)[1]
            correct += pred.eq(target.view(-1)).sum().item()

    return loss_all / len(data_loader), correct/total


def train_gen_sep_kl_indepent(model, p_model, data_loader, optimizer, p_optimizer, loss_fun, loss_g, Specific_heads, client_idx, a_iter, device):
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    client_num = len(Specific_heads.keys())
    assert client_num != 0
    l_lambda = 1/(client_num-1)
    loss_all = 0
    total = 0
    correct = 0

    p_model.train()
    model.eval()
    if a_iter == 0:
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            output_p = p_model(data)
            p_optimizer.zero_grad()
            loss = loss_fun(output_p, target)
            loss.backward()
            p_optimizer.step()

            loss_all += loss.item()
            total += target.size(0)
            pred = output_p.data.max(1)[1]
            correct += pred.eq(target.view(-1)).sum().item()
    else:
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            output_g = model(data)
            output_p = p_model(data)
            p_optimizer.zero_grad()
            part1 = kl_loss(F.log_softmax(output_p, dim=1), F.softmax(output_g.detach(), dim=1))
            part2 = loss_fun(output_p, target)
            loss = part1 + part2
            loss.backward()
            p_optimizer.step()

            loss_all += loss.item()
            total += target.size(0)
            pred = output_p.data.max(1)[1]
            correct += pred.eq(target.view(-1)).sum().item()

    model.train()
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)
        feature_g = model.produce_feature(data)
        output_g = model.classifier(feature_g)

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

