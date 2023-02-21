import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict, defaultdict


def others_train(version, model, p_model, Extra_modules, paggregation_models, train_loader, test_loader, optimizer, p_optimizer, loss_fun, criterion_ba, Specific_head, Specific_adaptor, valid_value, client_idx, device, args, a_iter, phase):
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

    elif args.version in [5, 15, 29, 30, 31]:
        a_otimizer = optim.SGD(params=p_model.f_adaptor.parameters(), lr=args.lr, momentum=args.momentum)
        p_optimizer = optim.SGD(params=[{'params':p_model.features.parameters()},
                             {'params':p_model.classifier.parameters()}], lr=args.lr)
        train_loss, train_acc = train_ada_shabby(model, p_model, train_loader, optimizer, 
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
        train_loss, train_acc = train_ada_residual(model, p_model, copy_model, train_loader, optimizer, 
                                          p_optimizer, a_otimizer, loss_fun, criterion_ba, Specific_head, client_idx, device)

    elif args.version in [7, 19]:
        a_otimizer = optim.SGD(params=p_model.f_adaptor.parameters(), lr=args.lr, momentum=args.momentum)
        p_optimizer = optim.SGD(params=[{'params':p_model.features.parameters()},
                             {'params':p_model.classifier.parameters()}], lr=args.lr)
        train_loss, train_acc = train_gen_plus0(model, p_model, train_loader, optimizer, 
                                          p_optimizer, a_otimizer, loss_fun, criterion_ba, Specific_head, Specific_adaptor, client_idx, device)

    elif args.version in [8, 20]:
        a_otimizer = optim.SGD(params=p_model.f_adaptor.parameters(), lr=args.lr, momentum=args.momentum)
        p_optimizer = optim.SGD(params=[{'params':p_model.features.parameters()},
                             {'params':p_model.classifier.parameters()}], lr=args.lr)
        train_loss, train_acc = train_gen_plus1(model, p_model, train_loader, optimizer, 
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
        train_loss, train_acc = train_gen_plus2(model, p_model, copy_model, train_loader, optimizer, 
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
        train_loss, train_acc = train_gen_plus3(model, p_model, copy_model, train_loader, optimizer, 
                                          p_optimizer, a_otimizer, loss_fun, criterion_ba, Specific_head, Specific_adaptor, client_idx, device)                

    elif args.version == 11:
        a_otimizer = optim.SGD(params=p_model.f_adaptor.parameters(), lr=args.lr, momentum=args.momentum)
        p_optimizer = optim.SGD(params=[{'params':p_model.features.parameters()},
                             {'params':p_model.classifier.parameters()}], lr=args.lr)
        train_loss, train_acc = train_ada_shabby_kl(model, p_model, train_loader, optimizer, 
                                          p_optimizer, a_otimizer, loss_fun, criterion_ba, Specific_head, client_idx, device)

    elif args.version == 12:
        a_otimizer = optim.SGD(params=p_model.f_adaptor.parameters(), lr=args.lr, momentum=args.momentum)
        p_optimizer = optim.SGD(params=[{'params':p_model.features.parameters()},
                             {'params':p_model.classifier.parameters()}], lr=args.lr)
        train_loss, train_acc = train_ada_shabby_klcr(model, p_model, train_loader, optimizer, 
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
        train_loss, train_acc = train_ada_residual_kl(model, p_model, copy_model, train_loader, optimizer, 
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
        train_loss, train_acc = train_ada_residual_klcr(model, p_model, copy_model, train_loader, optimizer, 
                                          p_optimizer, a_otimizer, loss_fun, criterion_ba, Specific_head, client_idx, device)

    elif args.version == 24:
        a_otimizer = optim.SGD(params=p_model.f_adaptor.parameters(), lr=args.lr, momentum=args.momentum)
        p_optimizer = optim.SGD(params=[{'params':p_model.features.parameters()},
                             {'params':p_model.classifier.parameters()}], lr=args.lr)
        train_loss, train_acc = train_ada_shabby2(model, p_model, train_loader, optimizer, 
                                          p_optimizer, a_otimizer, loss_fun, criterion_ba, Specific_head, client_idx, device)

    elif args.version == 50:
        train_loss, train_acc = train_tt_kll1(model, p_model, train_loader, test_loader, 
                                optimizer, p_optimizer, loss_fun, criterion_ba, Specific_head, valid_value, client_idx, device, a_iter)

    elif args.version in [32, 33]:
        if phase == 'Train':
            train_loss, train_acc = train_gen0(model, p_model, train_loader, optimizer, 
                                          p_optimizer, loss_fun, criterion_ba, Specific_head, client_idx, device)
        elif phase == 'Finetune':
            # Knowledge Distillation flag is set to 1 if g branch is more accurate than p branch on validation dataset
            valid_onehot = {0:1, 1:1, 2:1, 3:0, 4:1, 5:1}
            train_loss, train_acc = finetune_kl(model, p_model, test_loader, optimizer, 
                                          p_optimizer, loss_fun, criterion_ba, Specific_head, valid_value, client_idx, device, a_iter)

    elif args.version == 34:
        train_loss, train_acc = train_kl(model, p_model, train_loader, optimizer, 
                                  p_optimizer, loss_fun, criterion_ba, Specific_head, valid_value, client_idx, device)

    elif args.version == 35:
        train_loss, train_acc = train_kl(model, p_model, train_loader, optimizer, 
                                  p_optimizer, loss_fun, criterion_ba, Extra_modules, valid_value, client_idx, device)

    elif args.version == 36:
        train_loss, train_acc = train_kl_mutual(model, p_model, train_loader, optimizer, 
                                  p_optimizer, loss_fun, criterion_ba, Specific_head, client_idx, device)

    elif args.version == 37:
        train_loss, train_acc = train_kl_mutual(model, p_model, train_loader, optimizer, 
                                  p_optimizer, loss_fun, criterion_ba, Extra_modules, client_idx, device)

    elif args.version == 38:
        train_loss, train_acc = train_KDCL_Linear(model, p_model, train_loader, optimizer, 
                                  p_optimizer, loss_fun, criterion_ba, Specific_head, client_idx, device)

    elif args.version == 39:
        train_loss, train_acc = train_KDCL_wLinear(model, p_model, train_loader, optimizer, 
                                  p_optimizer, loss_fun, criterion_ba, Specific_head, client_idx, device)

    elif args.version == 40:
        a_otimizer = optim.SGD(params=p_model.f_adaptor.parameters(), lr=args.lr, momentum=args.momentum)
        p_optimizer = optim.SGD(params=[{'params':p_model.features.parameters()},
                             {'params':p_model.classifier.parameters()}], lr=args.lr)
        train_loss, train_acc = train_gen_shabby_p(model, p_model, train_loader, optimizer, 
                                          p_optimizer, a_otimizer, loss_fun, criterion_ba, Specific_head, Specific_adaptor, client_idx, a_iter, device)

    elif args.version == 41:
        a_otimizer = optim.SGD(params=p_model.f_adaptor.parameters(), lr=args.lr, momentum=args.momentum)
        p_optimizer = optim.SGD(params=[{'params':p_model.features.parameters()},
                             {'params':p_model.classifier.parameters()}], lr=args.lr)
        train_loss, train_acc = train_gen_shabby_full(model, p_model, train_loader, optimizer, 
                                          p_optimizer, a_otimizer, loss_fun, criterion_ba, Specific_head, Specific_adaptor, client_idx, a_iter, device)

    elif args.version == 42:
        a_otimizer = optim.SGD(params=p_model.f_adaptor.parameters(), lr=args.lr, momentum=args.momentum)
        p_optimizer = optim.SGD(params=[{'params':p_model.features.parameters()},
                             {'params':p_model.classifier.parameters()}], lr=args.lr)
        train_loss, train_acc = train_gen_shabby_g(model, p_model, train_loader, optimizer, 
                                          p_optimizer, a_otimizer, loss_fun, criterion_ba, Specific_head, Specific_adaptor, client_idx, a_iter, device)

    elif args.version == 43:
        a_otimizer = optim.SGD(params=p_model.f_adaptor.parameters(), lr=args.lr, momentum=args.momentum)
        p_optimizer = optim.SGD(params=[{'params':p_model.features.parameters()},
                             {'params':p_model.classifier.parameters()}], lr=args.lr)
        train_loss, train_acc = train_ada_shabby_trial(model, p_model, train_loader, optimizer, 
                                          p_optimizer, a_otimizer, loss_fun, criterion_ba, Specific_head, client_idx, device)

    elif args.version == 44:
        train_loss, train_acc = train_gen_l2(model, p_model, train_loader, optimizer, 
                                          p_optimizer, loss_fun, criterion_ba, Specific_head, client_idx, device)

    elif args.version == 45:
        train_loss, train_acc = train_gen_l1(model, p_model, train_loader, optimizer, 
                                          p_optimizer, loss_fun, criterion_ba, Specific_head, client_idx, device)

    elif args.version == 51:
        train_loss, train_acc = train_gen_orth(model, p_model, train_loader, optimizer, 
                                          p_optimizer, loss_fun, criterion_ba, Specific_head, client_idx, device)

    elif args.version == 53:
        train_loss, train_acc = train_gen_orthG(model, p_model, train_loader, optimizer, 
                                          p_optimizer, loss_fun, criterion_ba, Specific_head, client_idx, device)

    elif args.version == 54:
        train_loss, train_acc = train_gen_orthP(model, p_model, train_loader, optimizer, 
                                          p_optimizer, loss_fun, criterion_ba, Specific_head, client_idx, device)

    elif args.version == 55:
        train_loss, train_acc = train_gen_orthP(model, p_model, train_loader, optimizer, 
                                          p_optimizer, loss_fun, criterion_ba, Extra_modules, client_idx, device)

    elif args.version == 56:
        a_otimizer = optim.SGD(params=paggregation_models[client_idx].parameters(), lr=args.lr)
        train_loss, train_acc = train_gen_orthP_plus(model, p_model, paggregation_models[client_idx], train_loader, optimizer, 
                                          p_optimizer, a_otimizer, loss_fun, criterion_ba, Specific_head, client_idx, device)

    elif args.version == 57:
        a_otimizer = optim.SGD(params=paggregation_models[client_idx].parameters(), lr=args.lr)
        train_loss, train_acc = train_gen_orthP_extend(model, p_model, paggregation_models[client_idx], train_loader, optimizer, 
                                          p_optimizer, a_otimizer, loss_fun, criterion_ba, Specific_head, client_idx, device)

    elif args.version == 62:
        train_loss, train_acc = train_gen_valid(model, personalized_models[client_idx], train_loaders[client_idx], optimizers[client_idx], 
                                          p_optimizer, loss_fun, criterion_ba, Specific_head, client_idx, device)

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


def train_gen_valid(model, p_model, data_loader, optimizer, p_optimizer, loss_fun, loss_g, Specific_heads, client_idx, device):
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


def train_gen_orth(model, p_model, data_loader, optimizer, p_optimizer, loss_fun, loss_g, Specific_heads, client_idx, device):
    client_num = len(Specific_heads.keys())
    assert client_num != 0
    l_lambda = 1/(client_num-1)
    model.train()
    p_model.train()
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        bat_num = data.shape[0]
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
                output_gen = Spe_classifier(feature_g)
                part2 += loss_g(output_gen, target)

        inner_product = torch.mm(feature_g, feature_p.reshape(-1, bat_num).detach())
        part3 = torch.norm(inner_product)
        # part3 = 0.1 * (torch.norm(inner_product)/bat_num)
        # print(inner_product)
        # print(part3)

        loss = part1 + part2 + part3
        loss.backward()
        optimizer.step()

        p_optimizer.zero_grad()
        part1 = loss_fun(output_p, target)

        inner_product = torch.mm(feature_p, feature_g.reshape(-1, bat_num).detach())
        part2 = torch.norm(inner_product)
        # part2 = 0.1 * (torch.norm(inner_product)/bat_num)

        loss_p = part1 + part2
        loss_p.backward()
        p_optimizer.step()

        loss_all += loss_p.item()
        total += target.size(0)
        pred = (output_g.detach()+output_p).data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()
    return loss_all / len(data_loader), correct/total


def train_gen_orthG(model, p_model, data_loader, optimizer, p_optimizer, loss_fun, loss_g, Specific_heads, client_idx, device):
    client_num = len(Specific_heads.keys())
    assert client_num != 0
    l_lambda = 1/(client_num-1)
    model.train()
    p_model.train()
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        bat_num = data.shape[0]
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
                output_gen = Spe_classifier(feature_g)
                part2 += loss_g(output_gen, target)

        inner_product = torch.mm(feature_g, feature_p.reshape(-1, bat_num).detach())
        part3 = torch.norm(inner_product)
        # part3 = 0.1 * (torch.norm(inner_product)/bat_num)
        # print(inner_product)
        # print(part3)

        loss = part1 + part2 + part3
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


def train_gen_orthP(model, p_model, data_loader, optimizer, p_optimizer, loss_fun, loss_g, Specific_heads, client_idx, device):
    client_num = len(Specific_heads.keys())
    assert client_num != 0
    l_lambda = 1/(client_num-1)
    model.train()
    p_model.train()
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        bat_num = data.shape[0]
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
                output_gen = Spe_classifier(feature_g)
                part2 += loss_g(output_gen, target)

        loss = part1 + part2
        loss.backward()
        optimizer.step()

        p_optimizer.zero_grad()
        part1 = loss_fun(output_p, target)

        inner_product = torch.mm(feature_p, feature_g.reshape(-1, bat_num).detach())
        part2 = torch.norm(inner_product)
        # part2 = 0.1 * (torch.norm(inner_product)/bat_num)
        # print(inner_product)
        # print(part2)

        loss_p = part1 + part2
        loss_p.backward()
        p_optimizer.step()

        loss_all += loss_p.item()
        total += target.size(0)
        pred = (output_g.detach()+output_p).data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()
    return loss_all / len(data_loader), correct/total


def train_gen_orthP_plus(model, p_model, back_model, data_loader, optimizer, p_optimizer, a_optimizer, loss_fun, loss_g, Specific_heads, client_idx, device):
    client_num = len(Specific_heads.keys())
    assert client_num != 0
    l_lambda = 1/(client_num-1)
    model.train()
    back_model.train()
    p_model.train()
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        bat_num = data.shape[0]
        data = data.to(device)
        target = target.to(device)
        feature_g = model.produce_feature(data)
        output_g = model.head(feature_g)
        feature_backup = back_model.produce_feature(data)
        output_backup = back_model.head(feature_backup)

        output_p = p_model(data)

        optimizer.zero_grad()
        part1 = loss_g(output_g, target)

        part2 = 0
        for idxx in range(client_num):
            if idxx != client_idx:
                Spe_classifier = Specific_heads[idxx]
                output_gen = Spe_classifier(feature_g)
                part2 += loss_g(output_gen, target)

        loss = part1 + part2
        loss.backward()
        optimizer.step()

        a_optimizer.zero_grad()
        part1 = loss_fun(output_backup, target)

        inner_product = torch.mm(feature_backup, feature_g.reshape(-1, bat_num).detach())
        part2 = torch.norm(inner_product)

        loss_backup = part1 + part2
        loss_backup.backward()
        a_optimizer.step()

        p_optimizer.zero_grad()
        loss_p = loss_fun(output_p, target)
        loss_p.backward()
        p_optimizer.step()

        loss_all += loss_p.item()
        total += target.size(0)
        pred = (output_g.detach()+output_p).data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()
    return loss_all / len(data_loader), correct/total


def train_gen_orthP_extend(model, p_model, back_model, data_loader, optimizer, p_optimizer, a_optimizer, loss_fun, loss_g, Specific_heads, client_idx, device):
    client_num = len(Specific_heads.keys())
    assert client_num != 0
    l_lambda = 1/(client_num-1)
    model.train()
    back_model.train()
    p_model.train()
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        bat_num = data.shape[0]
        data = data.to(device)
        target = target.to(device)
        feature_g = model.produce_feature(data)
        output_g = model.head(feature_g)
        feature_backup = back_model.produce_feature(data)
        output_backup = back_model.head(feature_backup)

        output_p = p_model(data)

        optimizer.zero_grad()
        part1 = loss_g(output_g, target)

        part2 = 0
        for idxx in range(client_num):
            if idxx != client_idx:
                Spe_classifierA, Spe_classifierB = Specific_heads[idxx]
                output_genA = Spe_classifierA(feature_g)
                output_genB = Spe_classifierB(feature_g)
                part2 += loss_g(output_genA, target)
                part2 += loss_g(output_genB, target)

        loss = part1 + part2
        loss.backward()
        optimizer.step()

        a_optimizer.zero_grad()
        part1 = loss_fun(output_backup, target)

        inner_product = torch.mm(feature_backup, feature_g.reshape(-1, bat_num).detach())
        part2 = torch.norm(inner_product)

        loss_backup = part1 + part2
        loss_backup.backward()
        a_optimizer.step()

        p_optimizer.zero_grad()
        loss_p = loss_fun(output_p, target)
        loss_p.backward()
        p_optimizer.step()

        loss_all += loss_p.item()
        total += target.size(0)
        pred = (output_g.detach()+output_p).data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()
    return loss_all / len(data_loader), correct/total


def train_gen_l2(model, p_model, data_loader, optimizer, p_optimizer, loss_fun, loss_g, Specific_heads, client_idx, device):
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
                output_gen = Spe_classifier(feature_g)
                part2 += loss_g(output_gen, target)
        loss = part1 + part2
        loss.backward()
        optimizer.step()

        p_optimizer.zero_grad()
        loss_p = loss_fun(output_p, target)

        w_diff = torch.tensor(0., device=device)
        for w in p_model.parameters():
            w_diff += torch.pow(torch.norm(w), 2)
        w_diff = torch.sqrt(w_diff)
        loss_p += 0.05 * w_diff
 
        loss_p.backward()
        p_optimizer.step()

        loss_all += loss_p.item()
        total += target.size(0)
        pred = (output_g.detach()+output_p).data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()
    return loss_all / len(data_loader), correct/total


def train_gen_l1(model, p_model, data_loader, optimizer, p_optimizer, loss_fun, loss_g, Specific_heads, client_idx, device):
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
                output_gen = Spe_classifier(feature_g)
                part2 += loss_g(output_gen, target)
        loss = part1 + part2
        loss.backward()
        optimizer.step()

        p_optimizer.zero_grad()
        loss_p = loss_fun(output_p, target)

        # print("Before Regularization: ", loss_p.item())
        w_diff = torch.tensor(0., device=device)
        for w in p_model.parameters():
            w_diff += torch.sum(torch.abs(w))
            
        w_diff = w_diff
        loss_p += 1e-5 * w_diff
        # print("After Regularization: ", loss_p.item())

        loss_p.backward()
        p_optimizer.step()

        loss_all += loss_p.item()
        total += target.size(0)
        pred = (output_g.detach()+output_p).data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()
    # assert False
    return loss_all / len(data_loader), correct/total


def finetune_kl(model, p_model, unlabeled_loader, optimizer, p_optimizer, loss_fun, loss_g, Specific_heads, onehot_value, client_idx, device, a_iter):
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    client_num = len(Specific_heads.keys())
    assert client_num != 0
    l_lambda = 1/(client_num-1)
    model.train()
    p_model.train()
    loss_all = 0
    total = 0
    correct = 0

    inner_count = 0
    for data, target in unlabeled_loader:
        if data.shape[0] != 32:
            continue
        data = data.to(device)
        target = target.to(device)
        output_g = model(data)
        output_p = p_model(data)

        if onehot_value == 0:
            optimizer.zero_grad()
            loss = kl_loss(F.log_softmax(output_g, dim=1), F.softmax(output_p.detach(), dim=1))
            loss.backward()
            optimizer.step()
            loss_all += loss.item()

        elif onehot_value == 1:
            p_optimizer.zero_grad()
            loss_p = kl_loss(F.log_softmax(output_p, dim=1), F.softmax(output_g.detach(), dim=1))
            loss_p.backward()
            p_optimizer.step()
            loss_all += loss_p.item()

        total += target.size(0)
        pred = (output_g.detach()+output_p).data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    return loss_all / len(unlabeled_loader), correct/total


def train_kl(model, p_model, data_loader, optimizer, p_optimizer, loss_fun, loss_g, Specific_heads, onehot_value, client_idx, device):
    kl_loss = nn.KLDivLoss(reduction='batchmean')
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
                output_gen = Spe_classifier(feature_g)
                part2 += loss_g(output_gen, target)
        loss = part1 + part2
        if onehot_value == 0:
            part3 = kl_loss(F.log_softmax(output_g, dim=1), F.softmax(output_p.detach(), dim=1))
            loss += part3
        loss.backward()
        optimizer.step()

        p_optimizer.zero_grad()
        loss_p = loss_fun(output_p, target)
        if onehot_value == 1:
            part3 = kl_loss(F.log_softmax(output_p, dim=1), F.softmax(output_g.detach(), dim=1))
            loss_p += part3
        loss_p.backward()
        p_optimizer.step()

        loss_all += loss_p.item()
        total += target.size(0)
        pred = (output_g.detach()+output_p).data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()
    return loss_all / len(data_loader), correct/total


def train_kl_mutual(model, p_model, data_loader, optimizer, p_optimizer, loss_fun, loss_g, Specific_heads, client_idx, device):
    kl_loss = nn.KLDivLoss(reduction='batchmean')
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
                output_gen = Spe_classifier(feature_g)
                part2 += loss_g(output_gen, target)
        part3 = kl_loss(F.log_softmax(output_g, dim=1), F.softmax(output_p.detach(), dim=1))
        loss = part1 + part2 + part3
        loss.backward()
        optimizer.step()

        p_optimizer.zero_grad()
        loss_p = loss_fun(output_p, target)
        part3 = kl_loss(F.log_softmax(output_p, dim=1), F.softmax(output_g.detach(), dim=1))
        loss_p += part3
        loss_p.backward()
        p_optimizer.step()

        loss_all += loss_p.item()
        total += target.size(0)
        pred = (output_g.detach()+output_p).data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()
    return loss_all / len(data_loader), correct/total


def train_KDCL_Linear(model, p_model, data_loader, optimizer, p_optimizer, loss_fun, loss_g, Specific_heads, client_idx, device):
    kl_loss = nn.KLDivLoss(reduction='batchmean')
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

        t_output, ttt_flag = -1, 0
        for dix in range(data.shape[0]):
            lab = int(target[dix])

            g_logits = output_g[dix][lab]
            p_logits = output_p[dix][lab]
            if g_logits > p_logits:
                t_logits = output_g[dix]
            else:
                t_logits = output_p[dix]

            if ttt_flag == 0:
                t_output = t_logits.view(1, -1)
                ttt_flag = 1
            else:
                t_output = torch.cat((t_output, t_logits.view(1, -1)), 0)
        t_output = t_output.detach()

        optimizer.zero_grad()
        part1 = loss_g(output_g, target)

        part2 = 0
        for idxx in range(client_num):
            if idxx != client_idx:
                Spe_classifier = Specific_heads[idxx]
                output_gen = Spe_classifier(feature_g)
                part2 += loss_g(output_gen, target)
        part3 = kl_loss(F.log_softmax(output_g, dim=1), F.softmax(t_output, dim=1))
        loss = part1 + part2 + part3
        loss.backward()
        optimizer.step()

        p_optimizer.zero_grad()
        loss_p = loss_fun(output_p, target)
        part3 = kl_loss(F.log_softmax(output_p, dim=1), F.softmax(t_output, dim=1))
        loss_p += part3
        loss_p.backward()
        p_optimizer.step()

        loss_all += loss_p.item()
        total += target.size(0)
        pred = (output_g.detach()+output_p).data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()
    return loss_all / len(data_loader), correct/total


def train_KDCL_wLinear(model, p_model, data_loader, optimizer, p_optimizer, loss_fun, loss_g, Specific_heads, client_idx, device):
    kl_loss = nn.KLDivLoss(reduction='mean')
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

        t_output, ttt_flag = -1, 0
        kl_weight = []
        bat_num = data.shape[0]
        for dix in range(bat_num):
            lab = int(target[dix])

            g_logits = output_g[dix][lab]
            p_logits = output_p[dix][lab]
            if g_logits > p_logits:
                t_logits = output_g[dix]
                if g_logits.item()<=0:
                    kl_weight.append(0)
                elif g_logits.item()>=1:
                    kl_weight.append(1)
                else:
                    kl_weight.append(g_logits.item())
            else:
                t_logits = output_p[dix]
                if p_logits.item()<=0:
                    kl_weight.append(0)
                elif p_logits.item()>=1:
                    kl_weight.append(1)
                else:
                    kl_weight.append(p_logits.item())

            if ttt_flag == 0:
                t_output = t_logits.view(1, -1)
                ttt_flag = 1
            else:
                t_output = torch.cat((t_output, t_logits.view(1, -1)), 0)
        t_output = t_output.detach()

        optimizer.zero_grad()
        part1 = loss_g(output_g, target)

        part2 = 0
        for idxx in range(client_num):
            if idxx != client_idx:
                Spe_classifier = Specific_heads[idxx]
                output_gen = Spe_classifier(feature_g)
                part2 += loss_g(output_gen, target)

        g_part3, p_part3 = 0, 0
        T_score = F.softmax(t_output, dim=1)
        G_score = F.log_softmax(output_g, dim=1)
        P_score = F.log_softmax(output_p, dim=1)
        for iix in range(bat_num):
            g_part3 += kl_loss(G_score[iix], T_score[iix]) * kl_weight[iix]
            p_part3 += kl_loss(P_score[iix], T_score[iix]) * kl_weight[iix]

        loss = part1 + part2 + (g_part3/bat_num)
        loss.backward()
        optimizer.step()

        p_optimizer.zero_grad()
        loss_p = loss_fun(output_p, target)
        loss_p += (p_part3/bat_num)
        loss_p.backward()
        p_optimizer.step()

        loss_all += loss_p.item()
        total += target.size(0)
        pred = (output_g.detach()+output_p).data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()
    return loss_all / len(data_loader), correct/total


def train_tt_kll1(model, p_model, data_loader, unlabeled_loader, optimizer, p_optimizer, loss_fun, loss_g, Specific_heads, onehot_value, client_idx, device, a_iter):
    kl_loss = nn.KLDivLoss(reduction='batchmean')
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


def train_gen_shabby_p(model, p_model, data_loader, optimizer, p_optimizer, a_otimizer, loss_fun, loss_g, Specific_heads, Specific_adaptors, client_idx, a_iter, device):
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
        part1_g = loss_g(output_g, target)

        part2_g = 0
        part2_p = 0
        part3_p = 0
        for idxx in range(client_num):
            if idxx != client_idx:
                Spe_classifier = Specific_heads[idxx]
                Specific_adaptor = Specific_adaptors[idxx]
                
                output_gen = Spe_classifier(feature_g)
                output_per = Spe_classifier(feature_p)
                if a_iter >= 50:
                    feature_adapt = Specific_adaptor(feature_g).detach()
                    output_adapt = p_model.classifier(feature_adapt)
                    part3_p += loss_g(output_adapt, target)

                part2_g += loss_g(output_gen, target)
                part2_p += loss_g(output_per, target)

        loss = part1_g + part2_g 
        loss.backward()
        optimizer.step()

        p_optimizer.zero_grad()
        part1_p = loss_fun(output_p, target)
        loss_p = part1_p + part2_p + part3_p
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


def train_gen_shabby_g(model, p_model, data_loader, optimizer, p_optimizer, a_otimizer, loss_fun, loss_g, Specific_heads, Specific_adaptors, client_idx, a_iter, device):
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
        part1_g = loss_g(output_g, target)

        part2_g, part3_g, part2_p = 0, 0, 0
        for idxx in range(client_num):
            if idxx != client_idx:
                Spe_classifier = Specific_heads[idxx]
                Specific_adaptor = Specific_adaptors[idxx]
                
                output_gen = Spe_classifier(feature_g)
                output_per = Spe_classifier(feature_p)
                if a_iter >= 50:
                    feature_adapt = Specific_adaptor(feature_g).detach()
                    output_adapt = model.classifier(feature_adapt)
                    part3_g += loss_g(output_adapt, target)

                part2_g += loss_g(output_gen, target)
                part2_p += loss_g(output_per, target)

        loss = part1_g + part2_g + part3_g
        loss.backward()
        optimizer.step()

        p_optimizer.zero_grad()
        part1_p = loss_fun(output_p, target)
        loss_p = part1_p + part2_p
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


def train_gen_shabby_full(model, p_model, data_loader, optimizer, p_optimizer, a_otimizer, loss_fun, loss_g, Specific_heads, Specific_adaptors, client_idx, a_iter, device):
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
        part1_g = loss_g(output_g, target)

        part2_g, part3_g, part2_p, part3_p = 0, 0, 0, 0
        for idxx in range(client_num):
            if idxx != client_idx:
                Spe_classifier = Specific_heads[idxx]
                Specific_adaptor = Specific_adaptors[idxx]
                
                output_gen = Spe_classifier(feature_g)
                output_per = Spe_classifier(feature_p)
                if a_iter >= 50:
                    feature_adapt = Specific_adaptor(feature_g).detach()
                    output_adapt_g = model.classifier(feature_adapt)
                    output_adapt_p = p_model.classifier(feature_adapt)
                    part3_g += loss_g(output_adapt_g, target)
                    part3_p += loss_g(output_adapt_p, target)

                part2_g += loss_g(output_gen, target)
                part2_p += loss_g(output_per, target)

        loss = part1_g + part2_g + part3_g
        loss.backward()
        optimizer.step()

        p_optimizer.zero_grad()
        part1_p = loss_fun(output_p, target)
        loss_p = part1_p + part2_p + part3_p
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


def train_ada_shabby_trial(model, p_model, data_loader, optimizer, p_optimizer, a_otimizer, loss_fun, loss_g, Specific_heads, client_idx, device):
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
        adapt_feature = p_model.make_feature_adapt(feature_g.detach())
        adapt_classifier = copy.deepcopy(p_model.classifier)
        adapt_loss = loss_fun(adapt_classifier(adapt_feature), target)
        adapt_loss.backward()
        a_otimizer.step()

        loss_all += loss_p.item()
        total += target.size(0)
        pred = (output_g.detach()+output_p).data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()
    return loss_all / len(data_loader), correct/total

