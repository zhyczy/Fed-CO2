import numpy as np


def others_test(version, model, p_model, extra_modules, data_loader, loss_fun, global_prototype, device, flog):
    return


def peer_test_softmax_learnable(model, p_model, extra_models, data_loader, loss_fun, device):
    model.eval()
    p_model.eval()
    g_tau_emb, p_tau_emb = extra_models
    g_tau_emb.eval()
    p_tau_emb.eval()

    g_tau = torch.relu(g_tau_emb(torch.tensor([0], dtype=torch.long).to(device)))+0.2
    p_tau = torch.relu(p_tau_emb(torch.tensor([0], dtype=torch.long).to(device)))+1

    loss_all, loss_ga, loss_pa = 0, 0, 0
    total = 0
    correct, correct_g, correct_p = 0, 0, 0

    for data, target in data_loader:

        data = data.to(device)
        target = target.to(device)
        output_g = softmax_sharp(model(data), tau=g_tau)
        output_p = softmax_sharp(p_model(data), tau=p_tau)

        output =  output_g + output_p
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


def peer_test_softmax_normal(model, p_model, data_loader, loss_fun, device):
    model.eval()
    p_model.eval()
    loss_all, loss_ga, loss_pa = 0, 0, 0
    total = 0
    correct, correct_g, correct_p = 0, 0, 0
    for data, target in data_loader:

        data = data.to(device)
        target = target.to(device)
        output_g = F.softmax(model(data), dim=1)
        output_p = F.softmax(p_model(data), dim=1)

        output = output_g + output_p
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


def peer_test_KL(model, p_model, a_model, data_loader, loss_fun, device):
    model.eval()
    p_model.eval()
    a_model.eval()
    loss_all, loss_ga, loss_pa, loss_gpa = 0, 0, 0, 0
    total = 0
    correct, correct_g, correct_p, correct_gp = 0, 0, 0, 0
    for data, target in data_loader:

        data = data.to(device)
        target = target.to(device)
        output_g = model(data)
        output_p = p_model(data)
        output = a_model(data)
        output_gp = output_g + output_p

        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        loss_g = loss_fun(output_g, target)
        loss_p = loss_fun(output_p, target)
        loss_gp = loss_fun(output_gp, target)
        loss_ga += loss_g
        loss_pa += loss_p
        loss_gpa += loss_gp

        pred_g = output_g.data.max(1)[1]
        correct_g += pred_g.eq(target.view(-1)).sum().item()
        pred_p = output_p.data.max(1)[1]
        correct_p += pred_p.eq(target.view(-1)).sum().item()
        pred_gp = output_gp.data.max(1)[1]
        correct_gp += pred_gp.eq(target.view(-1)).sum().item()

    test_loss = [loss_all/len(data_loader), loss_ga/len(data_loader), loss_pa/len(data_loader), loss_gpa/len(data_loader)]
    test_acc = [correct/total, correct_g/total, correct_p/total, correct_gp/total]
    return test_loss, test_acc


def peer_test_hyper_special(model, p_model, hnet, data_loader, loss_fun, client_idx, device):
    model.eval()
    p_model.eval()
    hnet.eval()
    loss_all, loss_ga, loss_pa = 0, 0, 0
    total = 0
    correct, correct_g, correct_p = 0, 0, 0

    node_weights, node_weights_g = hnet(torch.tensor([client_idx], dtype=torch.long).to(device), True)
    p_model.load_state_dict(node_weights, strict=False)
    model.load_state_dict(node_weights_g, strict=False)
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


def peer_test_hyper(model, p_model, hnet, data_loader, loss_fun, client_idx, device):
    model.eval()
    p_model.eval()
    hnet.eval()
    loss_all, loss_ga, loss_pa = 0, 0, 0
    total = 0
    correct, correct_g, correct_p = 0, 0, 0

    node_weights = hnet(torch.tensor([client_idx], dtype=torch.long).to(device), True)
    p_model.load_state_dict(node_weights, strict=False)
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


def normal_test_softmax(model, data_loader, loss_fun, device):
    model.eval()
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:

        data = data.to(device)
        target = target.to(device)
        output = F.softmax(model(data), dim=1)
        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    return loss_all / len(data_loader), correct/total


def fedtp_test(idx, model, hnet, data_loader, loss_fun, device):
    hnet.eval()
    model.eval()
    loss_all = 0
    total = 0
    correct = 0
    node_weights = hnet(torch.tensor([idx], dtype=torch.long).to(device), True)
    model.load_state_dict(node_weights, strict=False)
    for data, target in data_loader:

        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()


def log_write_dictionary(logfile, dictionary, mode='Loss', data='domainnet', division='Test'):
    dataset = {'domainnet':['Clipart', 'Infograph', 'Painting', 'Quickdraw', 'Real', 'Sketch'],
               'office_home':['Amazon', 'Caltech', 'DSLR', 'Webcam']}
    data_name_list = dataset[data]
    logfile.write('Adapt detail for {:<10s}-Phase'.format(division))
    key_list = list(dictionary.keys())
    for ky in range(len(dictionary.keys())):
        kky = key_list[ky]
        if ky != len(dictionary.keys())-1:
            logfile.write('Site-{:<10s} {:<10s}:{:.4f} | '.format(data_name_list[kky], mode, dictionary[kky]))
        else:
            logfile.write('Site-{:<10s} {:<10s}:{:.4f}'.format(data_name_list[kky], mode, dictionary[kky]))


def show_dictionary(logfile, dictionary, a_iter, mode='Loss', data='domainnet', division='Test'):
    dataset = {'domainnet':['Clipart', 'Infograph', 'Painting', 'Quickdraw', 'Real', 'Sketch'],
               'office_home':['Amazon', 'Caltech', 'DSLR', 'Webcam']}
    data_name_list = dataset[data]
    s1 = 'Adapt detail for {:<10s}-Phase'.format(division)
    key_list = list(dictionary.keys())
    # if division == 'Train':
    #     print("dictionary length: " ,len(key_list))
    for ky in range(len(dictionary.keys())):
        kky = key_list[ky]
        # if a_iter == 1:
        #     print("key: ",kky)
        #     print("Key name: ", data_name_list[kky])
        if ky != len(dictionary.keys())-1:
            s1 += 'Site-{:<10s} {:<10s}:{:.4f} | '.format(data_name_list[kky], mode, dictionary[kky])
        else:
            s1 += 'Site-{:<10s} {:<10s}:{:.4f}'.format(data_name_list[kky], mode, dictionary[kky])
    print(s1)


def log_write_running_dictionary(logfile, dictionary, data='domainnet'):
    dataset = {'domainnet':['Clipart', 'Infograph', 'Painting', 'Quickdraw', 'Real', 'Sketch'],
               'office_home':['Amazon', 'Caltech', 'DSLR', 'Webcam']}
    data_name_list = dataset[data]
    key_list = list(dictionary.keys())
    for ky in range(len(dictionary.keys())):
        kky = key_list[ky]
        running_record = dictionary[kky]
        running_avg = np.mean(running_record)
        running_var = np.var(running_record)
        logfile.write('Site-{:<10s} | T1: {:.4f} | T2: {:.4f} | T3: {:.4f} | T4: {:.4f} | T5: {:.4f}'.format(data_name_list[kky], running_record[0], running_record[1], running_record[2], running_record[3], running_record[4]))
        logfile.write('Site-{:<10s} | Avg: {:.4f} | Var: {:.4f}'.format(data_name_list[kky], running_avg, running_var))


def show_running_dictionary(dictionary, data='domainnet'):
    dataset = {'domainnet':['Clipart', 'Infograph', 'Painting', 'Quickdraw', 'Real', 'Sketch'],
               'office_home':['Amazon', 'Caltech', 'DSLR', 'Webcam']}
    data_name_list = dataset[data]
    key_list = list(dictionary.keys())
    for ky in range(len(dictionary.keys())):
        kky = key_list[ky]
        running_record = dictionary[kky]
        running_avg = np.mean(running_record)
        running_var = np.var(running_record)
        print('Site-{:<10s} | T1: {:.4f} | T2: {:.4f} | T3: {:.4f} | T4: {:.4f} | T5: {:.4f}'.format(data_name_list[kky], running_record[0], running_record[1], running_record[2], running_record[3], running_record[4]))
        print('Site-{:<10s} | Avg: {:.4f} | Var: {:.4f}'.format(data_name_list[kky], running_avg, running_var))

    