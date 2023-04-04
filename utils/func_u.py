import numpy as np


def others_test(version, model, p_model, extra_modules, data_loader, loss_fun, global_prototype, device, flog):

    return


def softmax_sharp(x, tau=1):
    x_exp = torch.exp(x/tau)
    x_sum = torch.sum(x_exp, dim=1).view(-1,1)
    return x_exp/x_sum


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
               'office_home':['Amazon', 'Caltech', 'DSLR', 'Webcam'],
               'digits':['MNIST', 'SVHN', 'USPS', 'SynthDigits', 'MNIST-M']}
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
               'office_home':['Amazon', 'Caltech', 'DSLR', 'Webcam'],
               'digits':['MNIST', 'SVHN', 'USPS', 'SynthDigits', 'MNIST-M']}
    data_name_list = dataset[data]
    key_list = list(dictionary.keys())
    for ky in range(len(dictionary.keys())):
        kky = key_list[ky]
        running_record = dictionary[kky]
        running_avg = np.mean(running_record)
        running_var = np.var(running_record)
        print('Site-{:<10s} | T1: {:.4f} | T2: {:.4f} | T3: {:.4f} | T4: {:.4f} | T5: {:.4f}'.format(data_name_list[kky], running_record[0], running_record[1], running_record[2], running_record[3], running_record[4]))
        print('Site-{:<10s} | Avg: {:.4f} | Var: {:.4f}'.format(data_name_list[kky], running_avg, running_var))

    