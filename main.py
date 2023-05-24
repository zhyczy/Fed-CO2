"""
federated learning with different aggregation strategy
"""
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import pickle as pkl
from utils.data_utils import *
from utils.methods import local_training
from utils.utils import  communication, test
from nets.models import *
import argparse
import time
import copy
import torchvision.transforms as transforms
import random

     
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)    
    torch.cuda.manual_seed_all(seed) 
    random.seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true', help='whether to log')
    parser.add_argument('--test', action='store_true', help ='test the pretrained model')
    parser.add_argument('--percent', type = float, default= 0.1, help ='percentage of dataset to train')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0, help='momentum')
    parser.add_argument('--batch', type = int, default= 32, help ='batch size')
    parser.add_argument('--iters', type = int, default=300, help = 'iterations for communication')
    parser.add_argument('--wk_iters', type = int, default=1, help = 'optimization iters in local worker between communication')
    parser.add_argument('--mode', type = str, default='fedbn', help='[fedbn | fedavg | fedprox | fedper | fedrod | moon | fed-co2 | copa]')
    parser.add_argument('--mu', type=float, default=1e-3, help='The hyper parameter for fedprox')
    parser.add_argument('--save_path', type = str, default='checkpoint/domainnet', help='path to save the checkpoint')
    parser.add_argument('--resume', action='store_true', help ='resume training from the save path checkpoint')
    parser.add_argument('--dataset', type=str, default='domainnet')
    parser.add_argument('--backbone', type=str, default='alexnet')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--imbalance_train', action='store_true')
    parser.add_argument('--reg_value', type=float, default=10)
    parser.add_argument('--beta', type=float, default=0.3)
    parser.add_argument('--min_img_num', type=int, default=2)
    parser.add_argument('--divide', type=int, default=5)

    args = parser.parse_args()
    data_num_list = []

    if args.dataset == 'domainnet':
        if args.imbalance_train:
            exp_folder = 'fed_domainnet_label_skew'
            print("label skew in Train")
            print("Dirichlet alpha value: ", args.beta)
            print("min image number in each class: ", args.min_img_num)
            print("Divide into %d fold" %args.divide)
            train_loaders, test_loaders, data_num_list = prepare_data_domainNet_partition_train(args)
        else:
            exp_folder = 'fed_domainnet'
            print("No label skew")
            train_loaders, val_loaders, test_loaders = prepare_data_domainNet(args)
    elif args.dataset == 'office':
        if args.imbalance_train:
            exp_folder = 'fed_office_label_skew'
            print("label skew in Train")
            print("Dirichlet alpha value: ", args.beta)
            print("min image number in each class: ", args.min_img_num)
            print("Divide into %d fold" %args.divide)
            train_loaders, test_loaders, data_num_list = prepare_data_office_partition_train(args)
        else:
            exp_folder = 'fed_office'
            print("No label skew")
            train_loaders, val_loaders, test_loaders = prepare_data_office(args)
    elif args.dataset == 'digits':
        if args.imbalance_train:
            exp_folder = 'fed_digits_label_skew'
            print("label skew in Train")
            print("Dirichlet alpha value: ", args.beta)
            print("min image number in each class: ", args.min_img_num)
            print("Divide into %d fold" %args.divide)
            train_loaders, test_loaders, data_num_list = prepare_data_digits_partition_train(args)
        else:
            exp_folder = 'fed_digits'
            print("No label skew")
            train_loaders, test_loaders = prepare_data_digits(args)
        val_loaders = None

    args.save_path = os.path.join(args.save_path, exp_folder)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path, '{}'.format(args.mode))

    log = args.log

    if log:
        if args.dataset == 'domainnet':
            log_path = os.path.join('logs/domainnet', exp_folder)
        elif args.dataset == 'office':
            log_path = os.path.join('logs/office/', exp_folder)
        elif args.dataset == 'digits':
            log_path = os.path.join('logs/digits/', exp_folder)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logfile = open(os.path.join(log_path,'{}_{}.log'.format(args.backbone, args.mode)), 'a')
        # logfile = open(os.path.join(log_path,'{}.log'.format(args.mode)), 'a') 
        logfile.write('==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        logfile.write('===Setting===\n')
        logfile.write("Backnone: %s\n" %args.backbone)
        logfile.write('    lr: {}\n'.format(args.lr))
        logfile.write('    batch: {}\n'.format(args.batch))
        logfile.write('    iters: {}\n'.format(args.iters))

    print(" Federated Algorithm: ", args.mode)
    # setup model
    if args.mode in ['fedper' ,'fedrod']:
        if args.dataset == 'digits':
            server_model = DigitModel_rod().to(device)
        else:
            server_model = AlexNet_rod().to(device)
    elif args.mode == 'fed-co2':
        if args.dataset == 'digits':
            server_model = DigitModel().to(device)
        else:
            server_model = AlexNet().to(device)
    elif args.mode == 'copa':
        if args.dataset == 'digits':
            server_model = DigitModel_head().to(device)
        else:
            server_model = AlexNet_peer().to(device)
    elif args.mode == 'moon':
        if args.dataset == 'digits':
            server_model = DigitModel_moon().to(device)
        else:
            server_model = AlexNet_moon().to(device)
    else:
        if args.dataset == 'digits':
            server_model = DigitModel().to(device)
        else:
            server_model = AlexNet().to(device)
    loss_fun = nn.CrossEntropyLoss()

    # name of each datasets
    if args.dataset == 'domainnet':
        datasets = ['Clipart', 'Infograph', 'Painting', 'Quickdraw', 'Real', 'Sketch']
    elif args.dataset == 'office':
        datasets = ['Amazon', 'Caltech', 'DSLR', 'Webcam']
    elif args.dataset == 'digits':
        datasets = ['MNIST', 'SVHN', 'USPS', 'SynthDigits', 'MNIST-M']

    # federated client number
    client_num = len(datasets)
    if args.imbalance_train:
        total_num = sum(data_num_list)
        print(data_num_list)
    client_weights = [1/client_num for i in range(client_num)]
    
    # each local client model
    models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]
    paggregation_models = []
    personalized_models = []
    extra_modules = []
    
    if args.mode == 'fed-co2':
        personalized_models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]

    elif args.mode in ['fedper', 'fedrod']:
        if args.dataset == 'digits':
            priviate_head = D_Head().to(device)
        else:
            priviate_head = P_Head().to(device)
        personalized_models = [copy.deepcopy(priviate_head).to(device) for idx in range(client_num)]

    elif args.mode == 'moon':
        paggregation_models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]

    best_changed = False

    if args.test:
        print('Loading snapshots...')
        if args.imbalance_train:
            if args.dataset == 'domainnet':
                checkpoint = torch.load('checkpoint/domainnet/fed_domainnet_label_skew/{}'.format(args.mode))
            elif args.dataset == 'office':
                checkpoint = torch.load('checkpoint/domainnet/fed_office_label_skew/{}'.format(args.mode))
            elif args.dataset == 'digits':
                checkpoint = torch.load('checkpoint/domainnet/fed_digits_label_skew/{}'.format(args.mode))
        else:
            if args.dataset == 'domainnet':
                checkpoint = torch.load('checkpoint/domainnet/fed_domainnet/{}'.format(args.mode))
            elif args.dataset == 'office':
                checkpoint = torch.load('checkpoint/domainnet/fed_office/{}'.format(args.mode))
            elif args.dataset == 'digits':
                checkpoint = torch.load('checkpoint/domainnet/fed_digits/{}'.format(args.mode))

        if args.mode.lower() in ['fedbn', 'singleset']:
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
        elif args.mode.lower() in ['fedper', 'fedrod']:
            server_model.load_state_dict(checkpoint['server_model'])
            for client_idx in range(client_num):
                personalized_models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
                models[client_idx].load_state_dict(checkpoint['server_model'])
        elif args.mode.lower() == 'fed-co2':
            for client_idx in range(client_num):
                personalized_models[client_idx].load_state_dict(checkpoint['pmodel_{}'.format(client_idx)])
                models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
        elif args.mode == 'copa':
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
                extra_modules[client_idx] = copy.deepcopy(models[client_idx].head)
        else:
            server_model.load_state_dict(checkpoint['server_model'])
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['server_model'])

        with torch.no_grad():
            for test_idx, test_loader in enumerate(test_loaders):
                if args.mode in ['fed-co2', 'fedper', 'fedrod']:
                    p_model = personalized_models[test_idx]
                else:
                    p_model = None
                _, test_acc = test(test_idx, models[test_idx], p_model, extra_modules, test_loader, loss_fun, device, args)
                if args.mode == 'fed-co2':
                    print(' {:<11s}| Test  Acc: {:.4f} | G  Acc: {:.4f} | P  Acc: {:.4f}'.format(datasets[test_idx], test_acc[0], test_acc[1], test_acc[2]))
                else:
                    print(' {:<11s}| Test  Acc: {:.4f}'.format(datasets[test_idx], test_acc))       
        exit(0)

    # log the best for each model on all datasets
    best_epoch = 0
    best_acc = [0. for j in range(client_num)] 
    start_iter = 0

    # Start training
    for a_iter in range(start_iter, args.iters):
        optimizers = [optim.SGD(params=models[idx].parameters(), lr=args.lr, momentum=args.momentum) for idx in range(client_num)]
        for wi in range(args.wk_iters):
            print("============ Train epoch {} ============".format(wi + a_iter * args.wk_iters))
            if args.log:
                logfile.write("============ Train epoch {} ============\n".format(wi + a_iter * args.wk_iters)) 

            train_loss, train_acc = local_training(models, personalized_models, paggregation_models, server_model, extra_modules, 
                                                    args, train_loaders, optimizers, loss_fun, device, a_iter=a_iter)
        if args.mode == 'moon':
            paggregation_models = copy.deepcopy(models)
        elif args.mode == 'copa':
            extra_modules = {}
            for client_idx in range(client_num):
                extra_modules[client_idx] = copy.deepcopy(models[client_idx].head)
        with torch.no_grad():
            # Aggregation
            server_model, models = communication(args, server_model, models, personalized_models, extra_modules, paggregation_models, client_weights)

        with torch.no_grad():     
            # Report loss after aggregation
            for client_idx, model in enumerate(models):
                if args.mode in ['fed-co2', 'fedper', 'fedrod']:
                    p_model = personalized_models[client_idx]
                else:
                    p_model = None
                
                train_loss, train_acc = test(client_idx, model, p_model, extra_modules, train_loaders[client_idx], loss_fun, device, args)
                if args.mode == 'fed-co2':
                    print(' Site-{:<10s}| Train Loss: {:.4f} | G_branch Loss: {:.4f} | P_branch Loss: {:.4f} | Train Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f}'.format(datasets[client_idx] ,train_loss[0], train_loss[1], train_loss[2], train_acc[0], train_acc[1], train_acc[2]))
                else:
                    print(' Site-{:<10s}| Train Loss: {:.4f} | Train Acc: {:.4f}'.format(datasets[client_idx] ,train_loss, train_acc))
                if args.log:
                    if args.mode == 'fed-co2':
                        logfile.write(' Site-{:<10s}| Train Loss: {:.4f} | G_branch Loss: {:.4f} | P_branch Loss: {:.4f} | Train Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f}'.format(datasets[client_idx] ,train_loss[0], train_loss[1], train_loss[2], train_acc[0], train_acc[1], train_acc[2]))
                    else:
                        logfile.write(' Site-{:<10s}| Train Loss: {:.4f} | Train Acc: {:.4f}\n'.format(datasets[client_idx] ,train_loss, train_acc))

            if args.dataset != 'digits' and args.imbalance_train == False:
                # Validation
                val_acc_list = [None for j in range(client_num)]
                for client_idx, model in enumerate(models):
                    if args.mode in ['fed-co2', 'fedper', 'fedrod']:
                        p_model = personalized_models[client_idx]
                    else:
                        p_model = None

                    val_loss, val_acc = test(client_idx, model, p_model, extra_modules, val_loaders[client_idx], loss_fun, device, args)
                    if args.mode == 'fed-co2':
                        val_acc_list[client_idx] = val_acc[0]  
                        print(' Site-{:<10s}| Val Loss: {:.4f} | G_branch Loss: {:.4f} | P_branch Loss: {:.4f} | Val Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f}'.format(datasets[client_idx] ,val_loss[0], val_loss[1], val_loss[2], val_acc[0], val_acc[1], val_acc[2]))
                    else:
                        val_acc_list[client_idx] = val_acc
                        print(' Site-{:<10s}| Val  Loss: {:.4f} | Val  Acc: {:.4f}'.format(datasets[client_idx], val_loss, val_acc))
                    if args.log:
                        if args.mode == 'fed-co2':
                            logfile.write(' Site-{:<10s}| Val Loss: {:.4f} | G_branch Loss: {:.4f} | P_branch Loss: {:.4f} | Val Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f}'.format(datasets[client_idx] ,val_loss[0], val_loss[1], val_loss[2], val_acc[0], val_acc[1], val_acc[2]))
                        else:
                            logfile.write(' Site-{:<10s}| Val  Loss: {:.4f} | Val  Acc: {:.4f}\n'.format(datasets[client_idx], val_loss, val_acc))

                # Record best
                if np.mean(val_acc_list) > np.mean(best_acc):
                    for client_idx in range(client_num):
                        best_acc[client_idx] = val_acc_list[client_idx]
                        best_epoch = a_iter
                        best_changed=True
                        print(' Best site-{:<10s}| Epoch:{} | Val Acc: {:.4f}'.format(datasets[client_idx], best_epoch, best_acc[client_idx]))
                        if args.log:
                            logfile.write(' Best site-{:<10s} | Epoch:{} | Val Acc: {:.4f}\n'.format(datasets[client_idx], best_epoch, best_acc[client_idx]))
            
                if best_changed:     
                    print(' Saving the local and server checkpoint to {}...'.format(SAVE_PATH))
                    logfile.write(' Saving the local and server checkpoint to {}...\n'.format(SAVE_PATH))
                    if args.mode in ['fedbn', 'local', 'copa']:
                        if args.save_model:
                            if args.dataset == 'domainnet':
                                torch.save({
                                    'model_0': models[0].state_dict(),
                                    'model_1': models[1].state_dict(),
                                    'model_2': models[2].state_dict(),
                                    'model_3': models[3].state_dict(),
                                    'model_4': models[4].state_dict(),
                                    'model_5': models[5].state_dict(),
                                    'server_model': server_model.state_dict(),
                                    'best_epoch': best_epoch,
                                    'best_acc': best_acc,
                                    'a_iter': a_iter
                                }, SAVE_PATH)
                            elif args.dataset == 'office':
                                torch.save({
                                    'model_0': models[0].state_dict(),
                                    'model_1': models[1].state_dict(),
                                    'model_2': models[2].state_dict(),
                                    'model_3': models[3].state_dict(),
                                    'server_model': server_model.state_dict(),
                                    'best_epoch': best_epoch,
                                    'best_acc': best_acc,
                                    'a_iter': a_iter
                                }, SAVE_PATH)
                        best_changed = False
                        for client_idx, datasite in enumerate(datasets):
                            _, test_acc = test(client_idx, models[client_idx], None, extra_modules, test_loaders[client_idx], loss_fun, device, args)
                            print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, best_epoch, test_acc))
                            if args.log:
                                logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, best_epoch, test_acc))
                    
                    elif args.mode == 'fed-co2':
                        if args.save_model:
                            if args.dataset == 'domainnet':
                                torch.save({
                                    'model_0':  models[0].state_dict(),
                                    'model_1':  models[1].state_dict(),
                                    'model_2':  models[2].state_dict(),
                                    'model_3':  models[3].state_dict(),
                                    'model_4':  models[4].state_dict(),
                                    'model_5':  models[5].state_dict(),
                                    'pmodel_0': personalized_models[0].state_dict(),
                                    'pmodel_1': personalized_models[1].state_dict(),
                                    'pmodel_2': personalized_models[2].state_dict(),
                                    'pmodel_3': personalized_models[3].state_dict(),
                                    'pmodel_4': personalized_models[4].state_dict(),
                                    'pmodel_5': personalized_models[5].state_dict(),
                                    'best_epoch': best_epoch,
                                    'best_acc': best_acc,
                                    'a_iter': a_iter
                                }, SAVE_PATH)
                            elif args.dataset == 'office':
                                torch.save({
                                    'model_0':  models[0].state_dict(),
                                    'model_1':  models[1].state_dict(),
                                    'model_2':  models[2].state_dict(),
                                    'model_3':  models[3].state_dict(),
                                    'pmodel_0': personalized_models[0].state_dict(),
                                    'pmodel_1': personalized_models[1].state_dict(),
                                    'pmodel_2': personalized_models[2].state_dict(),
                                    'pmodel_3': personalized_models[3].state_dict(),
                                    'best_epoch': best_epoch,
                                    'best_acc': best_acc,
                                    'a_iter': a_iter
                                }, SAVE_PATH)
                        best_changed = False
                        for client_idx, datasite in enumerate(datasets):
        
                            _, test_acc = test(client_idx, models[client_idx], personalized_models[client_idx], extra_modules, test_loaders[client_idx], loss_fun, device, args)
                            print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f}'.format(datasite, best_epoch, test_acc[0], test_acc[1], test_acc[2]))
                            if args.log:
                                logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f}'.format(datasite, best_epoch, test_acc[0], test_acc[1], test_acc[2]))
                                
                    elif args.mode in ['fedper', 'fedrod']:
                        if args.save_model:
                            if args.dataset == 'domainnet':
                                torch.save({
                                    'server_model': server_model.state_dict(),
                                    'model_0': personalized_models[0].state_dict(),
                                    'model_1': personalized_models[1].state_dict(),
                                    'model_2': personalized_models[2].state_dict(),
                                    'model_3': personalized_models[3].state_dict(),
                                    'model_4': personalized_models[4].state_dict(),
                                    'model_5': personalized_models[5].state_dict(),
                                    'best_epoch': best_epoch,
                                    'best_acc': best_acc,
                                    'a_iter': a_iter
                                }, SAVE_PATH)
                            elif args.dataset == 'office':
                                torch.save({
                                    'server_model': server_model.state_dict(),
                                    'model_0': personalized_models[0].state_dict(),
                                    'model_1': personalized_models[1].state_dict(),
                                    'model_2': personalized_models[2].state_dict(),
                                    'model_3': personalized_models[3].state_dict(),
                                    'best_epoch': best_epoch,
                                    'best_acc': best_acc,
                                    'a_iter': a_iter
                                }, SAVE_PATH)
                        best_changed = False
                        for client_idx, datasite in enumerate(datasets):
        
                            _, test_acc = test(client_idx, models[client_idx], personalized_models[client_idx], extra_modules, test_loaders[client_idx], loss_fun, device, args)
                            print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, best_epoch, test_acc))
                            if args.log:
                                logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, best_epoch, test_acc))

                    else:
                        if args.save_model:
                            torch.save({
                                'server_model': server_model.state_dict(),
                                'best_epoch': best_epoch,
                                'best_acc': best_acc,
                                'a_iter': a_iter
                            }, SAVE_PATH)
                        best_changed = False
                        for client_idx, datasite in enumerate(datasets):
                            _, test_acc = test(client_idx, server_model, None, None, test_loaders[client_idx], loss_fun, device, args)
                            print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, best_epoch, test_acc))
                            if args.log:
                                logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, best_epoch, test_acc))

            else:
                if a_iter<=15 or (a_iter+1)%50==0:
                    if args.mode in ['fedbn', 'local', 'copa']:
                        for client_idx, datasite in enumerate(datasets):
                            _, test_acc = test(client_idx, models[client_idx], None, extra_modules, test_loaders[client_idx], loss_fun, device, args)
                            print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, a_iter, test_acc))
                            if args.log:
                                logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, a_iter, test_acc))
                    
                    elif args.mode.lower() == 'fed-co2':
    
                        for client_idx, datasite in enumerate(datasets):
                            _, test_acc = test(client_idx, models[client_idx], personalized_models[client_idx], extra_modules, test_loaders[client_idx], loss_fun, device, args)
                            print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f}'.format(datasite, a_iter, test_acc[0], test_acc[1], test_acc[2])) 
                            if args.log:
                                logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f}'.format(datasite, a_iter, test_acc[0], test_acc[1], test_acc[2]))
                                
                    elif args.mode.lower() in ['fedper', 'fedrod']:
                        for client_idx, datasite in enumerate(datasets):
        
                            _, test_acc = test(client_idx, models[client_idx], personalized_models[client_idx], extra_modules, test_loaders[client_idx], loss_fun, device, args)
                            print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, a_iter, test_acc))
                            if args.log:
                                logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, a_iter, test_acc))
                    else:
                        for client_idx, datasite in enumerate(datasets):
                            _, test_acc = test(client_idx, server_model, None, None, test_loaders[client_idx], loss_fun, device, args)
                            print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, a_iter, test_acc))
                            if args.log:
                                logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, a_iter, test_acc))

            if a_iter == args.iters-1:
                if args.dataset != 'digits' and args.imbalance_train == False:
                    if args.mode in ['fedbn', 'local', 'copa']:
                        for client_idx, datasite in enumerate(datasets):
                            _, test_acc = test(client_idx, models[client_idx], None, extra_modules, test_loaders[client_idx], loss_fun, device, args)
                            print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, best_epoch, test_acc))
                            if args.log:
                                logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, best_epoch, test_acc))
                    
                    elif args.mode =='fed-co2':
                        for client_idx, datasite in enumerate(datasets):
                            _, test_acc = test(client_idx, models[client_idx], personalized_models[client_idx], extra_modules, test_loaders[client_idx], loss_fun, device, args)
                            print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f}'.format(datasite, best_epoch, test_acc[0], test_acc[1], test_acc[2]))
                            if args.log:
                                logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f}'.format(datasite, best_epoch, test_acc[0], test_acc[1], test_acc[2]))
                                
                    elif args.mode in ['fedper', 'fedrod']:
                        for client_idx, datasite in enumerate(datasets):       
                            _, test_acc = test(client_idx, models[client_idx], personalized_models[client_idx], extra_modules, test_loaders[client_idx], loss_fun, device, args)
                            print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, best_epoch, test_acc))
                            if args.log:
                                logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, best_epoch, test_acc))
                    else:
                        for client_idx, datasite in enumerate(datasets):
                            _, test_acc = test(client_idx, server_model, None, None, test_loaders[client_idx], loss_fun, device, args)
                            print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, best_epoch, test_acc))
                            if args.log:
                                logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, best_epoch, test_acc))

                else:
                    print(' Saving checkpoints to {}...'.format(SAVE_PATH))
                    logfile.write(' Saving checkpoints to {}...\n'.format(SAVE_PATH))                 
                    if args.mode in ['fedbn', 'local', 'copa']:
                        if args.save_model:             
                            torch.save({
                                'model_0': models[0].state_dict(),
                                'model_1': models[1].state_dict(),
                                'model_2': models[2].state_dict(),
                                'model_3': models[3].state_dict(),
                                'model_4': models[4].state_dict(),
                                'server_model': server_model.state_dict(),
                                'best_epoch': best_epoch,
                                'best_acc': best_acc,
                                'a_iter': a_iter
                            }, SAVE_PATH)
                    elif args.mode == 'fed-co2':
                        if args.save_model:
                            torch.save({
                                'model_0': models[0].state_dict(),
                                'model_1': models[1].state_dict(),
                                'model_2': models[2].state_dict(),
                                'model_3': models[3].state_dict(),
                                'model_4': models[4].state_dict(),
                                'pmodel_0': personalized_models[0].state_dict(),
                                'pmodel_1': personalized_models[1].state_dict(),
                                'pmodel_2': personalized_models[2].state_dict(),
                                'pmodel_3': personalized_models[3].state_dict(),
                                'pmodel_4': personalized_models[4].state_dict(),
                                'best_epoch': best_epoch,
                                'best_acc': best_acc,
                                'a_iter': a_iter
                                }, SAVE_PATH)
                    elif args.mode in ['fedper', 'fedrod']:
                        if args.save_model:
                            torch.save({
                                'server_model': server_model.state_dict(),
                                'model_0': personalized_models[0].state_dict(),
                                'model_1': personalized_models[1].state_dict(),
                                'model_2': personalized_models[2].state_dict(),
                                'model_3': personalized_models[3].state_dict(),
                                'model_4': personalized_models[4].state_dict(),
                                'best_epoch': best_epoch,
                                'best_acc': best_acc,
                                'a_iter': a_iter
                            }, SAVE_PATH)
                    else:
                        if args.save_model:
                            torch.save({
                                'server_model': server_model.state_dict(),
                                'best_epoch': best_epoch,
                                'best_acc': best_acc,
                                'a_iter': a_iter
                            }, SAVE_PATH)


        if log:
            logfile.flush()
    

    if log:
        logfile.flush()
        logfile.close()
