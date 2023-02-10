"""
federated learning with different aggregation strategy on domainnet dataset
"""
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import pickle as pkl
from utils.data_utils import DomainNetDataset, prepare_data
from utils.methods import local_training
from utils.utils import  communication, test
from nets.models import AlexNet, AlexNet_rod, AlexNet_peer, P_Head
from nets.vit import ViT, ViTHyper
import argparse
import time
import copy
import torchvision.transforms as transforms
import random

     
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed= 1
    np.random.seed(seed)
    torch.manual_seed(seed)    
    torch.cuda.manual_seed_all(seed) 
    random.seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true', help='whether to log')
    parser.add_argument('--test', action='store_true', help ='test the pretrained model')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0, help='momentum')
    parser.add_argument('--batch', type = int, default= 32, help ='batch size')
    parser.add_argument('--iters', type = int, default=300, help = 'iterations for communication')
    parser.add_argument('--wk_iters', type = int, default=1, help = 'optimization iters in local worker between communication')
    parser.add_argument('--mode', type = str, default='fedbn', help='[FedBN | FedAvg | FedProx | fedper | fedrod | fedtp | peer]')
    parser.add_argument('--mu', type=float, default=1e-3, help='The hyper parameter for fedprox')
    parser.add_argument('--save_path', type = str, default='../checkpoint/domainnet', help='path to save the checkpoint')
    parser.add_argument('--resume', action='store_true', help ='resume training from the save path checkpoint')
    parser.add_argument('--dataset', type=str, default='domainnet')
    parser.add_argument('--version', type=int, default=1)
    parser.add_argument('--backbone', type=str, default='alexnet')
    args = parser.parse_args()

    if args.dataset == 'domainnet':
        exp_folder = 'fed_domainnet'
    elif args.dataset == 'office_home':
        exp_folder = 'fed_office'

    args.save_path = os.path.join(args.save_path, exp_folder)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path, '{}{}'.format(args.mode, args.version))

    log = args.log

    if log:
        if args.dataset == 'domainnet':
            log_path = os.path.join('logs/domainnet', exp_folder)
        elif args.dataset == 'office_home':
            log_path = os.path.join('logs/office/', exp_folder)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logfile = open(os.path.join(log_path,'{}_{}{}.log'.format(args.backbone, args.mode, args.version)), 'a')
        # logfile = open(os.path.join(log_path,'{}.log'.format(args.mode)), 'a')
        logfile.write('==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        logfile.write('===Setting===\n')
        logfile.write("Backnone: %s\n" %args.backbone)
        logfile.write('    lr: {}\n'.format(args.lr))
        logfile.write('    batch: {}\n'.format(args.batch))
        logfile.write('    iters: {}\n'.format(args.iters))
    
    train_loaders, val_loaders, test_loaders = prepare_data(args)

    
    # setup model
    if args.mode in ['fedper' ,'fedrod']:
        server_model = AlexNet_rod().to(device)
    elif args.mode == 'fedtp':
        server_model = ViT(image_size = 256, patch_size = 16, num_classes = 10, dim = 768, depth = 6, heads = 8, mlp_dim = 3072,
                  dropout = 0.1, emb_dropout = 0.1).to(device)
    elif args.mode == 'peer':
        server_model = AlexNet_peer().to(device)
    else:
        if args.backbone == 'vit':
            server_model = ViT(image_size = 256, patch_size = 16, num_classes = 10, dim = 768, depth = 6, heads = 8, mlp_dim = 3072,
                  dropout = 0.1, emb_dropout = 0.1).to(device)
        else:
            server_model = AlexNet().to(device)
    loss_fun = nn.CrossEntropyLoss()

    # name of each datasets
    if args.dataset == 'domainnet':
        datasets = ['Clipart', 'Infograph', 'Painting', 'Quickdraw', 'Real', 'Sketch']
    elif args.dataset == 'office_home':
        datasets = ['Amazon', 'Caltech', 'DSLR', 'Webcam']
    # federated client number
    client_num = len(datasets)
    client_weights = [1/client_num for i in range(client_num)]
    
    # each local client model
    models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]
    paggregation_models = []
    personalized_models = []
    hnet = None
    global_prototypes = []
    if args.mode == 'peer':
        paggregation_models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]
        AlexNet_peer().to(device)
        if args.version == 1:
            print("Version1: Train two models seperately. Only one engages in global aggregation.")
            personalized_models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]
        
        elif args.version == 2:
            print("Version2: Train two models, domain-invariant one uses peer mechanism.")
            personalized_models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]

        elif args.version == 3:
            print("Version3: Train two models, add metric to maximize feature distance and minimize logits distance")
            personalized_models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]
        elif args.version == 6:
            print("Version6: Train two models, add metric to maximize feature distance and minimize logit matrix with identity matrix")
            personalized_models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]
        elif args.version == 7:
            print("Verson7:  Initialize private models personally, use e^-x to maximize feature distance and minimize logit matrix with identity matrix")
            # for key, para in server_model.state_dict().items():
            #     print(key)
            personalized_models = [AlexNet_peer().to(device) for idx in range(client_num)]
            # print("server model:  ")
            # print(server_model.state_dict()["features.conv1.bias"])
            # print(server_model.state_dict()["head.bias"])
            # for idx in range(client_num):
            #     print("personalized_model:  ", idx)
            #     print(personalized_models[idx].state_dict()["features.conv1.bias"])
            #     print(personalized_models[idx].state_dict()["head.bias"])
            # assert False
        elif args.version == 8:
            print("Version8:  Version 7 + increase CD weight with iter goes up")
            personalized_models = [AlexNet_peer().to(device) for idx in range(client_num)]
        elif args.version == 9:
            print("Version9: Version 6 + Initialize + maximize prototype distance instead of feature distance")
            personalized_models = [AlexNet_peer().to(device) for idx in range(client_num)]

        elif args.version == 4:
            print("Version4: Train two models, add metric and peer mechanism")
            personalized_models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]
        
        elif args.version == 5:
            print("Version5: Train two models, add metric and add global prototype calibration")
            personalized_models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]

    elif args.mode in ['fedper', 'fedrod']:
        priviate_head = P_Head().to(device)
        personalized_models = [copy.deepcopy(priviate_head).to(device) for idx in range(client_num)]
    elif args.mode == 'fedtp':
        hnet = ViTHyper(client_num, 128, hidden_dim = 256, dim=768, 
        heads = 8, dim_head = 64, n_hidden = 3, depth=6, client_sample=client_num).to(device)

    best_changed = False

    if args.test:
        print('Loading snapshots...')
        checkpoint = torch.load('/public/home/caizhy/work/Peer/checkpoint/domainnet/fed_domainnet/{}'.format(args.mode.lower()))
        server_model.load_state_dict(checkpoint['server_model'])
        if args.mode.lower() in ['fedbn', 'local']:
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
        elif args.mode.lower() in ['fedper', 'fedrod']:
            for client_idx in range(client_num):
                personalized_models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
                models[client_idx].load_state_dict(checkpoint['server_model'])
        elif args.mode.lower() == 'peer':
            for client_idx in range(client_num):
                personalized_models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
                if args.version in [1, 3, 5, 6, 7, 8, 9]:
                    models[client_idx].load_state_dict(checkpoint['server_model'])
                elif args.version in [2, 4]:
                    models[client_idx].load_state_dict(checkpoint['server_model_{}'.format(client_idx)])
        elif args.mode.lower() == 'fedtp':
            hnet.load_state_dict(checkpoint['hnet'])
        else:
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['server_model'])
        for test_idx, test_loader in enumerate(test_loaders):
            if args.mode in ['peer', 'fedper', 'fedrod']:
                p_model = personalized_models[test_idx]
            else:
                p_model = None
            _, test_acc = test(test_idx, models[test_idx], p_model, test_loader, loss_fun, device, args, hnet)
            print(' {:<11s}| Test  Acc: {:.4f}'.format(datasets[test_idx], test_acc))
        
        exit(0)

    if args.resume:
        checkpoint = torch.load(SAVE_PATH)
        server_model.load_state_dict(checkpoint['server_model'])
        if args.mode.lower() == 'fedbn':
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
        else:
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['server_model'])
        best_epoch, best_acc  = checkpoint['best_epoch'], checkpoint['best_acc']
        start_iter = int(checkpoint['a_iter']) + 1

        print('Resume training from epoch {}'.format(start_iter))
    else:
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

            train_loss, train_acc, proto_dict = local_training(models, personalized_models, hnet, server_model, global_prototypes, args, train_loaders, optimizers, loss_fun, device, a_iter=a_iter)  
        
        with torch.no_grad():
            # Aggregation
            server_model, models, global_prototypes = communication(args, server_model, models, personalized_models, paggregation_models, client_weights, proto_dict)

            # Report loss after aggregation
            for client_idx, model in enumerate(models):
                if args.mode in ['peer', 'fedper', 'fedrod']:
                    p_model = personalized_models[client_idx]
                else:
                    p_model = None
                train_loss, train_acc = test(client_idx, model, p_model, train_loaders[client_idx], loss_fun, device, args, hnet)
                print(' Site-{:<10s}| Train Loss: {:.4f} | Train Acc: {:.4f}'.format(datasets[client_idx] ,train_loss, train_acc))
                if args.log:
                    logfile.write(' Site-{:<10s}| Train Loss: {:.4f} | Train Acc: {:.4f}\n'.format(datasets[client_idx] ,train_loss, train_acc))

            # Validation
            val_acc_list = [None for j in range(client_num)]
            for client_idx, model in enumerate(models):
                if args.mode in ['peer', 'fedper', 'fedrod']:
                    p_model = personalized_models[client_idx]
                else:
                    p_model = None
                val_loss, val_acc = test(client_idx, model, p_model, val_loaders[client_idx], loss_fun, device, args, hnet)
                val_acc_list[client_idx] = val_acc
                print(' Site-{:<10s}| Val  Loss: {:.4f} | Val  Acc: {:.4f}'.format(datasets[client_idx], val_loss, val_acc))
                if args.log:
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
                if args.mode.lower() in ['fedbn', 'local']:
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
                    elif args.dataset == 'office_home':
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
                        _, test_acc = test(client_idx, models[client_idx], None, test_loaders[client_idx], loss_fun, device, args, None)
                        print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, best_epoch, test_acc))
                        if args.log:
                            logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, best_epoch, test_acc))
                
                elif args.mode.lower() in ['peer', 'fedper', 'fedrod']:
                    if args.mode.lower() == 'peer' and args.version in [1, 3, 5, 6, 7, 8, 9]:
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
                        elif args.dataset == 'office_home':
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
                    else:
                        if args.dataset == 'domainnet':
                            torch.save({
                                'server_model': server_model.state_dict(),
                                'server_model_0': models[0].state_dict(),
                                'server_model_1': models[1].state_dict(),
                                'server_model_2': models[2].state_dict(),
                                'server_model_3': models[3].state_dict(),
                                'server_model_4': models[4].state_dict(),
                                'server_model_5': models[5].state_dict(),
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
                        elif args.dataset == 'office_home':
                            torch.save({
                                'server_model': server_model.state_dict(),
                                'server_model_0': models[0].state_dict(),
                                'server_model_1': models[1].state_dict(),
                                'server_model_2': models[2].state_dict(),
                                'server_model_3': models[3].state_dict(),
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
                        _, test_acc = test(client_idx, server_model, personalized_models[client_idx], test_loaders[client_idx], loss_fun, device, args, None)
                        print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, best_epoch, test_acc))
                        if args.log:
                            logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, best_epoch, test_acc))

                elif args.mode.lower() == 'fedtp':
                    torch.save({
                        'server_model': server_model.state_dict(),
                        'hnet': hnet.state_dict(), 
                        'best_epoch': best_epoch,
                        'best_acc': best_acc,
                        'a_iter': a_iter
                    }, SAVE_PATH)
                    best_changed = False
                    for client_idx, datasite in enumerate(datasets):
                        _, test_acc = test(client_idx, server_model, None, test_loaders[client_idx], loss_fun, device, args, hnet)
                        print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, best_epoch, test_acc))
                        if args.log:
                            logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, best_epoch, test_acc))

                else:
                    torch.save({
                        'server_model': server_model.state_dict(),
                        'best_epoch': best_epoch,
                        'best_acc': best_acc,
                        'a_iter': a_iter
                    }, SAVE_PATH)
                    best_changed = False
                    for client_idx, datasite in enumerate(datasets):
                        _, test_acc = test(client_idx, server_model, None, test_loaders[client_idx], loss_fun, device, args, None)
                        print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, best_epoch, test_acc))
                        if args.log:
                            logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, best_epoch, test_acc))

        if log:
            logfile.flush()
    if log:
        logfile.flush()
        logfile.close()
