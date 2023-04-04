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
from utils.data_utils import prepare_data_domainNet, prepare_data_officeHome, prepare_data_digits
from utils.methods import local_training
from utils.utils import  communication, test, set_client_weight, visualize, log_write_dictionary, show_dictionary, visualize_combination, adapt_lambda
from utils.func_v import definite_version
from nets.models import AlexNet, AlexNet_rod, AlexNet_peer, P_Head, DigitModel, DigitModel_rod, D_Head
from nets.vit import ViT, ViTHyper
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
    parser.add_argument('--mode', type = str, default='fedbn', help='[FedBN | FedAvg | FedProx | fedper | fedrod | fedtp | peer | AlignFed | COPA]')
    parser.add_argument('--mu', type=float, default=1e-3, help='The hyper parameter for fedprox')
    parser.add_argument('--save_path', type = str, default='../checkpoint/domainnet', help='path to save the checkpoint')
    parser.add_argument('--resume', action='store_true', help ='resume training from the save path checkpoint')
    parser.add_argument('--dataset', type=str, default='domainnet')
    parser.add_argument('--version', type=int, default=1)
    parser.add_argument('--backbone', type=str, default='alexnet')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--reg_value', type=float, default=10)

    args = parser.parse_args()

    if args.dataset == 'domainnet':
        exp_folder = 'fed_domainnet'
        train_loaders, val_loaders, test_loaders = prepare_data_domainNet(args)
    elif args.dataset == 'office_home':
        exp_folder = 'fed_office'
        train_loaders, val_loaders, test_loaders = prepare_data_officeHome(args)
    elif args.dataset == 'digits':
        exp_folder = 'fed_digits'
        train_loaders, test_loaders = prepare_data_digits(args)
        val_loaders = None

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
        elif args.dataset == 'digits':
            log_path = os.path.join('logs/digits/', exp_folder)
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


    # setup model
    if args.mode in ['fedper' ,'fedrod']:
        if args.dataset == 'digits':
            server_model = DigitModel_rod().to(device)
        else:
            server_model = AlexNet_rod().to(device)
    elif args.mode == 'fedtp':
        server_model = ViT(image_size = 256, patch_size = 16, num_classes = 10, dim = 768, depth = 6, heads = 8, mlp_dim = 3072,
                  dropout = 0.1, emb_dropout = 0.1).to(device)
    elif args.mode == 'peer':
        if args.dataset == 'digits':
            server_model = DigitModel().to(device)
        else:
            if args.version in [18, 25, 63, 70, 76, 81, 37, 56, 57,
                                66, 67, 68, 69, 71, 73, 78,
                                79, 80, 82, 83, 88, 89, 90]:
                server_model = AlexNet().to(device)
            else:
                server_model = AlexNet_peer().to(device)
    elif args.mode == 'AlignFed':
        server_model = AlexNet_peer().to(device)
        print("AlignFed Version: ", args.version)
    elif args.mode == 'COPA':
        server_model = AlexNet_peer().to(device)
        print(" COPA ")
    else:
        if args.backbone == 'vit':
            server_model = ViT(image_size = 256, patch_size = 16, num_classes = 10, dim = 768, depth = 6, heads = 8, mlp_dim = 3072,
                  dropout = 0.1, emb_dropout = 0.1).to(device)
        else:
            if args.dataset == 'digits':
                server_model = DigitModel().to(device)
            else:
                server_model = AlexNet().to(device)
    loss_fun = nn.CrossEntropyLoss()

    # for n, p in server_model.state_dict().items():
    #     print(n,": ",p.shape)
    # assert False 

    # name of each datasets
    if args.dataset == 'domainnet':
        datasets = ['Clipart', 'Infograph', 'Painting', 'Quickdraw', 'Real', 'Sketch']
    elif args.dataset == 'office_home':
        datasets = ['Amazon', 'Caltech', 'DSLR', 'Webcam']
    elif args.dataset == 'digits':
        datasets = ['MNIST', 'SVHN', 'USPS', 'SynthDigits', 'MNIST-M']

    # federated client number
    client_num = len(datasets)
    client_weights = [1/client_num for i in range(client_num)]
    
    # each local client model
    models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]
    paggregation_models = []
    personalized_models = []
    extra_modules = []
    hnet = None
    global_prototypes = []
    global_prototype = None
    valid_onehot = {x:0  for x in range(client_num)}
    weight_dict = {x:[1/client_num for i in range(client_num)]  for x in range(client_num)}
    

    if args.mode == 'peer':
        if args.version == 17:
            print("Version17: Train two models separately")

        elif args.version == 18:
            print("Version18: Global branch generalize, train two branches separately")

        elif args.version == 25:
            print("Version25: P branch generalizes as well")

        elif args.version == 69:
            print("Version69: P branch generalize only, train two branches separately")

        elif args.version == 28:
            print("Version28: P&G branch generalize on pretrained heads")

        elif args.version == 63:
            print("Version63: quick experiment, G branch now is based on FedBN")

        elif args.version == 70:
            print("Version70: Use validation set to learn lambda")

        elif args.version == 76:
            print("Version76: V76 uses validation set to learn lambda and G branch is now fedBN")

        elif args.version == 81:
            print("Version81: V81 validation set to learn lambda, G branch is fedBN, P branch is generalized")

        elif args.version == 37:
            print("Version37: KD before local training, 2 phases")

        elif args.version == 56:
            print("Version56: P&G new KD, P&G generalize, personal running BNs")

        elif args.version == 66:
            print("Version66: two branches are trained independently, G&P new KD")

        elif args.version == 67:
            print("Version67: two branches are trained independently, G new KD")

        elif args.version == 68:
            print("Version68: two branches are trained independently, P new KD")

        elif args.version == 71:
            print("Version71: two branches are trained independently, personalize BNs")

        elif args.version == 73:
            print("Version73: two branches are trained independently, learn lambda")

        elif args.version == 78:
            print("Version78: two branches are trained independently, P&G new KD, personal BNs")

        elif args.version == 79:
            print("Version79: two branches are trained independently, G new KD, personal BNs")

        elif args.version == 80:
            print("Version80: two branches are trained independently, P new KD, personal BNs")

        elif args.version == 82:
            print("Version82: two branches are trained independently, P generalizes, personal BNs")

        elif args.version == 83:
            print("Version83: two branches are trained independently, P&G generalize, personal BNs")

        elif args.version == 88:
            print("Version88: G new KD, P&G generalize, personal BNs")

        elif args.version == 89:
            print("Version89: P new KD, P&G generalize, personal BNs")

        elif args.version == 90:
            print("Version90: P&G new KD, P&G generalize, personal BNs, learn λ")

        else:
            definite_version(args.version)


        if args.version in [70, 76, 81, 42, 47, 57, 73, 90]:
            assert args.dataset != 'digits'
            assert args.mode == 'peer'
            meb = nn.Embedding(num_embeddings=1, embedding_dim=1).to(device)
            # print(meb.state_dict())
            meb.state_dict()['weight'].data.copy_(torch.zeros([1], dtype=torch.long))
            # print(meb.state_dict())
            extra_modules = [copy.deepcopy(meb) for idx in range(client_num)]
            personalized_models = [AlexNet().to(device) for idx in range(client_num)]
        else:
            personalized_models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]


    elif args.mode in ['fedper', 'fedrod']:
        if args.dataset == 'digits':
            priviate_head = D_Head().to(device)
        else:
            priviate_head = P_Head().to(device)
        personalized_models = [copy.deepcopy(priviate_head).to(device) for idx in range(client_num)]


    elif args.mode == 'fedtp':
        hnet = ViTHyper(client_num, 128, hidden_dim = 256, dim=768, 
        heads = 8, dim_head = 64, n_hidden = 3, depth=6, client_sample=client_num).to(device)

    best_changed = False

    if args.test:
        print('Loading snapshots...')
        checkpoint = torch.load('/public/home/caizhy/work/Peer/checkpoint/domainnet/fed_domainnet/{}{}'.format(args.mode.lower(),args.version))
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
                models[client_idx].load_state_dict(checkpoint['server_model'])
        elif args.mode.lower() == 'fedtp':
            hnet.load_state_dict(checkpoint['hnet'])
        else:
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['server_model'])

        with torch.no_grad():
            for test_idx, test_loader in enumerate(test_loaders):
                if args.mode in ['peer', 'fedper', 'fedrod']:
                    p_model = personalized_models[test_idx]
                else:
                    p_model = None
                a_model = None

                _, test_acc = test(test_idx, models[test_idx], p_model, a_model, extra_modules, test_loader, loss_fun, device, args, hnet, global_prototype)
                if args.version == 18:
                    print(' {:<11s}'.format(datasets[test_idx]))
                    # visualize_d(test_idx, models[test_idx], p_model, extra_modules, train_loaders[test_idx], loss_fun, device, args, hnet, global_prototype)
                    # visualize(test_idx, models[test_idx], p_model, extra_modules, train_loaders[test_idx], loss_fun, device, args, hnet, global_prototype)
                    # visualize_combination(test_idx, models[test_idx], personalized_models, test_loaders[test_idx], loss_fun, device, args)
                    adapt_lambda(test_idx, models[test_idx], personalized_models[test_idx], val_loaders[test_idx], test_loaders[test_idx], loss_fun, device, args)
                if args.mode == 'peer':
                    print(' {:<11s}| Test  Acc: {:.4f} | G  Acc: {:.4f} | P  Acc: {:.4f}'.format(datasets[test_idx], test_acc[0], test_acc[1], test_acc[2]))
                else:
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

            train_loss, train_acc, proto_dict, weight_dict = local_training(models, personalized_models, paggregation_models, hnet, server_model, global_prototypes, extra_modules, 
                                                               valid_onehot, weight_dict, args, train_loaders, test_loaders, optimizers, loss_fun, device, a_iter=a_iter)
        
        with torch.no_grad():
            # Aggregation
            # if len(weight_dict[0]) != 0:
            #     assert args.version in [39, 40, 41, 58, 59, 60]
            #     server_model, models, global_prototypes = communication(args, server_model, models, personalized_models, extra_modules, paggregation_models, weight_dict, proto_dict, a_iter)
            #     for ccidx in range(client_num):
            #         print("Dataset: ", datasets[ccidx], " weight list: ", weight_dict[ccidx])
            #     if args.log:
            #         for ccidx in range(client_num):
            #             wei_list = weight_dict[ccidx]
            #             if args.dataset == 'domainnet':
            #                 logfile.write(' Site-{:<10s}| w1: {:.4f} | w2: {:.4f} | w3: {:.4f} | w4: {:.4f} | w5: {:.4f} | w6: {:4f}\n'.format(datasets[ccidx], wei_list[0], wei_list[1], wei_list[2], wei_list[3], wei_list[4], wei_list[5]))
            #             else:
            #                 logfile.write(' Site-{:<10s}| w1: {:.4f} | w2: {:.4f} | w3: {:.4f} | w4: {:.4f}\n'.format(datasets[ccidx], wei_list[0], wei_list[1], wei_list[2], wei_list[3]))
            # else:
            server_model, models, global_prototypes = communication(args, server_model, models, personalized_models, extra_modules, paggregation_models, client_weights, proto_dict, a_iter)

        if args.version in [70, 76, 81, 57, 73, 90]:
            assert args.dataset != 'digits'
            assert args.mode == 'peer'
            train_loss, train_acc, _, _ = local_training(models, personalized_models, None, None, None, None, extra_modules, 
                                        None, None, args, None, val_loaders, None, loss_fun, device, a_iter=a_iter, phase='Valid')
            
            for ccidx in range(client_num):
                e_model = extra_modules[ccidx]
                e_model.eval()
                lambda_p = torch.sigmoid(e_model(torch.tensor([0], dtype=torch.long).to(device)))
                lambda_g = 1 - lambda_p
                print("Dataset: ", datasets[ccidx], " lambda_g: ", lambda_g[0].item(), " lambda_p: ", lambda_p[0].item())
                if args.log:
                    logfile.write(' Site-{:<10s}| lambda_g: {:.4f} | lambda_p: {:.4f}\n'.format(datasets[ccidx], lambda_g[0].item(), lambda_p[0].item()))


        with torch.no_grad():
                
            # Report loss after aggregation
            for client_idx, model in enumerate(models):
                if args.mode in ['peer', 'fedper', 'fedrod']:
                    p_model = personalized_models[client_idx]
                else:
                    p_model = None
                a_model = None
                
                train_loss, train_acc = test(client_idx, model, p_model, a_model, extra_modules, train_loaders[client_idx], loss_fun, device, args, hnet, global_prototype, flog=True)
                if args.mode == 'peer':
                    print(' Site-{:<10s}| Train Loss: {:.4f} | G_branch Loss: {:.4f} | P_branch Loss: {:.4f} | Train Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f}'.format(datasets[client_idx] ,train_loss[0], train_loss[1], train_loss[2], train_acc[0], train_acc[1], train_acc[2]))
                else:
                    print(' Site-{:<10s}| Train Loss: {:.4f} | Train Acc: {:.4f}'.format(datasets[client_idx] ,train_loss, train_acc))
                if args.log:
                    if args.mode == 'peer':
                        logfile.write(' Site-{:<10s}| Train Loss: {:.4f} | G_branch Loss: {:.4f} | P_branch Loss: {:.4f} | Train Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f}'.format(datasets[client_idx] ,train_loss[0], train_loss[1], train_loss[2], train_acc[0], train_acc[1], train_acc[2]))
                    else:
                        logfile.write(' Site-{:<10s}| Train Loss: {:.4f} | Train Acc: {:.4f}\n'.format(datasets[client_idx] ,train_loss, train_acc))

            if args.dataset != 'digits':
                # Validation
                val_acc_list = [None for j in range(client_num)]
                for client_idx, model in enumerate(models):
                    if args.mode in ['peer', 'fedper', 'fedrod']:
                        p_model = personalized_models[client_idx]
                    else:
                        p_model = None
                    a_model = None

                    val_loss, val_acc = test(client_idx, model, p_model, a_model, extra_modules, val_loaders[client_idx], loss_fun, device, args, hnet, global_prototype)
                    if args.mode == 'peer':

                        val_acc_list[client_idx] = val_acc[0]  
                        print(' Site-{:<10s}| Val Loss: {:.4f} | G_branch Loss: {:.4f} | P_branch Loss: {:.4f} | Val Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f}'.format(datasets[client_idx] ,val_loss[0], val_loss[1], val_loss[2], val_acc[0], val_acc[1], val_acc[2]))
                    else:
                        val_acc_list[client_idx] = val_acc
                        print(' Site-{:<10s}| Val  Loss: {:.4f} | Val  Acc: {:.4f}'.format(datasets[client_idx], val_loss, val_acc))
                    if args.log:
                        if args.mode == 'peer':
                            #G acc is higher than P acc
                            # if val_acc[1] > val_acc[2]:
                            #     valid_onehot[client_idx] = 1
                            # else:
                            #     valid_onehot[client_idx] = 0

                            logfile.write(' Site-{:<10s}| Val Loss: {:.4f} | G_branch Loss: {:.4f} | P_branch Loss: {:.4f} | Val Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f}'.format(datasets[client_idx] ,val_loss[0], val_loss[1], val_loss[2], val_acc[0], val_acc[1], val_acc[2]))
                        else:
                            logfile.write(' Site-{:<10s}| Val  Loss: {:.4f} | Val  Acc: {:.4f}\n'.format(datasets[client_idx], val_loss, val_acc))

                # if args.mode == 'peer' and args.version in [46, 47, 48, 49, 50, 34, 35]:
                #     print(valid_onehot)

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
                    if args.mode in ['fedbn', 'local', 'AlignFed', 'COPA']:
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
                            _, test_acc = test(client_idx, models[client_idx], None, None, None, test_loaders[client_idx], loss_fun, device, args, None, None)
                            print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, best_epoch, test_acc))
                            if args.log:
                                logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, best_epoch, test_acc))
                    
                    elif args.mode in ['peer', 'fedper', 'fedrod']:
                        if args.save_model:
                            if args.mode.lower() == 'peer':
                                if args.version in [63, 37]:
                                    if args.dataset == 'domainnet':
                                        torch.save({
                                            'model_0': models[0].state_dict(),
                                            'model_1': models[1].state_dict(),
                                            'model_2': models[2].state_dict(),
                                            'model_3': models[3].state_dict(),
                                            'model_4': models[4].state_dict(),
                                            'model_5': models[5].state_dict(),
                                            'p_model_0': personalized_models[0].state_dict(),
                                            'p_model_1': personalized_models[1].state_dict(),
                                            'p_model_2': personalized_models[2].state_dict(),
                                            'p_model_3': personalized_models[3].state_dict(),
                                            'p_model_4': personalized_models[4].state_dict(),
                                            'p_model_5': personalized_models[5].state_dict(),
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
                                            'p_model_0': personalized_models[0].state_dict(),
                                            'p_model_1': personalized_models[1].state_dict(),
                                            'p_model_2': personalized_models[2].state_dict(),
                                            'p_model_3': personalized_models[3].state_dict(),
                                            'best_epoch': best_epoch,
                                            'best_acc': best_acc,
                                            'a_iter': a_iter
                                        }, SAVE_PATH)
                                elif args.version == 70:
                                    if args.dataset == 'domainnet':
                                        torch.save({
                                            'server_model': server_model.state_dict(),
                                            'emodel_0': extra_modules[0].state_dict(),
                                            'emodel_1': extra_modules[1].state_dict(),
                                            'emodel_2': extra_modules[2].state_dict(),
                                            'emodel_3': extra_modules[3].state_dict(),
                                            'emodel_4': extra_modules[4].state_dict(),
                                            'emodel_5': extra_modules[5].state_dict(),
                                            'p_model_0': personalized_models[0].state_dict(),
                                            'p_model_1': personalized_models[1].state_dict(),
                                            'p_model_2': personalized_models[2].state_dict(),
                                            'p_model_3': personalized_models[3].state_dict(),
                                            'p_model_4': personalized_models[4].state_dict(),
                                            'p_model_5': personalized_models[5].state_dict(),
                                            'best_epoch': best_epoch,
                                            'best_acc': best_acc,
                                            'a_iter': a_iter
                                        }, SAVE_PATH)
                                    elif args.dataset == 'office_home':
                                        torch.save({
                                            'server_model': server_model.state_dict(),
                                            'emodel_0': extra_modules[0].state_dict(),
                                            'emodel_1': extra_modules[1].state_dict(),
                                            'emodel_2': extra_modules[2].state_dict(),
                                            'emodel_3': extra_modules[3].state_dict(),
                                            'p_model_0': personalized_models[0].state_dict(),
                                            'p_model_1': personalized_models[1].state_dict(),
                                            'p_model_2': personalized_models[2].state_dict(),
                                            'p_model_3': personalized_models[3].state_dict(),
                                            'best_epoch': best_epoch,
                                            'best_acc': best_acc,
                                            'a_iter': a_iter
                                        }, SAVE_PATH)
                                elif args.version in [76, 81]:
                                    if args.dataset == 'domainnet':
                                        torch.save({
                                            'model_0': models[0].state_dict(),
                                            'model_1': models[1].state_dict(),
                                            'model_2': models[2].state_dict(),
                                            'model_3': models[3].state_dict(),
                                            'model_4': models[4].state_dict(),
                                            'model_5': models[5].state_dict(),
                                            'emodel_0': extra_modules[0].state_dict(),
                                            'emodel_1': extra_modules[1].state_dict(),
                                            'emodel_2': extra_modules[2].state_dict(),
                                            'emodel_3': extra_modules[3].state_dict(),
                                            'emodel_4': extra_modules[4].state_dict(),
                                            'emodel_5': extra_modules[5].state_dict(),
                                            'p_model_0': personalized_models[0].state_dict(),
                                            'p_model_1': personalized_models[1].state_dict(),
                                            'p_model_2': personalized_models[2].state_dict(),
                                            'p_model_3': personalized_models[3].state_dict(),
                                            'p_model_4': personalized_models[4].state_dict(),
                                            'p_model_5': personalized_models[5].state_dict(),
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
                                            'emodel_0': extra_modules[0].state_dict(),
                                            'emodel_1': extra_modules[1].state_dict(),
                                            'emodel_2': extra_modules[2].state_dict(),
                                            'emodel_3': extra_modules[3].state_dict(),
                                            'p_model_0': personalized_models[0].state_dict(),
                                            'p_model_1': personalized_models[1].state_dict(),
                                            'p_model_2': personalized_models[2].state_dict(),
                                            'p_model_3': personalized_models[3].state_dict(),
                                            'best_epoch': best_epoch,
                                            'best_acc': best_acc,
                                            'a_iter': a_iter
                                        }, SAVE_PATH)
                                else:
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
                        best_changed = False
                        for client_idx, datasite in enumerate(datasets):
                            a_model = None
                            _, test_acc = test(client_idx, models[client_idx], personalized_models[client_idx], a_model, extra_modules, test_loaders[client_idx], loss_fun, device, args, hnet, global_prototype)
                            if args.mode == 'peer':
                                print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f}'.format(datasite, best_epoch, test_acc[0], test_acc[1], test_acc[2]))
                            else:
                                print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, best_epoch, test_acc))
                            if args.log:
                                if args.mode == 'peer':
                                    logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f}'.format(datasite, best_epoch, test_acc[0], test_acc[1], test_acc[2]))
                                else:
                                    logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, best_epoch, test_acc))

                    elif args.mode.lower() == 'fedtp':
                        if args.save_model:
                            torch.save({
                                'server_model': server_model.state_dict(),
                                'hnet': hnet.state_dict(), 
                                'best_epoch': best_epoch,
                                'best_acc': best_acc,
                                'a_iter': a_iter
                            }, SAVE_PATH)
                        best_changed = False
                        for client_idx, datasite in enumerate(datasets):
                            _, test_acc = test(client_idx, server_model, None, None, None, test_loaders[client_idx], loss_fun, device, args, hnet, None)
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
                            _, test_acc = test(client_idx, server_model, None, None, None, test_loaders[client_idx], loss_fun, device, args, None, None)
                            print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, best_epoch, test_acc))
                            if args.log:
                                logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, best_epoch, test_acc))

            else:
                if a_iter<=15 or (a_iter+1)%50==0:
                    if args.mode in ['fedbn', 'local', 'AlignFed', 'COPA']:
                        for client_idx, datasite in enumerate(datasets):
                            _, test_acc = test(client_idx, models[client_idx], None, None, None, test_loaders[client_idx], loss_fun, device, args, None, None)
                            print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, a_iter, test_acc))
                            if args.log:
                                logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, a_iter, test_acc))
                    elif args.mode.lower() in ['peer', 'fedper', 'fedrod']:
                        for client_idx, datasite in enumerate(datasets):
                            a_model = None
                            _, test_acc = test(client_idx, models[client_idx], personalized_models[client_idx], a_model, extra_modules, test_loaders[client_idx], loss_fun, device, args, hnet, global_prototype)
                            if args.mode == 'peer':
                                print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f}'.format(datasite, a_iter, test_acc[0], test_acc[1], test_acc[2]))
                            else:
                                print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, a_iter, test_acc))
                            if args.log:
                                if args.mode == 'peer':
                                    logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f}'.format(datasite, a_iter, test_acc[0], test_acc[1], test_acc[2]))
                                else:
                                    logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, a_iter, test_acc))
                    elif args.mode.lower() == 'fedtp':
                        for client_idx, datasite in enumerate(datasets):
                            _, test_acc = test(client_idx, server_model, None, None, None, test_loaders[client_idx], loss_fun, device, args, hnet, None)
                            print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, a_iter, test_acc))
                            if args.log:
                                logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, a_iter, test_acc))
                    else:
                        for client_idx, datasite in enumerate(datasets):
                            _, test_acc = test(client_idx, server_model, None, None, None, test_loaders[client_idx], loss_fun, device, args, None, None)
                            print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, a_iter, test_acc))
                            if args.log:
                                logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, a_iter, test_acc))

            if a_iter == args.iters-1:
                if args.dataset != 'digits':
                    if args.mode in ['fedbn', 'local', 'AlignFed', 'COPA']:
                        for client_idx, datasite in enumerate(datasets):
                            _, test_acc = test(client_idx, models[client_idx], None, None, None, test_loaders[client_idx], loss_fun, device, args, None, None)
                            print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, best_epoch, test_acc))
                            if args.log:
                                logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, best_epoch, test_acc))
                    elif args.mode.lower() in ['peer', 'fedper', 'fedrod']:
                        for client_idx, datasite in enumerate(datasets):
                            a_model = None
                            _, test_acc = test(client_idx, models[client_idx], personalized_models[client_idx], a_model, extra_modules, test_loaders[client_idx], loss_fun, device, args, hnet, global_prototype)
                            if args.mode == 'peer':
                                print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f}'.format(datasite, best_epoch, test_acc[0], test_acc[1], test_acc[2]))
                            else:
                                print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, best_epoch, test_acc))
                            if args.log:
                                if args.mode == 'peer':
                                    logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f}'.format(datasite, best_epoch, test_acc[0], test_acc[1], test_acc[2]))
                                else:
                                    logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, best_epoch, test_acc))
                    elif args.mode.lower() == 'fedtp':
                        for client_idx, datasite in enumerate(datasets):
                            _, test_acc = test(client_idx, server_model, None, None, None, test_loaders[client_idx], loss_fun, device, args, hnet, None)
                            print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, best_epoch, test_acc))
                            if args.log:
                                logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, best_epoch, test_acc))
                    else:
                        for client_idx, datasite in enumerate(datasets):
                            _, test_acc = test(client_idx, server_model, None, None, None, test_loaders[client_idx], loss_fun, device, args, None, None)
                            print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, best_epoch, test_acc))
                            if args.log:
                                logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, best_epoch, test_acc))

                else:
                    print(' Saving checkpoints to {}...'.format(SAVE_PATH))
                    logfile.write(' Saving checkpoints to {}...\n'.format(SAVE_PATH))                 
                    if args.mode in ['fedbn', 'local', 'AlignFed', 'COPA']:
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
                    elif args.mode in ['peer', 'fedper', 'fedrod']:
                        if args.save_model:
                            if args.mode == 'peer':
                                if args.version in [63, 37]:
                                    torch.save({
                                        'model_0': models[0].state_dict(),
                                        'model_1': models[1].state_dict(),
                                        'model_2': models[2].state_dict(),
                                        'model_3': models[3].state_dict(),
                                        'model_4': models[4].state_dict(),
                                        'p_model_0': personalized_models[0].state_dict(),
                                        'p_model_1': personalized_models[1].state_dict(),
                                        'p_model_2': personalized_models[2].state_dict(),
                                        'p_model_3': personalized_models[3].state_dict(),
                                        'p_model_4': personalized_models[4].state_dict(),
                                        'best_epoch': best_epoch,
                                        'best_acc': best_acc,
                                        'a_iter': a_iter
                                    }, SAVE_PATH)
                                elif args.version == 70:
                                    torch.save({
                                        'server_model': server_model.state_dict(),
                                        'emodel_0': extra_modules[0].state_dict(),
                                        'emodel_1': extra_modules[1].state_dict(),
                                        'emodel_2': extra_modules[2].state_dict(),
                                        'emodel_3': extra_modules[3].state_dict(),
                                        'emodel_4': extra_modules[4].state_dict(),
                                        'p_model_0': personalized_models[0].state_dict(),
                                        'p_model_1': personalized_models[1].state_dict(),
                                        'p_model_2': personalized_models[2].state_dict(),
                                        'p_model_3': personalized_models[3].state_dict(),
                                        'p_model_4': personalized_models[4].state_dict(),
                                        'best_epoch': best_epoch,
                                        'best_acc': best_acc,
                                        'a_iter': a_iter
                                    }, SAVE_PATH)
                                elif args.version in [76, 81]:
                                    torch.save({
                                        'model_0': models[0].state_dict(),
                                        'model_1': models[1].state_dict(),
                                        'model_2': models[2].state_dict(),
                                        'model_3': models[3].state_dict(),
                                        'model_4': models[4].state_dict(),
                                        'emodel_0': extra_modules[0].state_dict(),
                                        'emodel_1': extra_modules[1].state_dict(),
                                        'emodel_2': extra_modules[2].state_dict(),
                                        'emodel_3': extra_modules[3].state_dict(),
                                        'emodel_4': extra_modules[4].state_dict(),
                                        'p_model_0': personalized_models[0].state_dict(),
                                        'p_model_1': personalized_models[1].state_dict(),
                                        'p_model_2': personalized_models[2].state_dict(),
                                        'p_model_3': personalized_models[3].state_dict(),
                                        'p_model_4': personalized_models[4].state_dict(),
                                        'best_epoch': best_epoch,
                                        'best_acc': best_acc,
                                        'a_iter': a_iter
                                    }, SAVE_PATH)
                                else:
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
                    elif args.mode.lower() == 'fedtp':
                        if args.save_model:
                            torch.save({
                                'server_model': server_model.state_dict(),
                                'hnet': hnet.state_dict(), 
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
