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
from utils.data_utils import prepare_data_domainNet, prepare_data_officeHome
from utils.methods import local_training
from utils.utils import  communication, test, set_client_weight, visualize, log_write_dictionary, show_dictionary, log_write_running_dictionary, show_running_dictionary, visualize_combination, adapt_lambda
from utils.func_v import definite_version
from nets.models import AlexNet, AlexNet_rod, AlexNet_peer, P_Head, AlexNet_ada, AlexNet_adaG, AlexNet_adaP, AlexNet_adapt, AlexNet_adaptP, AlexNet_adaptP_trial
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
    parser.add_argument('--mode', type = str, default='fedbn', help='[FedBN | FedAvg | FedProx | fedper | fedrod | fedtp | fedap | peer | AlignFed | COPA]')
    parser.add_argument('--mu', type=float, default=1e-3, help='The hyper parameter for fedprox')
    parser.add_argument('--save_path', type = str, default='../checkpoint/domainnet', help='path to save the checkpoint')
    parser.add_argument('--resume', action='store_true', help ='resume training from the save path checkpoint')
    parser.add_argument('--dataset', type=str, default='domainnet')
    parser.add_argument('--version', type=int, default=1)
    parser.add_argument('--backbone', type=str, default='alexnet')
    args = parser.parse_args()

    if args.dataset == 'domainnet':
        exp_folder = 'fed_domainnet'
        train_loaders, val_loaders, test_loaders = prepare_data_domainNet(args)
    elif args.dataset == 'office_home':
        exp_folder = 'fed_office'
        train_loaders, val_loaders, test_loaders = prepare_data_officeHome(args)


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
    
    # train_loaders, val_loaders, test_loaders = prepare_data(args)

    
    # setup model
    if args.mode in ['fedper' ,'fedrod']:
        server_model = AlexNet_rod().to(device)
    elif args.mode == 'fedtp':
        server_model = ViT(image_size = 256, patch_size = 16, num_classes = 10, dim = 768, depth = 6, heads = 8, mlp_dim = 3072,
                  dropout = 0.1, emb_dropout = 0.1).to(device)
    elif args.mode == 'peer':
        if args.version in [18, 25, 58, 62, 28, 46, 47, 48, 49, 50, 1, 2, 5, 15, 23, 24, 26, 29, 7, 8, 11, 12, 19, 20, 27, 30, 31, 32, 
                            33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 59, 60, 61, 63, 64, 70, 71, 76, 81, 82, 83, 84, 85, 86,
                            87, 88, 89]:
            server_model = AlexNet().to(device)
        elif args.version in [3, 4, 6, 16, 9, 10, 13, 14, 21, 22]:
            server_model = AlexNet_adaG().to(device)
            # print(server_model.state_dict().keys())
            # dark_model = AlexNet_adaG().to(device)
            # for kkk in server_model.state_dict().keys():
            #     if 'adap' not in kkk:
            #         if 'num_batches_tracked' not in kkk:
            #             print(kkk)
            #             dark_model.state_dict()[kkk].data.copy_(server_model.state_dict()[kkk])
            # assert False
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
    extra_modules = []
    hnet = None
    global_prototypes = []
    global_prototype = None
    valid_onehot = {x:0  for x in range(client_num)}
    weight_dict = {x:[1/client_num for i in range(client_num)]  for x in range(client_num)}
    

    if args.mode == 'peer':
        if args.version == 17:
            print("Version17: Train two models seperately")

        elif args.version == 18:
            print("Version18: Global branch generalize, train two branches seperately")

        elif args.version == 25:
            print("Version25: Version18 + personal branch generalize")

        elif args.version == 28:
            print("Version28: Version18 + generalized heads are pretrained")

        elif args.version == 52:
            print("Version52: P head is personalized")

        elif args.version == 58:
            print("Version58: V18 and show generalized branch details")

        elif args.version == 59:
            print("Version59: upper bound, use accuracy in test set as weights")

        elif args.version == 60:
            print("Version60: approximation version for version59 with validation set")

        elif args.version == 61:
            print("Version61: comparison version with V59 with no generalized branch")

        elif args.version == 63:
            print("Version63: quick experiment, G branch now is based on FedBN")

        elif args.version == 64:
            print("Version64: comparison version with V63 with no generalized branch")

        elif args.version == 70:
            print("Version70: Use validation set to learn lambda")

        elif args.version == 76:
            print("Version76: V76 uses validation set to learn lambda and G branch is now fedBN")

        elif args.version == 77:
            print("Version77: V77 V73 and use specific_head")

        elif args.version == 78:
            print("Version78: V78 V75 and use specific_head")

        elif args.version == 79:
            print("Version79: V79 upper bound uses other client's p model to calculate collaboration weight")

        elif args.version == 80:
            print("Version80: V80 approximation uses other client's g model to calculate collaboration weight")

        elif args.version == 81:
            print("Version81: V81 validation set to learn lambda, G branch is fedBN, P branch is generalized")

        elif args.version == 82:
            print("Version82: V82 uses validation set to learn lambda and G branch is now fedBN and FedPer")

        elif args.version == 83:
            print("Version83: V83 G branch is based on FedPer and G branch generalizes")

        elif args.version == 84:
            print("Version84: V84 G branch is based on FedPer and FedBN, G branch generalizes")

        elif args.version == 85:
            print("Version85: V82 and P branch is generalized")

        elif args.version == 86:
            print("Version86: V86 uses validation set to learn lambda and G is FedPer")

        elif args.version == 87:
            print("Version87: V87 uses validation set to learn lambda and G is FedPer, P generalizes")

        elif args.version == 88:
            print("Version88: V88 P branch shares feature extractor with G branch")

        elif args.version == 89:
            print("Version89: V89 P branch shares feature extractor with G branch, G branch generalizes")

        else:
            definite_version(args.version)


        if args.version == 31:
            personalized_models = [AlexNet_adaptP().to(device) for idx in range(client_num)]
        elif args.version in [88, 89]:
            personalized_models = [P_Head().to(device) for idx in range(client_num)]
        elif args.version in [70, 76, 81, 82, 85, 86, 87]:
            meb = nn.Embedding(num_embeddings=1, embedding_dim=1).to(device)
            # print(meb.state_dict())
            meb.state_dict()['weight'].data.copy_(torch.zeros([1], dtype=torch.long))
            # print(meb.state_dict())
            extra_modules = [copy.deepcopy(meb) for idx in range(client_num)]
            personalized_models = [AlexNet().to(device) for idx in range(client_num)]
        elif args.version == 71:
            extra_modules = [nn.Linear(20, 10).to(device) for idx in range(client_num)]
            personalized_models = [AlexNet().to(device) for idx in range(client_num)]
        elif args.version == 43:
            personalized_models = [AlexNet_adaptP_trial().to(device) for idx in range(client_num)]
        elif args.version == 55:
            extra_modules = { idx: nn.Linear(4096, 10).to(device) for idx in range(client_num)}
            RESULTS_PATH = os.path.join(args.save_path, '{}'.format('local'))
            savepoint = torch.load(RESULTS_PATH)
            ccc_model = AlexNet().to(device)
            for client_idx in range(client_num):
                ccc_model.load_state_dict(savepoint['model_{}'.format(client_idx)])
                extra_modules[client_idx].load_state_dict(ccc_model.state_dict(), strict=False)
            personalized_models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]
        elif args.version in [56, 57, 59, 60, 61, 72, 73, 74, 75, 77, 78, 79, 80]:
            paggregation_models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]
            personalized_models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]
        elif args.version in [65, 66, 68, 69]:
            extra_modules = {}
            personalized_models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]
        elif args.version == 28:
            extra_modules = { idx: P_Head().to(device) for idx in range(client_num)}
            RESULTS_PATH = os.path.join(args.save_path, '{}'.format('local'))
            savepoint = torch.load(RESULTS_PATH)
            ccc_model = AlexNet().to(device)
            for client_idx in range(client_num):
                ccc_model.load_state_dict(savepoint['model_{}'.format(client_idx)])
                extra_modules[client_idx].load_state_dict(ccc_model.state_dict(), strict=False)
            personalized_models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]
        elif args.version in [35 ,37]:
            extra_unit = copy.deepcopy(server_model.classifier)
            extra_modules = { idx: copy.deepcopy(extra_unit) for idx in range(client_num)}
            RESULTS_PATH = os.path.join(args.save_path, '{}'.format('local'))
            savepoint = torch.load(RESULTS_PATH)
            ccc_model = AlexNet().to(device)
            for client_idx in range(client_num):
                ccc_model.load_state_dict(savepoint['model_{}'.format(client_idx)])
                extra_modules[client_idx].load_state_dict(ccc_model.state_dict(), strict=False)
            personalized_models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]
        elif args.version in [23, 26]:
            extra_modules = [nn.Linear(256 * 6 * 6, 256 * 6 * 6, bias=False).to(device) for idx in range(client_num)]
            personalized_models = [AlexNet().to(device) for idx in range(client_num)]
        elif args.version in [1, 2]:
            personalized_models = [AlexNet_ada().to(device) for idx in range(client_num)]
        elif args.version in [3, 4]:
            personalized_models = [AlexNet_adaP().to(device) for idx in range(client_num)]
        elif args.version in [5, 15, 24, 29, 7, 8, 11, 12, 19, 20, 30, 40, 41, 42]:
            personalized_models = [AlexNet_adapt().to(device) for idx in range(client_num)]
        elif args.version in [6, 16, 9, 10, 13, 14, 21, 22]:
            personalized_models = [AlexNet().to(device) for idx in range(client_num)]
        else:
            personalized_models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]

    elif args.mode in ['fedper', 'fedrod']:
        priviate_head = P_Head().to(device)
        personalized_models = [copy.deepcopy(priviate_head).to(device) for idx in range(client_num)]

    elif args.mode == 'fedap':
        paggregation_models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]
        RESULTS_PATH = os.path.join(args.save_path, '{}'.format('fedbn'))
        savepoint = torch.load(RESULTS_PATH)
        for client_idx in range(client_num):
            paggregation_models[client_idx].load_state_dict(savepoint['model_{}'.format(client_idx)])
            # models[client_idx].load_state_dict(savepoint['model_{}'.format(client_idx)])
        client_weights = set_client_weight(train_loaders, paggregation_models, client_num, device)
        print("client weights: ", client_weights)

    elif args.mode == 'fedtp':
        hnet = ViTHyper(client_num, 128, hidden_dim = 256, dim=768, 
        heads = 8, dim_head = 64, n_hidden = 3, depth=6, client_sample=client_num).to(device)

    best_changed = False

    if args.test:
        print('Loading snapshots...')
        checkpoint = torch.load('/public/home/caizhy/work/Peer/checkpoint/domainnet/fed_domainnet/{}{}'.format(args.mode.lower(),args.version))
        server_model.load_state_dict(checkpoint['server_model'])
        if args.mode.lower() in ['fedbn', 'local', 'fedap']:
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
                _, test_acc = test(test_idx, models[test_idx], p_model, extra_modules, test_loader, loss_fun, device, args, hnet, global_prototype)
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
    best_acc_dict = {x:[] for x in range(client_num)}
    final_acc_dict = {x:[] for x in range(client_num)}
    for running_time in range(5):
        print("============ Running Time {} ============".format(running_time + 1))

        best_epoch = 0
        best_acc = [0. for j in range(client_num)] 

        if args.log:
            logfile.write("============ Running Time {} ============\n".format(running_time + 1))
        for a_iter in range(start_iter, args.iters):
            optimizers = [optim.SGD(params=models[idx].parameters(), lr=args.lr, momentum=args.momentum) for idx in range(client_num)]
            for wi in range(args.wk_iters):
                print("============ Train epoch {} ============".format(wi + a_iter * args.wk_iters))
                if args.log:
                    logfile.write("============ Train epoch {} ============\n".format(wi + a_iter * args.wk_iters)) 

                if args.version == 60:
                    train_loss, train_acc, proto_dict, weight_dict = local_training(models, personalized_models, paggregation_models, hnet, server_model, global_prototypes, extra_modules, 
                                                                   valid_onehot, weight_dict, args, train_loaders, val_loaders, optimizers, loss_fun, device, a_iter=a_iter)  
                else:
                    train_loss, train_acc, proto_dict, weight_dict = local_training(models, personalized_models, paggregation_models, hnet, server_model, global_prototypes, extra_modules, 
                                                                   valid_onehot, weight_dict, args, train_loaders, test_loaders, optimizers, loss_fun, device, a_iter=a_iter)
            
            with torch.no_grad():
                # Aggregation
                if len(weight_dict[0]) != 0:
                    assert args.version in [59, 60, 61, 72, 73, 74, 75, 77, 78, 79, 80]
                    server_model, models, global_prototypes = communication(args, server_model, models, personalized_models, extra_modules, paggregation_models, weight_dict, proto_dict, a_iter)
                    for ccidx in range(client_num):
                        print("Dataset: ", datasets[ccidx], " weight list: ", weight_dict[ccidx])
                    if args.log:
                        for ccidx in range(client_num):
                            wei_list = weight_dict[ccidx]
                            if args.dataset == 'domainnet':
                                logfile.write(' Site-{:<10s}| w1: {:.4f} | w2: {:.4f} | w3: {:.4f} | w4: {:.4f} | w5: {:.4f} | w6: {:4f}\n'.format(datasets[ccidx], wei_list[0], wei_list[1], wei_list[2], wei_list[3], wei_list[4], wei_list[5]))
                            else:
                                logfile.write(' Site-{:<10s}| w1: {:.4f} | w2: {:.4f} | w3: {:.4f} | w4: {:.4f}\n'.format(datasets[ccidx], wei_list[0], wei_list[1], wei_list[2], wei_list[3]))
                else:
                    server_model, models, global_prototypes = communication(args, server_model, models, personalized_models, extra_modules, paggregation_models, client_weights, proto_dict, a_iter)

            if args.version in [70, 71, 76, 81, 82, 85, 86, 87]:
                assert args.mode == 'peer'
                train_loss, train_acc, _, _ = local_training(models, personalized_models, None, None, None, None, extra_modules, 
                                            None, None, args, None, val_loaders, None, loss_fun, device, a_iter=a_iter, phase='Valid')
                
                if args.version != 71:
                    for ccidx in range(client_num):
                        e_model = extra_modules[ccidx]
                        e_model.eval()
                        lambda_p = torch.sigmoid(e_model(torch.tensor([0], dtype=torch.long).to(device)))
                        lambda_g = 1 - lambda_p
                        print("Dataset: ", datasets[ccidx], " lambda_g: ", lambda_g[0].item(), " lambda_p: ", lambda_p[0].item())
                        if args.log:
                            logfile.write(' Site-{:<10s}| lambda_g: {:.4f} | lambda_p: {:.4f}\n'.format(datasets[ccidx], lambda_g[0].item(), lambda_p[0].item()))

            with torch.no_grad():

                if args.mode == 'peer':
                    for client_idx, model in enumerate(models):
                        if args.version == 58:
                            extra_modules.append(copy.deepcopy(personalized_models[client_idx].classifier))
                        elif args.version in [65, 66, 68, 69]:
                            extra_modules[client_idx] = {}
                            for kky in personalized_models[client_idx].state_dict().keys():
                                if 'bn' in kky:
                                    if 'num_batches_tracked' not in kky:
                                        extra_modules[client_idx][kky] = copy.deepcopy(personalized_models[client_idx].state_dict()[kky])
                        elif args.version in [5, 11, 12, 19, 20 ,29, 31]:
                            extra_modules.append([copy.deepcopy(personalized_models[client_idx].f_adaptor), copy.deepcopy(personalized_models[client_idx].classifier)])
                        elif args.version in [6, 13, 14, 21, 22]:
                            extra_modules.append([copy.deepcopy(models[client_idx].adap3.state_dict()), copy.deepcopy(models[client_idx].adap4.state_dict()), copy.deepcopy(models[client_idx].adap5.state_dict()), copy.deepcopy(personalized_models[client_idx].classifier)])

                # Report loss after aggregation
                # Validation
                val_acc_list = [None for j in range(client_num)]
                for client_idx, model in enumerate(models):
                    if args.mode in ['peer', 'fedper', 'fedrod']:
                        p_model = personalized_models[client_idx]
                    else:
                        p_model = None
                    if args.version in [27, 30]:
                        val_loss, val_acc = test(client_idx, model, personalized_models, extra_modules, val_loaders[client_idx], loss_fun, device, args, hnet, global_prototype)
                    else:
                        val_loss, val_acc = test(client_idx, model, p_model, extra_modules, val_loaders[client_idx], loss_fun, device, args, hnet, global_prototype)
                    if args.mode == 'peer':
                        val_acc_list[client_idx] = val_acc[0]
                        if args.version == 27:
                            print(' Site-{:<10s}| Val Loss: {:.4f} | G_branch Loss: {:.4f} | P_branch Loss: {:.4f} | Val Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f}'.format(datasets[client_idx] ,val_loss[0], val_loss[1], val_loss[2], val_acc[0], val_acc[1], val_acc[2]))
                            show_dictionary(logfile, val_loss[3], a_iter, mode='Loss', data='domainnet', division='Val')
                            show_dictionary(logfile, val_acc[3], a_iter, mode='Acc', data='domainnet', division='Val')
                        elif args.version in [58, 65, 66, 68, 69]:
                            print(' Site-{:<10s}| Val Loss: {:.4f} | G_branch Loss: {:.4f} | P_branch Loss: {:.4f} | Gen Sum Loss: {:.4f} | Val Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f} | Gen Sum Acc: {:.4f}'.format(datasets[client_idx] ,val_loss[0], val_loss[1], val_loss[2], val_loss[3], val_acc[0], val_acc[1], val_acc[2], val_acc[3]))
                            show_dictionary(logfile, val_loss[4], a_iter, mode='Loss', data='domainnet', division='Val')
                            show_dictionary(logfile, val_acc[4], a_iter, mode='Acc', data='domainnet', division='Val')
                        elif args.version in [1, 2, 3, 4]:
                            print(' Site-{:<10s}| Val Loss: {:.4f} | G_branch Loss: {:.4f} | P_branch Loss: {:.4f} | G_adapt Loss: {:.4f} | P_adapt Loss: {:.4f} | Val Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f} | G_adapt Acc: {:.4f} | P_adapt Acc: {:.4f}'.format(datasets[client_idx] ,val_loss[0], val_loss[1], val_loss[2], val_loss[3], val_loss[4], val_acc[0], val_acc[1], val_acc[2], val_acc[3], val_acc[4]))
                        elif args.version in [5, 29, 6, 11, 12, 13, 14, 19, 20, 21, 22, 30, 31]:
                            print(' Site-{:<10s}| Val Loss: {:.4f} | G_branch Loss: {:.4f} | P_branch Loss: {:.4f} | G_adapt Loss: {:.4f} | Val Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f} | G_adapt Acc: {:.4f}'.format(datasets[client_idx] ,val_loss[0], val_loss[1], val_loss[2], val_loss[3], val_acc[0], val_acc[1], val_acc[2], val_acc[3]))
                            show_dictionary(logfile, val_loss[4], a_iter, mode='Loss', data='domainnet', division='Val')
                            show_dictionary(logfile, val_acc[4], a_iter, mode='Acc', data='domainnet', division='Val')
                        elif args.version in [15, 16, 23, 24, 26, 40, 41, 42, 43]:
                            print(' Site-{:<10s}| Val Loss: {:.4f} | G_branch Loss: {:.4f} | P_branch Loss: {:.4f} | G_adapt Loss: {:.4f} | Val Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f} | G_adapt Acc: {:.4f}'.format(datasets[client_idx] ,val_loss[0], val_loss[1], val_loss[2], val_loss[3], val_acc[0], val_acc[1], val_acc[2], val_acc[3]))
                        else:
                            print(' Site-{:<10s}| Val Loss: {:.4f} | G_branch Loss: {:.4f} | P_branch Loss: {:.4f} | Val Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f}'.format(datasets[client_idx] ,val_loss[0], val_loss[1], val_loss[2], val_acc[0], val_acc[1], val_acc[2]))
                    else:
                        val_acc_list[client_idx] = val_acc
                        print(' Site-{:<10s}| Val  Loss: {:.4f} | Val  Acc: {:.4f}'.format(datasets[client_idx], val_loss, val_acc))
                    if args.log:
                        if args.mode == 'peer':
                            #G acc is higher than P acc
                            if val_acc[1] > val_acc[2]:
                                valid_onehot[client_idx] = 1
                            else:
                                valid_onehot[client_idx] = 0
                            if args.version == 27:
                                logfile.write(' Site-{:<10s}| Val Loss: {:.4f} | G_branch Loss: {:.4f} | P_branch Loss: {:.4f} | Val Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f}'.format(datasets[client_idx] ,val_loss[0], val_loss[1], val_loss[2], val_acc[0], val_acc[1], val_acc[2]))
                                log_write_dictionary(logfile, val_loss[3], mode='Loss', data='domainnet', division='Val')
                                log_write_dictionary(logfile, val_acc[3], mode='Acc', data='domainnet', division='Val')
                            elif args.version in [58, 65, 66, 68, 69]:
                                logfile.write(' Site-{:<10s}| Val Loss: {:.4f} | G_branch Loss: {:.4f} | P_branch Loss: {:.4f} | Gen Sum Loss: {:.4f} | Val Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f} | Gen Sum Acc: {:.4f}'.format(datasets[client_idx] ,val_loss[0], val_loss[1], val_loss[2], val_loss[3], val_acc[0], val_acc[1], val_acc[2], val_acc[3]))
                                log_write_dictionary(logfile, val_loss[4], mode='Loss', data='domainnet', division='Val')
                                log_write_dictionary(logfile, val_acc[4], mode='Acc', data='domainnet', division='Val')
                            elif args.version in [1, 2, 3, 4]:
                                logfile.write(' Site-{:<10s}| Val Loss: {:.4f} | G_branch Loss: {:.4f} | P_branch Loss: {:.4f} | G_adapt Loss: {:.4f} | P_adapt Loss: {:.4f} | Val Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f} | G_adapt Acc: {:.4f} | P_adapt Acc: {:.4f}'.format(datasets[client_idx] ,val_loss[0], val_loss[1], val_loss[2], val_loss[3], val_loss[4], val_acc[0], val_acc[1], val_acc[2], val_acc[3], val_acc[4]))
                            elif args.version in [5, 29, 6, 11, 12, 13, 14, 19, 20, 21, 22, 30, 31]:
                                logfile.write(' Site-{:<10s}| Val Loss: {:.4f} | G_branch Loss: {:.4f} | P_branch Loss: {:.4f} | G_adapt Loss: {:.4f} | Val Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f} | G_adapt Acc: {:.4f}'.format(datasets[client_idx] ,val_loss[0], val_loss[1], val_loss[2], val_loss[3], val_acc[0], val_acc[1], val_acc[2], val_acc[3]))
                                log_write_dictionary(logfile, val_loss[4], mode='Loss', data='domainnet', division='Val')
                                log_write_dictionary(logfile, val_acc[4], mode='Acc', data='domainnet', division='Val')
                            elif args.version in [15, 16, 23, 24, 26, 40, 41, 42, 43]:
                                logfile.write(' Site-{:<10s}| Val Loss: {:.4f} | G_branch Loss: {:.4f} | P_branch Loss: {:.4f} | G_adapt Loss: {:.4f} | Val Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f} | G_adapt Acc: {:.4f}'.format(datasets[client_idx] ,val_loss[0], val_loss[1], val_loss[2], val_loss[3], val_acc[0], val_acc[1], val_acc[2], val_acc[3]))
                            else:
                                logfile.write(' Site-{:<10s}| Val Loss: {:.4f} | G_branch Loss: {:.4f} | P_branch Loss: {:.4f} | Val Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f}'.format(datasets[client_idx] ,val_loss[0], val_loss[1], val_loss[2], val_acc[0], val_acc[1], val_acc[2]))
                            
                            # if args.version in [43]:
                            #     logfile.write(' Site-{:<10s}| Val Loss: {:.4f} | G_branch Loss: {:.4f} | P_branch Loss: {:.4f} | M_branch Loss: {:.4f} | Val Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f} | M Acc: {:.4f}'.format(datasets[client_idx] ,val_loss[0], val_loss[1], val_loss[2], val_loss[3], val_acc[0], val_acc[1], val_acc[2], val_acc[3]))
                        else:
                            logfile.write(' Site-{:<10s}| Val  Loss: {:.4f} | Val  Acc: {:.4f}\n'.format(datasets[client_idx], val_loss, val_acc))

                if args.mode == 'peer' and args.version in [46, 47, 48, 49, 50, 34, 35]:
                    print(valid_onehot)

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
                    if args.mode in ['fedbn', 'local', 'fedap', 'AlignFed', 'COPA']:
                        best_changed = False
                        for client_idx, datasite in enumerate(datasets):
                            _, test_acc = test(client_idx, models[client_idx], None, None, test_loaders[client_idx], loss_fun, device, args, None, None)
                            print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, best_epoch, test_acc))

                            if len(best_acc_dict[client_idx]) == running_time:
                                best_acc_dict[client_idx].append(test_acc)
                            else:
                                best_acc_dict[client_idx][running_time] = test_acc

                            if args.log:
                                logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, best_epoch, test_acc))
                    
                    elif args.mode in ['peer', 'fedper', 'fedrod']:
                        best_changed = False
                        for client_idx, datasite in enumerate(datasets):
                            if args.version in [27, 30]:
                                _, test_acc = test(client_idx, models[client_idx], personalized_models, extra_modules, test_loaders[client_idx], loss_fun, device, args, None, global_prototype)
                            else:
                                _, test_acc = test(client_idx, models[client_idx], personalized_models[client_idx], extra_modules, test_loaders[client_idx], loss_fun, device, args, None, global_prototype)
                            if args.mode == 'peer':
                                if args.version == 27:
                                    print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f}'.format(datasite, best_epoch, test_acc[0], test_acc[1], test_acc[2]))
                                    show_dictionary(logfile, test_acc[3], a_iter, mode='Acc', data='domainnet', division='Test')
                                elif args.version in [58, 65, 66, 68, 69]:
                                    print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f} | Gen Sum Acc: {:.4f}'.format(datasite, best_epoch, test_acc[0], test_acc[1], test_acc[2], test_acc[3]))
                                    show_dictionary(logfile, test_acc[4], a_iter, mode='Acc', data='domainnet', division='Test')
                                elif args.version in [1, 2, 3, 4]:
                                    print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f} | G_adapt Acc: {:.4f} | P_adapt Acc: {:.4f}'.format(datasite, best_epoch, test_acc[0], test_acc[1], test_acc[2], test_acc[3], test_acc[4]))
                                elif args.version in [5, 29, 6, 11, 12, 13, 14, 19, 20, 21, 22, 30, 31]:
                                    print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f} | G_adapt Acc: {:.4f}'.format(datasite, best_epoch, test_acc[0], test_acc[1], test_acc[2], test_acc[3]))
                                    show_dictionary(logfile, test_acc[4], a_iter, mode='Acc', data='domainnet', division='Test')
                                elif args.version in [15, 16, 23, 24, 26, 40, 41, 42, 43]:
                                    print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f} | G_adapt Acc: {:.4f}'.format(datasite, best_epoch, test_acc[0], test_acc[1], test_acc[2], test_acc[3]))
                                else:
                                    print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f}'.format(datasite, best_epoch, test_acc[0], test_acc[1], test_acc[2]))

                                if len(best_acc_dict[client_idx]) == running_time:
                                    best_acc_dict[client_idx].append(test_acc[0])
                                else:
                                    best_acc_dict[client_idx][running_time] = test_acc[0]

                            else:
                                print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, best_epoch, test_acc))

                                if len(best_acc_dict[client_idx]) == running_time:
                                    best_acc_dict[client_idx].append(test_acc)
                                else:
                                    best_acc_dict[client_idx][running_time] = test_acc

                            if args.log:
                                if args.mode == 'peer':
                                    if args.version == 27:
                                        logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f}'.format(datasite, best_epoch, test_acc[0], test_acc[1], test_acc[2]))
                                        log_write_dictionary(logfile, test_acc[3], mode='Acc', data='domainnet', division='Test')
                                    elif args.version in [58, 65, 66, 68, 69]:
                                        logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f} | Gen Sum Acc: {:.4f}'.format(datasite, best_epoch, test_acc[0], test_acc[1], test_acc[2], test_acc[3]))
                                        log_write_dictionary(logfile, test_acc[4], mode='Acc', data='domainnet', division='Test')
                                    elif args.version in [1, 2, 3, 4]:
                                        logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f} | G_adapt Acc: {:.4f} | P_adapt Acc: {:.4f}'.format(datasite, best_epoch, test_acc[0], test_acc[1], test_acc[2], test_acc[3], test_acc[4]))
                                    elif args.version in [5, 29, 6, 11, 12, 13, 14, 19, 20, 21, 22, 30, 31]:
                                        logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f} | G_adapt Acc: {:.4f}'.format(datasite, best_epoch, test_acc[0], test_acc[1], test_acc[2], test_acc[3]))
                                        log_write_dictionary(logfile, test_acc[4], mode='Acc', data='domainnet', division='Test')
                                    elif args.version in [15, 16, 23, 24, 26, 40, 41, 42, 43]:
                                        logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f} | G_adapt Acc: {:.4f}'.format(datasite, best_epoch, test_acc[0], test_acc[1], test_acc[2], test_acc[3]))
                                    else:
                                        logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f}'.format(datasite, best_epoch, test_acc[0], test_acc[1], test_acc[2]))
                                else:
                                    logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, best_epoch, test_acc))

                    elif args.mode.lower() == 'fedtp':
                        best_changed = False
                        for client_idx, datasite in enumerate(datasets):
                            _, test_acc = test(client_idx, server_model, None, None, test_loaders[client_idx], loss_fun, device, args, hnet, None)
                            print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, best_epoch, test_acc))

                            if len(best_acc_dict[client_idx]) == running_time:
                                best_acc_dict[client_idx].append(test_acc)
                            else:
                                best_acc_dict[client_idx][running_time] = test_acc

                            if args.log:
                                logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, best_epoch, test_acc))

                    else:
                        best_changed = False
                        for client_idx, datasite in enumerate(datasets):
                            _, test_acc = test(client_idx, server_model, None, None, test_loaders[client_idx], loss_fun, device, args, None, None)
                            print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, best_epoch, test_acc))

                            if len(best_acc_dict[client_idx]) == running_time:
                                best_acc_dict[client_idx].append(test_acc)
                            else:
                                best_acc_dict[client_idx][running_time] = test_acc

                            if args.log:
                                logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, best_epoch, test_acc))

                if a_iter == args.iters-1:
                    if args.mode in ['fedbn', 'local', 'fedap', 'AlignFed', 'COPA']:
                        for client_idx, datasite in enumerate(datasets):
                            _, test_acc = test(client_idx, models[client_idx], None, None, test_loaders[client_idx], loss_fun, device, args, None, None)
                            print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, best_epoch, test_acc))

                            if len(final_acc_dict[client_idx]) == running_time:
                                final_acc_dict[client_idx].append(test_acc)
                            else:
                                final_acc_dict[client_idx][running_time] = test_acc

                            if args.log:
                                logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, best_epoch, test_acc))
                    elif args.mode.lower() in ['peer', 'fedper', 'fedrod']:
                        for client_idx, datasite in enumerate(datasets):
                            if args.version in [27, 30]:
                                _, test_acc = test(client_idx, models[client_idx], personalized_models, extra_modules, test_loaders[client_idx], loss_fun, device, args, None, global_prototype)
                            else:
                                _, test_acc = test(client_idx, models[client_idx], personalized_models[client_idx], extra_modules, test_loaders[client_idx], loss_fun, device, args, None, global_prototype)
                            if args.mode == 'peer':
                                if args.version == 27:
                                    print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f}'.format(datasite, best_epoch, test_acc[0], test_acc[1], test_acc[2]))
                                    show_dictionary(logfile, test_acc[3], a_iter, mode='Acc', data='domainnet', division='Test')
                                elif args.version in [58, 65, 66, 68, 69]:
                                    print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f} | Gen Sum Acc: {:.4f}'.format(datasite, best_epoch, test_acc[0], test_acc[1], test_acc[2], test_acc[3]))
                                    show_dictionary(logfile, test_acc[4], a_iter, mode='Acc', data='domainnet', division='Test')
                                elif args.version in [1, 2, 3, 4]:
                                    print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f} | G_adapt Acc: {:.4f} | P_adapt Acc: {:.4f}'.format(datasite, best_epoch, test_acc[0], test_acc[1], test_acc[2], test_acc[3], test_acc[4]))
                                elif args.version in [5, 29, 6, 11, 12, 13, 14, 19, 20, 21, 22, 30, 31]:
                                    print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f} | G_adapt Acc: {:.4f}'.format(datasite, best_epoch, test_acc[0], test_acc[1], test_acc[2], test_acc[3]))
                                    show_dictionary(logfile, test_acc[4], a_iter, mode='Acc', data='domainnet', division='Test')
                                elif args.version in [15, 16, 23, 24, 26, 40, 41, 42, 43]:
                                    print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f} | G_adapt Acc: {:.4f}'.format(datasite, best_epoch, test_acc[0], test_acc[1], test_acc[2], test_acc[3]))
                                else:
                                    print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f}'.format(datasite, best_epoch, test_acc[0], test_acc[1], test_acc[2]))
                            
                                if len(final_acc_dict[client_idx]) == running_time:
                                    final_acc_dict[client_idx].append(test_acc[0])
                                else:
                                    final_acc_dict[client_idx][running_time] = test_acc[0]

                            else:
                                print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, best_epoch, test_acc))

                                if len(final_acc_dict[client_idx]) == running_time:
                                    final_acc_dict[client_idx].append(test_acc)
                                else:
                                    final_acc_dict[client_idx][running_time] = test_acc

                            if args.log:
                                if args.mode == 'peer':
                                    if args.version == 27:
                                        logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f}'.format(datasite, best_epoch, test_acc[0], test_acc[1], test_acc[2]))
                                        log_write_dictionary(logfile, test_acc[3], mode='Acc', data='domainnet', division='Test')
                                    elif args.version in [58, 65, 66, 68, 69]:
                                        logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f} | Gen Sum Acc: {:.4f}'.format(datasite, best_epoch, test_acc[0], test_acc[1], test_acc[2], test_acc[3]))
                                        log_write_dictionary(logfile, test_acc[4], mode='Acc', data='domainnet', division='Test')
                                    elif args.version in [1, 2, 3, 4]:
                                        logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f} | G_adapt Acc: {:.4f} | P_adapt Acc: {:.4f}'.format(datasite, best_epoch, test_acc[0], test_acc[1], test_acc[2], test_acc[3], test_acc[4]))
                                    elif args.version in [5, 29, 6, 11, 12, 13, 14, 19, 20, 21, 22, 30, 31]:
                                        logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f} | G_adapt Acc: {:.4f}'.format(datasite, best_epoch, test_acc[0], test_acc[1], test_acc[2], test_acc[3]))
                                        log_write_dictionary(logfile, test_acc[4], mode='Acc', data='domainnet', division='Test')
                                    elif args.version in [15, 16, 23, 24, 26, 40, 41, 42, 43]:
                                        logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f} | G_adapt Acc: {:.4f}'.format(datasite, best_epoch, test_acc[0], test_acc[1], test_acc[2], test_acc[3]))
                                    else:
                                        logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f} | G Acc: {:.4f} | P Acc: {:.4f}'.format(datasite, best_epoch, test_acc[0], test_acc[1], test_acc[2]))
                                else:
                                    logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, best_epoch, test_acc))
                    elif args.mode.lower() == 'fedtp':
                        for client_idx, datasite in enumerate(datasets):
                            _, test_acc = test(client_idx, server_model, None, None, test_loaders[client_idx], loss_fun, device, args, hnet, None)
                            print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, best_epoch, test_acc))

                            if len(final_acc_dict[client_idx]) == running_time:
                                final_acc_dict[client_idx].append(test_acc)
                            else:
                                final_acc_dict[client_idx][running_time] = test_acc

                            if args.log:
                                logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, best_epoch, test_acc))
                    else:
                        for client_idx, datasite in enumerate(datasets):
                            _, test_acc = test(client_idx, server_model, None, None, test_loaders[client_idx], loss_fun, device, args, None, None)
                            print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, best_epoch, test_acc))

                            if len(final_acc_dict[client_idx]) == running_time:
                                final_acc_dict[client_idx].append(test_acc)
                            else:
                                final_acc_dict[client_idx][running_time] = test_acc

                            if args.log:
                                logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, best_epoch, test_acc))
                
                if args.version in [5, 29, 11, 12, 19, 20, 6, 13, 14, 21, 22, 31, 58]:
                    extra_modules = []
                elif args.version in [65, 66, 68, 69]:
                    extra_modules = {}

            if log:
                logfile.flush()
    
    print("best_acc_dict: ", best_acc_dict)
    log_write_running_dictionary(logfile, best_acc_dict, data=args.dataset)
    show_running_dictionary(best_acc_dict, data=args.dataset)

    print("final_acc_dict: ", final_acc_dict)
    log_write_running_dictionary(logfile, final_acc_dict, data=args.dataset)
    show_running_dictionary(final_acc_dict, data=args.dataset)

    if log:
        logfile.flush()
        logfile.close()