import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
from collections import OrderedDict, defaultdict
from pathlib import Path
import argparse
import logging
import os
import copy
from math import *
import random

import datetime
from torch.utils.tensorboard import SummaryWriter
from utils import *

from models.vit import ViT
from models.Hypernetworks import ViTHyper, ShakesHyper, Relation_net
from models.cnn import CNNHyper, CNNTarget
from model import *
from utils import *
from methods.method import *

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib as mpl

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vit', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='noniid-labeldir', help='the data partitioning strategy')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=1, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=10,  help='number of workers in a distributed cluster')
    parser.add_argument('--embed_extractor', type=str, default='hyperCnn',
                            help='communication strategy: fedavg/fedprox')
    parser.add_argument('--comm_round', type=int, default=200, help='number of maximum communication roun')
    parser.add_argument('--is_same_initial', type=int, default=1, help='Whether initial all the models with the same parameters in fedavg')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cpu', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox')
    parser.add_argument('--noise', type=float, default=0, help='how much noise we add to some party')
    parser.add_argument('--noise_type', type=str, default='None', help='Different level of noise or different space of noise')
    parser.add_argument('--rho', type=float, default=0, help='Parameter controlling the momentum SGD')
    parser.add_argument('--sample', type=float, default=0.4, help='Sample ratio for each communication round')
    parser.add_argument('--train_acc_pre', action='store_true')
    parser.add_argument('--eval_step', type=int, default=10)
    parser.add_argument('--test_round', type=int, default=0)
    parser.add_argument("--save_model", action='store_true')
    parser.add_argument("--comment", default="_")
    parser.add_argument("--definite_selection", action='store_true')
    parser.add_argument("--show_all_accuracy", action='store_true')
    parser.add_argument('--version', type=int, default=7)
    parser.add_argument('--log_flag', default=True)
    parser.add_argument('--alg', type=str, default='peer')
    parser.add_argument('--sample_part', action='store_true')

    """
    Used for hyperVit
    """
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--hyper_hid', type=int, default=100, help="hypernet hidden dim")
    parser.add_argument("--n-hidden", type=int, default=3, help="num. hidden layers")
    parser.add_argument('--layer_emd', action='store_true')
    parser.add_argument("--balanced_soft_max", action='store_true')
    parser.add_argument("--client_embed_size", type=int, default=128)

    args = parser.parse_args()
    return args


def init_nets(net_configs, dropout_p, n_parties, args):

    nets = {net_i: None for net_i in range(n_parties)}
    device = torch.device(args.device)

    for net_i in range(n_parties):       
        if args.model == "cnn":
            if args.dataset == 'cifar100':
                net = CNNTarget(n_kernels=16, out_dim=100)
            elif args.dataset == 'cifar10':
                net = CNNTarget(n_kernels=16, out_dim=10)
        elif args.model == "vit":
            net = ViT(image_size=32, patch_size=4, num_classes=100, dim=128, depth = args.depth, heads=8, mlp_dim=256,
                      dropout=0.1, emb_dropout=0.1)
        
        nets[net_i] = net.to(device)

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type


def init_hyper(args):
    # embed_dim = int(1 + args.n_parties / 4)
    embed_dim = args.client_embed_size
    batch_node = int(args.n_parties * 0.1)
   
    if args.model == "vit":
        hnet = ViTHyper(args.n_parties, embed_dim, hidden_dim = args.hyper_hid, dim=128,
                        heads=8, dim_head=64, n_hidden = args.n_hidden, depth=args.depth, client_sample=batch_node)          
    
    elif args.model == "cnn":
        if args.dataset == 'cifar100':
            hnet = CNNHyper(args.n_parties, embed_dim, hidden_dim=100,
                            n_hidden=args.n_hidden, n_kernels=16, out_dim=100)
        elif args.dataset == 'cifar10':
            hnet = CNNHyper(args.n_parties, embed_dim, hidden_dim=100,
                            n_hidden=args.n_hidden, n_kernels=16, out_dim=10)

    return hnet


def tsne_dot_with_legend(feat_narray, gt_narray, components=2, vis=False):
    # feat = np.load("knn-per50_cifar100_dir03_feature.npy")
    # t_gt = np.load("knn-per50_cifar100_dir03_label.npy")
    feat = feat_narray
    t_gt = gt_narray
    # Y = TSNE(n_components=2).fit_transform(feat)
    # y_min, y_max = np.min(Y, 0), np.max(Y, 0)
    # Y = (Y - y_min) / (y_max - y_min)
    Y_t = TSNE(n_components=components).fit_transform(feat)
    y_min, y_max = np.min(Y_t, 0), np.max(Y_t, 0)
    Y_t = (Y_t - y_min) / (y_max - y_min)

    if vis:
        fig = plt.figure(1,figsize=(9, 9))

        t_feat_Y = Y_t 
        t_label_set = np.unique(t_gt)
        # label2index_map_t = {t_label_set[i]:i-10 for i in range(10,21)}
        label2index_map_t = {t_label_set[i]:i for i in range(len(t_label_set))}
        # cmap = mpl.cm.get_cmap("plasma", 21)


        # cmap = mpl.cm.get_cmap("plasma", 20)
        cmap = mpl.cm.get_cmap("tab20")
        color_list = ['#00FFFF', "#7FFFD4", "#000000", '#0000FF', '#008000',
                      '#8A2BE2', '#A52A2A', '#DEB887', '#5F9EA0', '#D2691E',
                      '#FF7F50', '#DC143C', '#00FFFF', '#00008B', '#B8860B',
                      '#A9A9A9', '#006400', '#FF0000', '#FFFF00', '#FFA500']
        scatt_list = []
        for i in range(20):
            sc = plt.scatter(t_feat_Y[5*i:5*(i+1), 0], t_feat_Y[5*i:5*(i+1), 1] ,20 , [color_list[i] for _ in range(5)], 'o' )
            scatt_list.append(sc)
        # scatter1 = plt.scatter(t_feat_Y[:, 0], t_feat_Y[:, 1] ,20 , t_gt, 'o', cmap=cmap)
        # label_list = ['aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables', 
        #               'household electrical devices', 'household furniture', 'insects', 'large carnivores', 'large man-made outdoor scenes', 
        #               'large omnivores and herbivores', 'medium-sized mammals', 'non-insect invertebrates', 'people', 
        #               'reptiles', 'small mammals', 'trees', 'vehicles 1', 'vehicles 2']
        # plt.legend(scatt_list, label_list, title="coarse classes", bbox_to_anchor=(0.04,0.61))

        plt.xticks([])
        plt.yticks([])   

        ax = plt.gca() #获取当前坐标轴
        # 把右边和上边的边框去掉
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['bottom'].set_color('none')

        plt.show() 
        plt.savefig('dot_image_cnn_nolegend.png')
    return Y_t


if __name__ == '__main__':
    args = get_args()
    logging.info("Dataset: %s" % args.dataset)
    logging.info("Backbone: %s" % args.model)
    logging.info("Method: %s" % args.alg)
    logging.info("Partition: %s" % args.partition)
    logging.info("Version: %d" % args.version)
    logging.info("Beta: %f" % args.beta)
    logging.info("Sample rate: %f" % args.sample)
    logging.info("Print Accuracy on training set: %s" % args.train_acc_pre)
    logging.info("Save model: %s" % args.save_model)
    logging.info("Total running round: %s" % args.comm_round)
    logging.info("Test round fequency: %d" % args.eval_step)
    logging.info("Noise Type: %s" %args.noise_type)
    logging.info("Show every client's accuracy: %s" %args.show_all_accuracy)
    if args.noise_type != 'None':
        if args.partition != 'homo':
            raise NotImplementedError("Noise based feature skew only supports iid partition")
        logging.info("Max Noise: %d" %args.noise)
    if args.model in ["vit", "transformer"]:
        logging.info("Use Layer_embedding: %s" %args.layer_emd)
        logging.info("Transformer depth: %d" % args.depth)
        if args.embed_extractor == "hyperVit":
            logging.info("Hyper hidden dimension: %d" % args.hyper_hid)
            logging.info("Client embedding size: %d" %args.client_embed_size)
            logging.info("Use balance soft-max: %s" %args.balanced_soft_max)

    save_path = args.embed_extractor+"-"+args.model+"-"+str(args.n_parties)+"-"+args.dataset+"-"+args.partition+args.comment
    mkdirs(args.modeldir)
    device = torch.device(args.device)

    mkdirs(args.logdir)
    if args.log_file_name is None:
        argument_path= args.embed_extractor + " " + args.model + " " + str(args.version) + '-experiment_arguments-%s.json ' % datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
    else:
        argument_path=args.log_file_name+'.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)

    if args.log_file_name is None:
        args.log_file_name = args.model + " " + str(args.version) + '-experiment_log-%s ' % (datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S"))
    log_path=args.log_file_name+'.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.INFO, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(device)

    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    eval_step = args.eval_step
    acc_all = []

    logger.info("Partitioning data")
    logging.info("Client Number: %d" % args.n_parties)
    X_train, y_train, X_test, y_test, net_dataidx_map_train, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts = partition_data(
        args.dataset, args.datadir, args.partition, args.n_parties, beta=args.beta)


    logger.info("Initializing nets")
    nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
    global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
    global_model = global_models[0]

    logger.info("Initializing hyper")
    hnet = init_hyper(args).to(device)

    logger.info("Loading model")
    save_path = Path("results_table/"+save_path)
    if args.embed_extractor == 'hyperVit':
        outfile_vit = os.path.join(save_path, 'Vit_1500.tar')
    elif args.embed_extractor == 'hyperCnn':
        outfile_vit = os.path.join(save_path, 'cnn_1500.tar')
    outfile_hyp = os.path.join(save_path, 'HY_1500.tar')

    
    logger.info("Load: %s" %outfile_vit)
    tmp_vit = torch.load(outfile_vit)

    logger.info("Load: %s" %outfile_hyp)
    tmp_hyp = torch.load(outfile_hyp)

    vit_para = tmp_vit['state']
    global_model.load_state_dict(vit_para)
    hyp_para = tmp_hyp['state']
    hnet.load_state_dict(hyp_para)

    logging.info("Test beginning round: %d" %args.test_round)
    logging.info("Personailze each client's model")
    
    hnet.eval()
    embed_arr = None
    client_label_hash = []
    client_embedding_hash = 0
    ldx = 0
    similarity_matrix = []
    with torch.no_grad():
        for net_id in range(args.n_parties):        
            map_embedding = hnet.embeddings(torch.tensor([net_id], dtype=torch.long).to(device))
            client_embedding = hnet.mlp(map_embedding)
            client_label_hash.append(ldx)
            if net_id==0:
                client_embedding_hash = copy.deepcopy(client_embedding)
            else:
                client_embedding_hash = torch.cat([client_embedding_hash, client_embedding], 0)
            # print(client_embedding_hash)
            if (net_id+1)%5==0:
                ldx += 1
            node_weights = hnet(torch.tensor([net_id], dtype=torch.long).to(device))
            nets[net_id].load_state_dict(node_weights)
        
        #tsne, reduce embedding dimension
        label_arr = np.asarray(client_label_hash)
        embed_arr = client_embedding_hash.cpu().numpy()
        client_embedding_hash = torch.tensor(tsne_dot_with_legend(embed_arr, label_arr, components=3, vis=False))
        print(client_embedding_hash)
        # assert False

        client_embedding_copy = copy.deepcopy(client_embedding_hash)
        # print(client_embedding_copy.shape)
        # print(client_embedding.shape)
        n = client_embedding_hash.size(0)
        d = client_embedding_hash.size(1)
        # print(client_embedding_hash)
        # print(client_embedding_copy)
        client_embedding_copy = client_embedding_copy.t()
        # print(client_embedding_copy.shape)
        # print(client_embedding_hash.shape)
        # print(client_embedding_hash)
        # print(client_embedding_copy)
        # assert False
        inner_product = torch.mm(client_embedding_hash, client_embedding_copy)
        # print(inner_product)
        client_embedding_mode = (torch.sqrt(torch.pow(client_embedding_hash, 2).sum(1))).view(1,-1)
        client_embedding_modev = copy.deepcopy(client_embedding_mode).view(-1,1)
        # print(client_embedding_mode)
        # print(client_embedding_modev)
        inner_mode = torch.mm(client_embedding_modev, client_embedding_mode)
        # print(inner_mode)
        cosine_distance = torch.div(inner_product, inner_mode)
        print("Before normalize: ")
        print(cosine_distance[0])
        # # assert False
        if args.sample_part == False:
            row_sum = cosine_distance.sum(1)
            cosine_distance = torch.div(cosine_distance, row_sum)
        print("After normalize:  ")
        print(cosine_distance[0])
    # assert False        

    results_dict = defaultdict(list)
    eval_step = args.eval_step
    best_step = 0
    best_accuracy = -1
    test_round = args.test_round
    if args.sample_part:
        sample_rate = args.sample
    else:
        sample_rate = 1.0

    if args.alg == 'avg':
        total_data_points = sum([len(net_dataidx_map_train[r]) for r in range(args.n_parties)])
        fed_avg_freqs = [len(net_dataidx_map_train[r]) / total_data_points for r in range(args.n_parties)]
        global_para = global_model.state_dict()        
        for idx in range(args.n_parties):
            net_para = nets[idx].state_dict()
            if idx == 0:
                for key in net_para:
                    global_para[key] = net_para[key] * fed_avg_freqs[idx]
            else:
                for key in net_para:
                    global_para[key] += net_para[key] * fed_avg_freqs[idx]
        global_model.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round: %d" %round)

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * sample_rate)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net_per(nets, selected, args, net_dataidx_map_train, net_dataidx_map_test, logger, device=device)

            # update global model
            total_data_points = sum([len(net_dataidx_map_train[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map_train[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)

            if (round+1)>=test_round and (round+1)%eval_step == 0:
                if args.dataset == 'shakespeare':
                    train_results, train_avg_loss, train_acc, train_all_acc, test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_per_client_simple(
                    global_model, args, train_dl_global, test_dl_global, nets, device=device)
                else:
                    train_results, train_avg_loss, train_acc, train_all_acc, test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_per_client_simple(
                    global_model, args, net_dataidx_map_train, net_dataidx_map_test, nets, device=device)

                if args.log_flag:
                    logger.info('>> Global Model Train accuracy: %f' % train_acc)
                    logger.info('>> Global Model Test accuracy: %f' % test_acc)
                    logger.info('>> Test avg loss: %f' %test_avg_loss)

                results_dict['train_avg_loss'].append(train_avg_loss)
                results_dict['train_avg_acc'].append(train_acc)
                results_dict['test_avg_loss'].append(test_avg_loss)
                results_dict['test_avg_acc'].append(test_acc*100)
        print("test_all_acc: ", test_all_acc)
        # save_path = Path("results_table/"+save_path)
        # save_path.mkdir(parents=True, exist_ok=True)
  
        # accessories = args.alg + "-" + str(args.n_parties) + "-" + str(args.dataset) + "-" + args.partition + "-" + args.comment

        # if args.save_model:
        #     logger.info("Saving model")
        #     outfile_gmodel = os.path.join(save_path, 'gmodel_1500.tar')
        #     torch.save({'epoch':args.comm_round+1, 'state':global_model.state_dict()}, outfile_gmodel)

        # json_file_opt = "results_"+accessories+".json"
        # with open(str(save_path / json_file_opt), "w") as file:
        #     json.dump(results_dict, file, indent=4)

    elif args.alg == 'per':
        for round in range(args.comm_round):
            logger.info("in comm round: %d" % round)

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * sample_rate)]
            local_train_net_per(nets, selected, args, net_dataidx_map_train, net_dataidx_map_test, logger, device=device)

            # update global model
            total_data_points = sum([len(net_dataidx_map_train[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map_train[r]) / total_data_points for r in selected]

            if (round+1)>=test_round and (round+1)%eval_step == 0:
                train_results, train_avg_loss, train_acc, train_all_acc, test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_local(
                nets, args, net_dataidx_map_train, net_dataidx_map_test, device=device)

                if args.log_flag:
                    logger.info('>> Global Model Train accuracy: %f' % train_acc)
                    logger.info('>> Global Model Test accuracy: %f' % test_acc)
                    logger.info('>> Test avg loss: %f' %test_avg_loss)

                results_dict['train_avg_loss'].append(train_avg_loss)
                results_dict['train_avg_acc'].append(train_acc)
                results_dict['test_avg_loss'].append(test_avg_loss)
                results_dict['test_avg_acc'].append(test_acc*100)
        print("test_all_acc: ", test_all_acc)

    elif args.alg == 'peer':
        for round in range(args.comm_round):
            logger.info("in comm round: %d" % round)

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties)]
            local_train_net_per(nets, selected, args, net_dataidx_map_train, net_dataidx_map_test, logger, device=device)

            back_nets = {}
            for idx in range(len(selected)):
                net_para = copy.deepcopy(nets[selected[idx]].state_dict())
                fed_avg_freqs = cosine_distance[selected[idx]]
                if args.sample_part:
                    sorted_sim, indice_sim = torch.sort(fed_avg_freqs, descending=True)
                    # print(fed_avg_freqs)
                    # print(sorted_sim)
                    # print(indice_sim)
                    fed_avg_freqs = [sorted_sim[x] for x in range(int(args.n_parties*sample_rate))]
                    # print(fed_avg_freqs)
                    fed_sum = sum(fed_avg_freqs)
                    # print(fed_sum)
                    fed_avg_freqs = [x/fed_sum for x in fed_avg_freqs]
                    # print(fed_avg_freqs)
                    # assert False
                    for ne in range(int(args.n_parties*sample_rate)):
                        nei = int(indice_sim[ne])
                        nei_para = copy.deepcopy(nets[nei].state_dict())
                        if ne == 0:
                            for key in net_para:
                                net_para[key] = nei_para[key] * fed_avg_freqs[ne]
                        else:
                            for key in net_para:
                                net_para[key] += nei_para[key] * fed_avg_freqs[ne]
                else:
                    for nei in range(args.n_parties):
                        nei_para = nets[nei].state_dict()
                        if nei == 0:
                            for key in net_para:
                                net_para[key] = nei_para[key] * fed_avg_freqs[nei]
                        else:
                            for key in net_para:
                                net_para[key] += nei_para[key] * fed_avg_freqs[nei]

                back_nets[selected[idx]] = net_para
            for kk in back_nets.keys():
                nets[kk].load_state_dict(back_nets[kk])

            if (round+1)>=test_round and (round+1)%eval_step == 0:
                train_results, train_avg_loss, train_acc, train_all_acc, test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_local(
                nets, args, net_dataidx_map_train, net_dataidx_map_test, device=device)
            
                if args.log_flag:
                    logger.info('>> Global Model Train accuracy: %f' % train_acc)
                    logger.info('>> Global Model Test accuracy: %f' % test_acc)
                    logger.info('>> Test avg loss: %f' %test_avg_loss)

                results_dict['train_avg_loss'].append(train_avg_loss)
                results_dict['train_avg_acc'].append(train_acc)
                results_dict['test_avg_loss'].append(test_avg_loss)
                results_dict['test_avg_acc'].append(test_acc*100)
        print("test_all_acc: ", test_all_acc)
        # save_path = Path("results_table/"+save_path)
        # save_path.mkdir(parents=True, exist_ok=True)

        # accessories = args.embed_extractor + "-" + str(args.n_parties) + "-" + str(args.dataset) + "-" + args.comment
        
        # if args.save_model:
        #     logger.info("Saving model")
        #     outfile_hp = os.path.join(save_path,  'HY_1500.tar')
        #     outfile_vit = os.path.join(save_path, 'cnn_1500.tar')
        #     torch.save({'epoch':args.comm_round+1, 'state':hnet.state_dict()}, outfile_hp)
        #     torch.save({'epoch':args.comm_round+1, 'state':global_model.state_dict()}, outfile_vit)

        # json_file_opt = "results_"+accessories+".json"
        # with open(str(save_path / json_file_opt), "w") as file:
        #     json.dump(results_dict, file, indent=4)

    elif args.alg == 'relation':
        Relation_net = Relation_net(client_embedding_hash).to(device)
        optimizer = torch.optim.SGD(params=Relation_net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
        for round in range(args.comm_round):
            logger.info("in comm round: %d" % round)

            Relation_net.train()
            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties)]
            back_nets = {}
            weights = {}
            for idx in range(len(selected)):
                net_para = copy.deepcopy(nets[selected[idx]].state_dict())
                # print(client_embedding_hash.shape)
                learned_embedding = client_embedding_hash[idx]
                # print(learned_embedding)
                fed_avg_freqs = Relation_net(learned_embedding)
                # Rnet_grads = torch.autograd.grad(fed_avg_freqs, Relation_net.parameters(), grad_outputs=torch.ones_like(fed_avg_freqs), retain_graph=True)
                # print(Rnet_grads[0])
                # if idx==2:
                #     assert False
                # weights[selected[idx]] = fed_avg_freqs

                # print(fed_avg_freqs)
                # assert False
                if args.sample_part:
                    sorted_sim, indice_sim = torch.sort(fed_avg_freqs, descending=True)
                    fed_avg_freqs = [sorted_sim[x] for x in range(int(args.n_parties*sample_rate))]
                    fed_sum = sum(fed_avg_freqs)
                    fed_avg_freqs = [x/fed_sum for x in fed_avg_freqs]
                    for ne in range(int(args.n_parties*sample_rate)):
                        nei = int(indice_sim[ne])
                        nei_para = copy.deepcopy(nets[nei].state_dict())
                        if ne == 0:
                            for key in net_para:
                                net_para[key] = nei_para[key] * fed_avg_freqs[ne]
                        else:
                            for key in net_para:
                                net_para[key] += nei_para[key] * fed_avg_freqs[ne]
                else:
                    for nei in range(args.n_parties):
                        nei_para = nets[nei].state_dict()
                        if nei == 0:
                            for key in net_para:
                                # print(nei_para[key])
                                # print(fed_avg_freqs[nei])
                                net_para[key] = nei_para[key] * fed_avg_freqs[nei]
                                # print(net_para[key])
                        else:
                            for key in net_para:
                                net_para[key] += nei_para[key] * fed_avg_freqs[nei]
                # assert False
                back_nets[selected[idx]] = net_para
                weights[selected[idx]] = net_para
            for kk in back_nets.keys():
                nets[kk].load_state_dict(back_nets[kk])

            local_train_net_per(nets, selected, args, net_dataidx_map_train, net_dataidx_map_test, logger, device=device)

            for idx in range(len(selected)):
                final_state = nets[selected[idx]].state_dict()
                node_weights = weights[selected[idx]]
                inner_state = back_nets[selected[idx]]
                delta_theta = OrderedDict({k: inner_state[k] - final_state[k] for k in inner_state.keys()})
                # bp_vec = torch.ones_like(node_weights)
                # print(delta_theta)
                # print(list(node_weights))
                Rnet_grads = torch.autograd.grad(list(node_weights), Relation_net.parameters(), grad_outputs=list(delta_theta.values()), retain_graph=True)
                print(Rnet_grads)
                # print(len(Rnet_grads))
                if idx == 1:
                    assert False
                if idx == 0:
                    grads_update = [x for x in Rnet_grads]
                else:
                    for g in range(len(Rnet_grads)):
                        grads_update[g] += Rnet_grads[g]

            optimizer.zero_grad()
            for p, g in zip(Relation_net.parameters(), grads_update):
                p.grad = g
            optimizer.step()

            if (round+1)>=test_round and (round+1)%eval_step == 0:
                train_results, train_avg_loss, train_acc, train_all_acc, test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_local(
                nets, args, net_dataidx_map_train, net_dataidx_map_test, device=device)
            
                if args.log_flag:
                    logger.info('>> Global Model Train accuracy: %f' % train_acc)
                    logger.info('>> Global Model Test accuracy: %f' % test_acc)
                    logger.info('>> Test avg loss: %f' %test_avg_loss)

                results_dict['train_avg_loss'].append(train_avg_loss)
                results_dict['train_avg_acc'].append(train_acc)
                results_dict['test_avg_loss'].append(test_avg_loss)
                results_dict['test_avg_acc'].append(test_acc*100)

        print("test_all_acc: ", test_all_acc)
    acc_all  = np.asarray(results_dict['test_avg_acc'])
    logger.info("Accuracy Record: ")
    logger.info(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std  = np.std(acc_all)
    logger.info('Test Acc = %4.2f%% +- %4.2f%%' %(acc_mean, acc_std))
    if args.show_all_accuracy:
        logger.info("Accuracy in each client: ")
        logger.info(results_dict['test_all_acc'])