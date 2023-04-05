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

from models.vit import ViT
from models.Hypernetworks import ViTHyper, ProtoHyper, ShakesHyper, Layer_ViTHyper, Layer_ShakesHyper
from models.cnn import CNNHyper, CNNTarget, CNN_B
from models.knn_per import Mobilenet, NextCharacterLSTM
from models.language_transformer import Transformer
from model import *
from utils import *
from vggmodel import *
from resnetcifar import *
from methods.method import *

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
    parser.add_argument('--alg', type=str, default='hyperVit',
                            help='communication strategy: fedavg/fedprox')
    parser.add_argument('--comm_round', type=int, default=1500, help='number of maximum communication roun')
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
    parser.add_argument('--eval_step', type=int, default=5)
    parser.add_argument('--test_round', type=int, default=1300)
    parser.add_argument("--save_model", action='store_true')
    parser.add_argument("--comment", default="_")
    parser.add_argument("--definite_selection", action='store_true')
    parser.add_argument("--show_all_accuracy", action='store_true')

    """
    Used for hyperVit
    """
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--hyper_hid', type=int, default=100, help="hypernet hidden dim")
    parser.add_argument("--n-hidden", type=int, default=3, help="num. hidden layers")
    parser.add_argument('--layer_emd', action='store_true')
    parser.add_argument("--balanced_soft_max", action='store_true')
    parser.add_argument("--client_embed_size", type=int, default=128)

    """
    Used for protoVit 
    """
    parser.add_argument('--version', type=int, default=7)
    parser.add_argument('--log_flag', default=True)
    parser.add_argument('--beginning_round', type=int, default=50)
    parser.add_argument('--update_round', type=int, default=20)
    parser.add_argument('--calibrated', action='store_true')
    parser.add_argument('--lambda_value', type=float, default=0.5)
    parser.add_argument('--no_mlp_head', action='store_true')
    parser.add_argument('--position_embedding', action='store_true')
    parser.add_argument('--k_neighbor', action='store_true')
    parser.add_argument('--similarity', action='store_true')

    """
    Used for knn-per
    """
    parser.add_argument('--capacity_ratio', type=float, default=1.0)
    parser.add_argument('--k_value', default=10)
    parser.add_argument('--knn_weight', type=float, default=0.6)

    """
    Used for shakespeare
    """
    parser.add_argument('--chunk_len', type=int, default=5)

    """
    Used for pfedMe
    pfedMe alpha is inner loop learning rate, here we let is same to our learning rate
    """
    parser.add_argument('--pfedMe_k', type=int, default=5)
    parser.add_argument('--pfedMe_lambda', type=float, default=15)
    parser.add_argument('--pfedMe_beta', type=float, default=1)
    parser.add_argument('--pfedMe_mu', type=float, default=0)

    """
    Used for fedRod
    """
    parser.add_argument('--use_hyperRod', action='store_true')

    """
    Used for fedproto
    standard deviation: 2; rounds: 110; weight of proto loss: 0.1 
    local_bs 32

    """
    parser.add_argument('--fedproto_lambda', default=0.1)

    args = parser.parse_args()
    return args


def init_nets(net_configs, dropout_p, n_parties, args):

    nets = {net_i: None for net_i in range(n_parties)}
    device = torch.device(args.device)

    for net_i in range(n_parties):
        if args.dataset == "generated":
            net = PerceptronModel()
        
        elif args.model == "mlp":
            if args.dataset == 'covtype':
                input_size = 54
                output_size = 2
                hidden_sizes = [32,16,8]
            elif args.dataset == 'a9a':
                input_size = 123
                output_size = 2
                hidden_sizes = [32,16,8]
            elif args.dataset == 'rcv1':
                input_size = 47236
                output_size = 2
                hidden_sizes = [32,16,8]
            elif args.dataset == 'SUSY':
                input_size = 18
                output_size = 2
                hidden_sizes = [16,8]
            net = FcNet(input_size, hidden_sizes, output_size, dropout_p)
        
        elif args.model == "vgg":
            net = vgg11()
        
        elif args.model == "simple-cnn":
            if args.dataset in ("cifar10", "cinic10", "svhn"):
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
            elif args.dataset in ("cifar100"):
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=100)
            elif args.dataset in ("mnist", 'femnist', 'fmnist'):
                net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
            elif args.dataset == 'celeba':
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=2)
        
        elif args.model == "vit":
            if args.dataset == "cifar10":
                net = ViT(image_size = 32, patch_size = 4, num_classes = 10, dim = 128, depth = args.depth, heads = 8, mlp_dim = 256,
                  dropout = 0.1, emb_dropout = 0.1)      
            elif args.dataset == "cifar100":
                net = ViT(image_size=32, patch_size=4, num_classes=100, dim=128, depth = args.depth, heads=8, mlp_dim=256,
                          dropout=0.1, emb_dropout=0.1)

        elif args.model == "cnn":
            if args.dataset == "cifar10":
                net = CNNTarget(n_kernels=16)
            elif args.dataset == "cifar100":
                net = CNNTarget(n_kernels=16, out_dim=100)

        elif args.model == "cnn-b":
            if args.dataset == "cifar10":
                net = CNN_B(n_kernels=16)
            elif args.dataset == "cifar100":
                net = CNN_B(n_kernels=16, out_dim=100)

        elif args.model == "vgg-9":
            if args.dataset in ("mnist", 'femnist'):
                net = ModerateCNNMNIST()
            elif args.dataset in ("cifar10", "cinic10", "svhn"):
                net = ModerateCNN()
            elif args.dataset == 'celeba':
                net = ModerateCNN(output_dim=2)
        
        elif args.model == "resnet":
            net = ResNet50_cifar10()

        elif args.model == "vgg16":
            net = vgg16()

        elif args.model == "mobilent":
            if args.dataset == "cifar10":
                net = Mobilenet(n_classes=10, pretrained=True)
            elif args.dataset == "cifar100":
                net = Mobilenet(n_classes=100, pretrained=True)

        elif args.model == "lstm":
            # "input_size": len(string.printable),
            # "embed_size": 8, "hidden_size": 256,
            # "output_size": len(string.printable),
            # "n_layers": 2, "chunk_len": 80
            net = NextCharacterLSTM(
                input_size=SHAKESPEARE_CONFIG["input_size"],
                embed_size=SHAKESPEARE_CONFIG["embed_size"],
                hidden_size=SHAKESPEARE_CONFIG["hidden_size"],
                output_size=SHAKESPEARE_CONFIG["output_size"],
                n_layers=SHAKESPEARE_CONFIG["n_layers"])

        elif args.model == "transformer":
            # -b 256 -warmup 128000 -epoch 400
            # '-d_model', type=int, default=512
            # '-d_inner_hid', type=int, default=2048
            # '-d_k', type=int, default=64
            # '-d_v', type=int, default=64
            # '-n_head', type=int, default=8
            # '-n_layers', type=int, default=6
            # 'scale_emb_or_prj', type=str, default='prj'
            net = Transformer(n_src_vocab=len(string.printable), 
            n_trg_vocab=len(string.printable),
            d_k=64, d_v=64, d_model=128,
            d_word_vec=128, d_inner=256,
            n_layers=args.depth, n_head=8, dropout=0.1)

        else:
            raise NotImplementedError("not supported yet")
            # print("not supported yet")
            # exit(1)
        nets[net_i] = net.to(device)

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type


def init_hyper(args, sam_node=None):
    # embed_dim = int(1 + args.n_parties / 4) 
    embed_dim = args.client_embed_size
    batch_node = int(args.n_parties * args.sample)
    if args.model == "vit":
        if args.layer_emd:
            if args.dataset == "cifar10":
                hnet = Layer_ViTHyper(args, args.n_parties, embed_dim, hidden_dim=args.hyper_hid, dim=128,
                                heads=8, dim_head=64, n_hidden=args.n_hidden, depth=args.depth,
                                client_sample=batch_node)

            elif args.dataset == "cifar100":
                hnet = Layer_ViTHyper(args, args.n_parties, embed_dim, hidden_dim=args.hyper_hid, dim=128,
                                heads=8, dim_head=64, n_hidden=args.n_hidden, depth=args.depth,
                                client_sample=batch_node)
        else:
            if args.alg == "protoVit":
                if args.dataset == "cifar10":
                    hnet = ProtoHyper(args.n_parties, args=args, hidden_dim=args.hyper_hid, dim=128, 
                              heads = 8, dim_head = 64, n_hidden=args.n_hidden, depth=args.depth, client_sample=batch_node)
            
                elif args.dataset == "cifar100":
                    hnet = ProtoHyper(args.n_parties, args=args, hidden_dim=args.hyper_hid, dim=128,
                                heads=8, dim_head=64, n_hidden=args.n_hidden, depth=args.depth, client_sample=batch_node)

            else:
                if args.dataset == "cifar10":
                    hnet = ViTHyper(args.n_parties, embed_dim, hidden_dim = args.hyper_hid, dim=128, 
                              heads = 8, dim_head = 64, n_hidden = args.n_hidden, depth=args.depth, client_sample=batch_node)
                
                elif args.dataset == "cifar100":
                    hnet = ViTHyper(args.n_parties, embed_dim, hidden_dim = args.hyper_hid, dim=128,
                                heads=8, dim_head=64, n_hidden = args.n_hidden, depth=args.depth, client_sample=batch_node)
    
    elif args.model == "cnn":
        if args.dataset == "cifar10":
            hnet = CNNHyper(args.n_parties, embed_dim, hidden_dim=args.hyper_hid, n_hidden=args.n_hidden, n_kernels=16)
        
        elif args.dataset == "cifar100":
            hnet = CNNHyper(args.n_parties, embed_dim, hidden_dim=args.hyper_hid,
                            n_hidden=args.n_hidden, n_kernels=16, out_dim=100)

    elif args.model == "transformer":
        if args.dataset != "shakespeare":
            raise NotImplementedError("ShakesHyper only supports shakespeare dataset.")
        if args.layer_emd:
            hnet = Layer_ShakesHyper(args.n_parties, embed_dim, hidden_dim = args.hyper_hid, dim=128, 
            heads = 8, dim_head = 64, n_hidden = args.n_hidden, 
            depth=args.depth, client_sample=sam_node, device=args.device)
        else:
            hnet = ShakesHyper(args.n_parties, embed_dim, hidden_dim = args.hyper_hid, dim=128, 
            heads = 8, dim_head = 64, n_hidden = args.n_hidden, 
            depth=args.depth, client_sample=sam_node)

    return hnet


def init_personalized_parameters(args, client_number=None):
    personalized_pred_list = []
    if args.dataset == "cifar10":
        class_num = 10
    elif args.dataset == "cifar100":
        class_num = 100
    elif args.dataset == "shakespeare":
        class_num = 100

    if args.alg == 'perVit' or args.alg == 'protoVit':
        if args.model == 'vit':
            for nndx in range(args.n_parties):
                kqv_dict = OrderedDict()
                for ll in range(args.depth):
                    kqv_dict["transformer.layers."+str(ll)+".0.fn.to_qkv.weight"]=None
                personalized_pred_list.append(kqv_dict)
        elif args.model == 'transformer':
            for nndx in range(client_number):
                kqv_dict = OrderedDict()
                for ll in range(args.depth):
                    kqv_dict["encoder.layer_stack."+str(ll)+".slf_attn.w_qs.weight"]=None
                    kqv_dict["encoder.layer_stack."+str(ll)+".slf_attn.w_ks.weight"]=None
                    kqv_dict["encoder.layer_stack."+str(ll)+".slf_attn.w_vs.weight"]=None
                personalized_pred_list.append(kqv_dict)

    elif args.alg == 'fedRod':
        if args.model == "cnn":
            dim = 84
            for nndx in range(args.n_parties):
                if args.use_hyperRod:
                    p_class = nn.Linear(class_num, class_num*(dim+1)).to(args.device)
                else:
                    p_class = nn.Linear(dim, class_num).to(args.device)
                personalized_pred_list.append(p_class)
        elif args.model in ["vit", "transformer"]:
            dim = 128
            for nndx in range(args.n_parties):
                if args.use_hyperRod:
                    p_class = nn.Linear(class_num, class_num*(dim+1)).to(args.device)
                else:
                    p_class = nn.Linear(dim, class_num).to(args.device)
                personalized_pred_list.append(p_class)

    elif args.alg == 'fedPer' or args.alg == 'proto_cluster':
        if args.model == 'cnn':
            dim = 84
            for nndx in range(args.n_parties):
                para_dict = OrderedDict()
                # w_value = torch.zeros(class_num, dim).to(args.device)
                # b_value = torch.zeros(class_num).to(args.device)
                para_dict["fc3.weight"] = None
                para_dict["fc3.bias"] = None
                personalized_pred_list.append(para_dict)
        elif args.model == 'vit':
            dim = 128
            for nndx in range(args.n_parties):
                para_dict = OrderedDict()
                # w_value = torch.zeros(class_num, dim).to(args.device)
                # b_value = torch.zeros(class_num).to(args.device)
                para_dict["mlp_head.1.weight"] = None
                para_dict["mlp_head.1.bias"] = None
                personalized_pred_list.append(para_dict)
        elif args.model == 'transformer':
            dim = 128
            for nndx in range(client_number):
                para_dict = OrderedDict()
                # w_value = torch.zeros(class_num, dim).to(args.device)
                para_dict["trg_word_prj.weight"] = None
                personalized_pred_list.append(para_dict)
        elif args.model == 'lstm':
            dim = 256
            for nndx in range(client_number):
                para_dict = OrderedDict()
                # w_value = torch.zeros(class_num, dim).to(args.device)
                # b_value = torch.zeros(class_num).to(args.device)
                para_dict["decoder.weight"] = None
                para_dict["decoder.bias"] = None
                personalized_pred_list.append(para_dict)

    elif args.alg in ['fedBN', 'fedAP', 'pfedKL']:
        for nndx in range(args.n_parties):
            bn_dict = OrderedDict()
            for ll in range(4):
                bn_dict["bn"+str(ll+1)+".weight"]=None
                bn_dict["bn"+str(ll+1)+".bias"]=None
            personalized_pred_list.append(bn_dict)

    elif args.alg == 'hyperVit-Rod':
        dim = 128
        for nndx in range(args.n_parties):
            p_class = nn.Linear(dim, class_num).to(args.device)
            personalized_pred_list.append(p_class)

    elif args.alg == 'hyperVit-Per':
        dim = 128
        for nndx in range(args.n_parties):
            para_dict = OrderedDict()
            para_dict["mlp_head.1.weight"] = None
            para_dict["mlp_head.1.bias"] = None
            personalized_pred_list.append(para_dict)

    return personalized_pred_list


def view_image(train_dataloader):
    for (x, target) in train_dataloader:
        np.save("img.npy", x)
        print(x.shape)
        exit(0)


def get_partition_dict(dataset, partition, n_parties, init_seed=0, datadir='./data', logdir='./logs', beta=0.5):
    seed = init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    X_train, y_train, X_test, y_test, net_dataidx_map_train, traindata_cls_counts = partition_data(
        dataset, datadir, logdir, partition, n_parties, beta=beta)

    return net_dataidx_map_train


if __name__ == '__main__':
    # torch.set_printoptions(profile="full")
    args = get_args()
    logging.info("Dataset: %s" % args.dataset)
    logging.info("Backbone: %s" % args.model)
    logging.info("Method: %s" % args.alg)
    logging.info("Partition: %s" % args.partition)
    logging.info("Version: %d" % args.version)
    logging.info("Beta: %f" % args.beta)
    logging.info("Sample rate: %f" % args.sample)
    logging.info("Lambda: %f" % args.lambda_value)
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
        if args.alg == "protoVit":
            logging.info("Beginning_round: %d" %args.beginning_round)
            if args.beginning_round<=0:
                raise NotImplementedError("protoVit need at least one time warm up")
            logging.info("Update round: %d" % args.update_round)
            logging.info("Similarity Proto: %s" % args.similarity)
        elif args.alg in ["hyperVit", "hyperVit-Rod"]:
            logging.info("Hyper hidden dimension: %d" % args.hyper_hid)
            logging.info("Client embedding size: %d" %args.client_embed_size)
            logging.info("Use balance soft-max: %s" %args.balanced_soft_max)
    if args.alg == "fedprox":
        logging.info("mu value: %f" %args.mu)
    if args.alg == "fedRod":
        if args.use_hyperRod:
            logging.info("Use hyper Fed-Rod.")
        else:
            logging.info("Use linear Fed-Rod.")
    if args.test_round<=1:
        raise NotImplementedError("test round should be larger than 1")

    save_path = args.alg+"-"+args.model+"-"+str(args.n_parties)+"-"+args.dataset+"-"+args.partition+args.comment
    mkdirs(args.modeldir)
    device = torch.device(args.device)

    if args.calibrated or args.k_neighbor:
        if args.alg not in ['protoVit','hyperVit','fedavg','knn-per']:
            raise NotImplementedError("Calibration and Memory do not support this alg.")
        if args.calibrated and args.k_neighbor:
            raise NotImplementedError("Calibration and Memory can not be applied in the same time.")
        logging.info("Calibrated: %s" % args.calibrated)
        logging.info("lambda: %f" % args.lambda_value)
        logging.info("no_mlp_head: %s" % args.no_mlp_head)
        logging.info("Use memory: %s" % args.k_neighbor)

    mkdirs(args.logdir)
    if args.log_file_name is None:
        argument_path= args.alg + " " + args.model + " " + str(args.version) + '-experiment_arguments-%s.json ' % datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
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
    logging.info("Test beginning round: %d" %args.test_round)
    logging.info("Client Number: %d" % args.n_parties)
    X_train, y_train, X_test, y_test, net_dataidx_map_train, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts = partition_data(
        args.dataset, args.datadir, args.partition, args.n_parties, beta=args.beta, logdir=args.logdir)

    n_classes = len(np.unique(y_train))

    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                                        args.datadir,
                                                                                        args.batch_size,
                                                                                        32)
    logger.info("len train_dl_global: %d"  %len(train_ds_global))
    data_size = len(test_ds_global)

    results_dict = defaultdict(list)
    eval_step = args.eval_step
    best_step = 0
    best_accuracy = -1
    test_round = args.test_round

    if args.alg == 'fedavg':
        logger.info("Initializing nets")

        if args.k_neighbor:
            args.epochs = 1
            args.batch_size = 128

        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()

        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round: %d" %round)

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            if args.dataset == 'shakespeare':
                local_train_net_per(nets, selected, args, train_dl_global, test_dl_global, logger, device=device)
            else:
                local_train_net_per(nets, selected, args, net_dataidx_map_train, net_dataidx_map_test, logger, device=device)

            # update global model
            if args.dataset == 'shakespeare':
                instance_number_per_client = [len(train_dl_global[r].dataset) for r in selected]
                total_data_points = sum(instance_number_per_client)
                fed_avg_freqs = [instance_number_per_client[r] / total_data_points for r in range(len(instance_number_per_client))]
            else:
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
                # train_results, train_avg_loss, train_acc, train_all_acc = compute_accuracy_per_client_simple(global_model, args, net_dataidx_map_train, test=False, device=device)
                # test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_per_client_simple(global_model, args, net_dataidx_map_train, device=device)
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

        save_path = Path("results_table/"+save_path)
        save_path.mkdir(parents=True, exist_ok=True)
  
        if args.k_neighbor:
            accessories = args.alg + "-knn-" + str(args.n_parties) + "-" + str(args.dataset) + "-" + args.partition + "-" + args.comment
        elif args.calibrated:
            accessories = args.alg + "-cal-" + str(args.n_parties) + "-" + str(args.dataset) + "-" + args.partition + "-" + args.comment
        else:
            accessories = args.alg + "-" + str(args.n_parties) + "-" + str(args.dataset) + "-" + args.partition + "-" + args.comment
            
        print("test_all_acc: ", test_all_acc)
        if args.save_model:
            logger.info("Saving model")
            outfile_gmodel = os.path.join(save_path, 'gmodel_1500.tar')
            torch.save({'epoch':args.comm_round+1, 'state':global_model.state_dict()}, outfile_gmodel)

        json_file_opt = "results_"+accessories+".json"
        with open(str(save_path / json_file_opt), "w") as file:
            json.dump(results_dict, file, indent=4)

    elif args.alg == 'hyperVit':
        if args.model not in ["vit", "transformer"]:
            raise NotImplementedError("hyperVit only supports ViT and transformer")

        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)

        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        logger.info("Initializing hyper")
        if args.dataset == "shakespeare":
            sam_node = int(args.n_parties * args.sample)
            hnet = init_hyper(args, sam_node).to(device)
        else:
            hnet = init_hyper(args).to(device)

        optimizers = {
            'sgd': torch.optim.SGD(
                params=hnet.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3
            ),
            'adam': torch.optim.Adam(params=hnet.parameters(), lr=args.lr)
        }
        optimizer = optimizers['sgd']


        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round: %d" %round)

            hnet.train()
            grads_update = []

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]
            weights = hnet(torch.tensor([selected], dtype=torch.long).to(device),False)

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for ix in range(len(selected)):
                        node_weights = weights[ix]
                        idx = selected[ix]
                        nets[idx].load_state_dict(global_para)
                        nets[idx].load_state_dict(node_weights,strict=False)
            else:
                for ix in range(len(selected)):
                    node_weights = weights[ix]
                    idx = selected[ix]
                    nets[idx].load_state_dict(global_para)
                    nets[idx].load_state_dict(node_weights,strict=False)

            if args.dataset == 'shakespeare':
                local_train_net_per(nets, selected, args, train_dl_global, test_dl_global, logger, device=device)
            else:
                local_train_net_per(nets, selected, args, net_dataidx_map_train, net_dataidx_map_test, logger, device=device)

            # update global model
            if args.dataset == 'shakespeare':
                instance_number_per_client = [len(train_dl_global[r].dataset) for r in selected]
                total_data_points = sum(instance_number_per_client)
                fed_avg_freqs = [instance_number_per_client[r] / total_data_points for r in range(len(instance_number_per_client))]
            else:
                total_data_points = sum([len(net_dataidx_map_train[r]) for r in selected])
                fed_avg_freqs = [len(net_dataidx_map_train[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                final_state = nets[selected[idx]].state_dict()
                net_para = nets[selected[idx]].state_dict()

                node_weights = weights[idx]
                inner_state = OrderedDict({k: tensor.data for k, tensor in node_weights.items()})
                delta_theta = OrderedDict({k: inner_state[k] - final_state[k] for k in node_weights.keys()})
                hnet_grads = torch.autograd.grad(
                    list(node_weights.values()), hnet.parameters(), grad_outputs=list(delta_theta.values()), retain_graph=True
                )

                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                    grads_update = [fed_avg_freqs[idx]*x  for x in hnet_grads]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
                    for g in range(len(hnet_grads)):
                        grads_update[g] += fed_avg_freqs[idx] * hnet_grads[g]

            global_model.load_state_dict(global_para)
            optimizer.zero_grad()
            for p, g in zip(hnet.parameters(), grads_update):
                p.grad = g
            optimizer.step()

            if args.dataset == "cifar10":
                num_class = 10
            elif args.dataset == "cifar100":
                num_class = 100

            if (round+1)>=test_round and (round+1)%eval_step == 0:
                # train_results, train_avg_loss, train_acc, train_all_acc = compute_accuracy_per_client(hnet, global_model, args, net_dataidx_map_train,test=False, device=device)
                # test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_per_client(hnet, global_model, args, net_dataidx_map_train, device=device)
                if args.dataset == 'shakespeare':
                    train_results, train_avg_loss, train_acc, train_all_acc, test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_per_client(
                    hnet, nets, global_model, args, train_dl_global, test_dl_global, 0, device=device)
                else:
                    train_results, train_avg_loss, train_acc, train_all_acc, test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_per_client(
                    hnet, nets, global_model, args, net_dataidx_map_train, net_dataidx_map_test, num_class, device=device)
                
                if args.log_flag:
                    logger.info('>> Global Model Train accuracy: %f' % train_acc)
                    logger.info('>> Global Model Test accuracy: %f' % test_acc)
                    logger.info('>> Test avg loss: %f' %test_avg_loss)

                results_dict['train_avg_loss'].append(train_avg_loss)
                results_dict['train_avg_acc'].append(train_acc)
                results_dict['test_all_acc'] = test_all_acc
                results_dict['test_avg_loss'].append(test_avg_loss)
                results_dict['test_avg_acc'].append(test_acc*100)

        save_path = Path("results_table/"+save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        accessories = args.alg + "-" + str(args.n_parties) + "-" + str(args.dataset) + "-" + args.partition + "-" + args.comment
        
        if args.save_model:
            logger.info("Saving model")
            outfile_hp = os.path.join(save_path,  'HY_1500.tar')
            outfile_vit = os.path.join(save_path, 'Vit_1500.tar')
            torch.save({'epoch':args.comm_round+1, 'state':hnet.state_dict()}, outfile_hp)
            torch.save({'epoch':args.comm_round+1, 'state':global_model.state_dict()}, outfile_vit)

        json_file_opt = "results_"+accessories+".json"
        with open(str(save_path / json_file_opt), "w") as file:
            json.dump(results_dict, file, indent=4)

    elif args.alg == 'perVit':
        if args.model not in  ["vit", "transformer"]:
            raise NotImplementedError("perVit only supports ViT and transformer")

        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)

        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]
        global_para = global_model.state_dict()

        logger.info("Initializing Personalized QKV heads")
        if args.dataset == "shakespeare":
            client_number = int(args.n_parties)
            personalized_kqv_list = init_personalized_parameters(args, client_number)
        else:
            personalized_kqv_list = init_personalized_parameters(args)

        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round: %d" % round)

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for ix in range(len(selected)):
                        idx = selected[ix]
                        nets[idx].load_state_dict(global_para)
            else:
                for ix in range(len(selected)):
                    idx = selected[ix]
                    node_weights = personalized_kqv_list[idx]
                    nets[idx].load_state_dict(global_para)
                    nets[idx].load_state_dict(node_weights,strict=False)

            if args.dataset == 'shakespeare':
                local_train_net_per(nets, selected, args, train_dl_global, test_dl_global, logger, device=device)
            else:
                local_train_net_per(nets, selected, args, net_dataidx_map_train, net_dataidx_map_test, logger, device=device)

            # update global model
            if args.dataset == 'shakespeare':
                instance_number_per_client = [len(train_dl_global[r].dataset) for r in selected]
                total_data_points = sum(instance_number_per_client)
                fed_avg_freqs = [instance_number_per_client[r] / total_data_points for r in range(len(instance_number_per_client))]
            else:
                total_data_points = sum([len(net_dataidx_map_train[r]) for r in selected])
                fed_avg_freqs = [len(net_dataidx_map_train[r]) / total_data_points for r in selected]

            if round == 0:
                for iidx in range(args.n_parties):
                    final_state = nets[iidx].state_dict()
                    for ll in range(args.depth):
                        if args.dataset=="shakespeare":
                            personalized_kqv_list[iidx]["encoder.layer_stack."+str(ll)+".slf_attn.w_qs.weight"] = copy.deepcopy(final_state["encoder.layer_stack."+str(ll)+".slf_attn.w_qs.weight"])
                            personalized_kqv_list[iidx]["encoder.layer_stack."+str(ll)+".slf_attn.w_ks.weight"] = copy.deepcopy(final_state["encoder.layer_stack."+str(ll)+".slf_attn.w_ks.weight"])
                            personalized_kqv_list[iidx]["encoder.layer_stack."+str(ll)+".slf_attn.w_vs.weight"] = copy.deepcopy(final_state["encoder.layer_stack."+str(ll)+".slf_attn.w_vs.weight"])
                        else:
                            personalized_kqv_list[iidx]["transformer.layers."+str(ll)+".0.fn.to_qkv.weight"] = copy.deepcopy(final_state["transformer.layers."+str(ll)+".0.fn.to_qkv.weight"])

            for idx in range(len(selected)):
                if round != 0:
                    final_state = copy.deepcopy(nets[selected[idx]].state_dict())
                    for ll in range(args.depth):
                        if args.dataset=="shakespeare":
                            personalized_kqv_list[selected[idx]]["encoder.layer_stack."+str(ll)+".slf_attn.w_qs.weight"] = copy.deepcopy(final_state["encoder.layer_stack."+str(ll)+".slf_attn.w_qs.weight"])
                            personalized_kqv_list[selected[idx]]["encoder.layer_stack."+str(ll)+".slf_attn.w_ks.weight"] = copy.deepcopy(final_state["encoder.layer_stack."+str(ll)+".slf_attn.w_ks.weight"])
                            personalized_kqv_list[selected[idx]]["encoder.layer_stack."+str(ll)+".slf_attn.w_vs.weight"] = copy.deepcopy(final_state["encoder.layer_stack."+str(ll)+".slf_attn.w_vs.weight"])
                        else:
                            # wq, wk, wv = final_state["transformer.layers."+str(ll)+".0.fn.to_qkv.weight"].chunk(3)
                            # if idx == 0 and ll==0:
                            #     print("Net: ", selected[idx])
                            #     print("Level: ",ll+1)
                            #     print("WQ: ", wq)
                            #     print("WK: ", wk)
                            #     print("WV: ", wv)
                            personalized_kqv_list[selected[idx]]["transformer.layers."+str(ll)+".0.fn.to_qkv.weight"] = copy.deepcopy(final_state["transformer.layers."+str(ll)+".0.fn.to_qkv.weight"])
                net_para = nets[selected[idx]].state_dict()

                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]

            global_model.load_state_dict(global_para)

            if (round+1)>=test_round and (round+1)%eval_step == 0:
                # train_results, train_avg_loss, train_acc, train_all_acc = compute_accuracy_personally(personalized_kqv_list, global_model, args, net_dataidx_map_train,test=False, device=device)
                # test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_personally(personalized_kqv_list, global_model, args, net_dataidx_map_train, device=device)
                if args.dataset == "shakespeare":
                    train_results, train_avg_loss, train_acc, train_all_acc, test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_personally(
                    personalized_kqv_list, global_model, args, train_dl_global, test_dl_global, nets, round, device=device)
                else:
                    train_results, train_avg_loss, train_acc, train_all_acc, test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_personally(
                    personalized_kqv_list, global_model, args, net_dataidx_map_train, net_dataidx_map_test, nets, round, device=device)

                if args.log_flag:
                    logger.info('>> Global Model Train accuracy: %f' % train_acc)
                    logger.info('>> Global Model Test accuracy: %f' % test_acc)
                    logger.info('>> Test avg loss: %f' %test_avg_loss)

                results_dict['train_avg_loss'].append(train_avg_loss)
                results_dict['train_avg_acc'].append(train_acc)
                results_dict['test_avg_loss'].append(test_avg_loss)
                results_dict['test_avg_acc'].append(test_acc*100)

        save_path = Path("results_table/"+save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        accessories = args.alg + "-" + str(args.n_parties) + "-" + args.dataset + "-" + args.partition + "-" + args.comment
        
        if args.save_model:
            logger.info("Saving model")
            outfile_vit = os.path.join(save_path, 'Vit_1500.tar')
            torch.save({'epoch':args.comm_round+1, 'state':global_model.state_dict()}, outfile_vit)
            for ele in range(len(personalized_kqv_list)):
                p_qkv = os.path.join(save_path, 'QKV_1500_'+str(ele)+".tar")
                torch.save({'epoch':args.comm_round+1, 'state':personalized_kqv_list[ele]}, p_qkv)

        json_file_opt = "results_"+accessories+".json"
        with open(str(save_path / json_file_opt), "w") as file:
            json.dump(results_dict, file, indent=4)

    elif args.alg == 'hyperCnn':
        if args.model != "cnn":
            raise NotImplementedError("hyperCnn only supports cnn backbone")
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)

        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        logger.info("Initializing hyper")
        hnet = init_hyper(args).to(device)

        optimizers = {
            'sgd': torch.optim.SGD(
                params=hnet.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3
            ),
            'adam': torch.optim.Adam(params=hnet.parameters(), lr=args.lr)
        }
        optimizer = optimizers['sgd']

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round: %d" % round)

            hnet.train()
            grads_update = []

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]
            weights = {}

            if round == 0:
                if args.is_same_initial:
                    for ix in range(len(selected)):
                        idx = selected[ix]
                        node_weights = hnet(torch.tensor([idx], dtype=torch.long).to(device))
                        weights[ix] = node_weights
                        nets[idx].load_state_dict(node_weights)
            else:
                for ix in range(len(selected)):
                    idx = selected[ix]
                    node_weights = hnet(torch.tensor([idx], dtype=torch.long).to(device))
                    weights[ix] = node_weights
                    nets[idx].load_state_dict(node_weights)

            local_train_net_per(nets, selected, args, net_dataidx_map_train, net_dataidx_map_test, logger, device=device)

            # update global model
            total_data_points = sum([len(net_dataidx_map_train[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map_train[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                final_state = nets[selected[idx]].state_dict()

                node_weights = weights[idx]
                inner_state = OrderedDict({k: tensor.data for k, tensor in node_weights.items()})
                delta_theta = OrderedDict({k: inner_state[k] - final_state[k] for k in node_weights.keys()})
                hnet_grads = torch.autograd.grad(
                    list(node_weights.values()), hnet.parameters(), grad_outputs=list(delta_theta.values()), retain_graph=True
                )

                if idx == 0:
                    grads_update = [fed_avg_freqs[idx]*x  for x in hnet_grads]
                else:
                    for g in range(len(hnet_grads)):
                        grads_update[g] += fed_avg_freqs[idx] * hnet_grads[g]

            optimizer.zero_grad()
            for p, g in zip(hnet.parameters(), grads_update):
                p.grad = g
            optimizer.step()

            if (round+1)>=test_round and (round+1)%eval_step == 0:
                # train_results, train_avg_loss, train_acc, train_all_acc = compute_accuracy_percnn_client(hnet, global_model, args, net_dataidx_map_train,test=False, device=device)
                # test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_percnn_client(hnet, global_model, args, net_dataidx_map_train, device=device)
                train_results, train_avg_loss, train_acc, train_all_acc, test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_percnn_client(
                    hnet, global_model, args, net_dataidx_map_train, net_dataidx_map_test, device=device)
            
                if args.log_flag:
                    logger.info('>> Global Model Train accuracy: %f' % train_acc)
                    logger.info('>> Global Model Test accuracy: %f' % test_acc)
                    logger.info('>> Test avg loss: %f' %test_avg_loss)

                results_dict['train_avg_loss'].append(train_avg_loss)
                results_dict['train_avg_acc'].append(train_acc)
                results_dict['test_avg_loss'].append(test_avg_loss)
                results_dict['test_avg_acc'].append(test_acc*100)

        save_path = Path("results_table/"+save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        accessories = args.alg + "-" + str(args.n_parties) + "-" + str(args.dataset) + "-" + args.comment
        print("test_all_acc: ", test_all_acc)
        if args.save_model:
            logger.info("Saving model")
            outfile_hp = os.path.join(save_path,  'HY_1500.tar')
            outfile_vit = os.path.join(save_path, 'cnn_1500.tar')
            torch.save({'epoch':args.comm_round+1, 'state':hnet.state_dict()}, outfile_hp)
            torch.save({'epoch':args.comm_round+1, 'state':global_model.state_dict()}, outfile_vit)

        json_file_opt = "results_"+accessories+".json"
        with open(str(save_path / json_file_opt), "w") as file:
            json.dump(results_dict, file, indent=4)

    elif args.alg == 'pfedMe':
        if args.model not in ['lstm', 'transformer', 'vit', 'cnn']:
            raise NotImplementedError("pfedMe only supports lstm, cnncifar backbone")
        logger.info("Initializing nets")

        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()

        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round: %d" %round)

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            if args.dataset == 'shakespeare':
                update_dict = local_train_net_pfedMe(nets, selected, global_model, args, train_dl_global, test_dl_global, logger, device=device)
            else:
                update_dict = local_train_net_pfedMe(nets, selected, global_model, args, net_dataidx_map_train, net_dataidx_map_test, logger, device=device)

            # update global model
            if args.dataset == 'shakespeare':
                instance_number_per_client = [len(train_dl_global[r].dataset) for r in selected]
                total_data_points = sum(instance_number_per_client)
                fed_avg_freqs = [instance_number_per_client[r] / total_data_points for r in range(len(instance_number_per_client))]
            else:
                total_data_points = sum([len(net_dataidx_map_train[r]) for r in selected])
                fed_avg_freqs = [len(net_dataidx_map_train[r]) / total_data_points for r in selected]

            # prev_global = copy.deepcopy(list(global_model.parameters()))

            for param in global_model.parameters():
                param.data.zero_()

            for idx in range(len(selected)):
                net_para = update_dict[selected[idx]]
                if idx == 0:
                    for param, new_param in zip(global_model.parameters(), net_para):
                        param.data = new_param.data.clone() * fed_avg_freqs[idx]
                else:
                    for param, new_param in zip(global_model.parameters(), net_para):
                        param.data += new_param.data.clone() * fed_avg_freqs[idx]
            # for pre_param, param in zip(prev_global, global_model.parameters()):
            #     param.data = (1 - args.pfedMe_beta)*pre_param.data + args.pfedMe_beta*param.data

            if (round+1)>=test_round and (round+1)%eval_step == 0:
                if args.dataset == 'shakespeare':
                    train_results, train_avg_loss, train_acc, train_all_acc, test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_local(
                    nets, args, train_dl_global, test_dl_global, device=device)
                else:
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

        save_path = Path("results_table/"+save_path)
        save_path.mkdir(parents=True, exist_ok=True)
         
        accessories = args.alg + "-" + str(args.n_parties) + "-" + str(args.dataset) + "-" + args.comment

        if args.save_model:
            logger.info("Saving model")
            # chose_arr = [3,4,5]
            chose_arr = np.arange(args.n_parties)
            for node_idx in chose_arr:
                outfile_vit = os.path.join(save_path, 'Vit_'+str(node_idx)+'_1500.tar')
                torch.save({'epoch':args.comm_round+1, 'state':nets[node_idx].state_dict()}, outfile_vit)

        json_file_opt = "results_"+accessories+".json"
        with open(str(save_path / json_file_opt), "w") as file:
            json.dump(results_dict, file, indent=4)  

    elif args.alg == 'fedPer':
        if args.model not in ['cnn', 'vit', 'transformer', 'lstm']:
            raise NotImplementedError("fedPer uses cnn as backbone")
        logger.info("Initializing nets")

        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()

        logger.info("Initializing Personalized Classification head")
        if args.dataset == "shakespeare":
            client_number = int(args.n_parties)
            personalized_pred_list = init_personalized_parameters(args, client_number)
        else:
            personalized_pred_list = init_personalized_parameters(args)

        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round: %d" %round)

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    node_weights = personalized_pred_list[idx]
                    nets[idx].load_state_dict(global_para)
                    nets[idx].load_state_dict(node_weights, strict=False)

            if args.dataset == 'shakespeare':
                local_train_net_per(nets, selected, args, train_dl_global, test_dl_global, logger, device=device)
            else:
                local_train_net_per(nets, selected, args, net_dataidx_map_train, net_dataidx_map_test, logger, device=device)

            # update global model
            if args.dataset == 'shakespeare':
                instance_number_per_client = [len(train_dl_global[r].dataset) for r in selected]
                total_data_points = sum(instance_number_per_client)
                fed_avg_freqs = [instance_number_per_client[r] / total_data_points for r in range(len(instance_number_per_client))]
            else:
                total_data_points = sum([len(net_dataidx_map_train[r]) for r in selected])
                fed_avg_freqs = [len(net_dataidx_map_train[r]) / total_data_points for r in selected]

            if round == 0:
                for iidx in range(args.n_parties):
                    final_state = copy.deepcopy(nets[iidx].state_dict())
                    if args.dataset=="shakespeare":
                        if args.model == 'lstm':
                            personalized_pred_list[iidx]["decoder.weight"] = copy.deepcopy(final_state["decoder.weight"])
                            personalized_pred_list[iidx]["decoder.bias"] = copy.deepcopy(final_state["decoder.bias"])
                        elif args.model == 'transformer':
                            personalized_pred_list[iidx]["trg_word_prj.weight"] = copy.deepcopy(final_state["trg_word_prj.weight"])
                    else:
                        if args.model == 'cnn':
                            personalized_pred_list[iidx]["fc3.weight"] = copy.deepcopy(final_state["fc3.weight"])
                            personalized_pred_list[iidx]["fc3.bias"] = copy.deepcopy(final_state["fc3.bias"])
                        elif args.model == 'vit':
                            personalized_pred_list[iidx]["mlp_head.1.weight"] = copy.deepcopy(final_state["mlp_head.1.weight"])
                            personalized_pred_list[iidx]["mlp_head.1.bias"] = copy.deepcopy(final_state["mlp_head.1.bias"])
           
            for idx in range(len(selected)):
                if round != 0:
                    final_state = copy.deepcopy(nets[selected[idx]].state_dict())
                    if args.dataset=="shakespeare":
                        if args.model == 'lstm':
                            personalized_pred_list[selected[idx]]["decoder.weight"] = copy.deepcopy(final_state["decoder.weight"])
                            personalized_pred_list[selected[idx]]["decoder.bias"] = copy.deepcopy(final_state["decoder.bias"])
                        elif args.model == 'transformer':
                            personalized_pred_list[selected[idx]]["trg_word_prj.weight"] = copy.deepcopy(final_state["trg_word_prj.weight"])
                    else:
                        if args.model == 'cnn':
                            personalized_pred_list[selected[idx]]["fc3.weight"] = copy.deepcopy(final_state["fc3.weight"])
                            personalized_pred_list[selected[idx]]["fc3.bias"] = copy.deepcopy(final_state["fc3.bias"])
                        elif args.model == 'vit':
                            personalized_pred_list[selected[idx]]["mlp_head.1.weight"] = copy.deepcopy(final_state["mlp_head.1.weight"])
                            personalized_pred_list[selected[idx]]["mlp_head.1.bias"] = copy.deepcopy(final_state["mlp_head.1.bias"])
                
                net_para = nets[selected[idx]].state_dict()               
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]

            global_model.load_state_dict(global_para)

            if (round+1)>=test_round and (round+1)%eval_step == 0:
                if args.dataset == "shakespeare":
                    train_results, train_avg_loss, train_acc, train_all_acc, test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_personally(
                    personalized_pred_list, global_model, args, train_dl_global, test_dl_global, nets, round, device=device)
                else:
                    train_results, train_avg_loss, train_acc, train_all_acc, test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_personally(
                    personalized_pred_list, global_model, args, net_dataidx_map_train, net_dataidx_map_test, nets, round, device=device)

                if args.log_flag:
                    logger.info('>> Global Model Train accuracy: %f' % train_acc)
                    logger.info('>> Global Model Test accuracy: %f' % test_acc)
                    logger.info('>> Test avg loss: %f' %test_avg_loss)

                results_dict['train_avg_loss'].append(train_avg_loss)
                results_dict['train_avg_acc'].append(train_acc)
                results_dict['test_avg_loss'].append(test_avg_loss)
                results_dict['test_avg_acc'].append(test_acc*100)

        save_path = Path("results_table/"+save_path)
        save_path.mkdir(parents=True, exist_ok=True)
  
        accessories = args.alg + "-" + str(args.n_parties) + "-" + str(args.dataset) + "-" + args.partition + "-" + args.comment

        if args.save_model:
            logger.info("Saving model")
            outfile_gmodel = os.path.join(save_path, 'gmodel_1500.tar')
            torch.save({'epoch':args.comm_round+1, 'state':global_model.state_dict()}, outfile_gmodel)
            for ele in range(len(personalized_pred_list)):
                p_head = os.path.join(save_path, 'phead_1500_'+str(ele)+".tar")
                torch.save({'epoch':args.comm_round+1, 'state':personalized_pred_list[ele]}, p_head)

        json_file_opt = "results_"+accessories+".json"
        with open(str(save_path / json_file_opt), "w") as file:
            json.dump(results_dict, file, indent=4)

    elif args.alg == 'fedRod':
        if args.model not in ['cnn', 'vit', 'transformer', 'lstm']:
            raise NotImplementedError("fedRod uses cnn as backbone")
        logger.info("Initializing nets")

        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()

        if args.dataset == "cifar10":
            class_num = 10
        elif args.dataset == "cifar100":
            class_num = 100
        alpha_dict = {}
        for kd, kv in traindata_cls_counts.items():
            input_embedding = torch.zeros(class_num).to(device)
            sum_v = sum(kv.values())
            for indx, indv in kv.items():
                input_embedding[indx] = indv/sum_v
            alpha_dict[kd] = input_embedding
        
        global_hyper =None

        if args.use_hyperRod:
            if args.model == "cnn":
                gh_dim = 84
            else:
                gh_dim = 128
            logger.info("Initializing hyper classifier")
            global_hyper = nn.Linear(class_num, class_num*(gh_dim+1)).to(device)
            global_hpara = global_hyper.state_dict()

        else:
            logger.info("Initializing Personalized Classification head")
        
        if args.dataset == "shakespeare":
            client_number = int(args.n_parties)
            personalized_pred_list = init_personalized_parameters(args, client_number)
        else:
            assert args.balanced_soft_max
            personalized_pred_list = init_personalized_parameters(args)

        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

            if args.use_hyperRod:
                for p_h in personalized_pred_list:
                    p_h.load_state_dict(global_hpara)

        for round in range(args.comm_round):
            logger.info("in comm round: %d" %round)

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if args.use_hyperRod:
                global_hpara = global_hyper.state_dict()

            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
                        if args.use_hyperRod:
                            personalized_pred_list[idx].load_state_dict(global_hpara)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)
                    if args.use_hyperRod:
                        personalized_pred_list[idx].load_state_dict(global_hpara)

            if args.dataset == 'shakespeare':
                update_dict = local_train_net_fedRod(nets, selected, personalized_pred_list, 
                args, train_dl_global, test_dl_global, logger, device=device)
            else:
                update_dict = local_train_net_fedRod(nets, selected, personalized_pred_list, args, 
                net_dataidx_map_train, net_dataidx_map_test, logger, alpha=alpha_dict, device=device)

            # update global model
            if args.dataset == 'shakespeare':
                instance_number_per_client = [len(train_dl_global[r].dataset) for r in selected]
                total_data_points = sum(instance_number_per_client)
                fed_avg_freqs = [instance_number_per_client[r] / total_data_points for r in range(len(instance_number_per_client))]
            else:
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

                if args.use_hyperRod:
                    hnet_para = personalized_pred_list[selected[idx]].state_dict()
                    if idx == 0:
                        for key in hnet_para:
                            global_hpara[key] = hnet_para[key] * fed_avg_freqs[idx]
                    else:
                        for key in hnet_para:
                            global_hpara[key] += hnet_para[key] * fed_avg_freqs[idx]

            global_model.load_state_dict(global_para)
            if args.use_hyperRod:
                global_hyper.load_state_dict(global_hpara)

            if (round+1)>=test_round and (round+1)%eval_step == 0:
                if args.dataset == "shakespeare":
                    train_results, train_avg_loss, train_acc, train_all_acc, test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_perRod(
                    personalized_pred_list, global_model, args, train_dl_global, test_dl_global, alpha_dict, device=device)
                else:
                    train_results, train_avg_loss, train_acc, train_all_acc, test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_perRod(
                    personalized_pred_list, global_model, args, net_dataidx_map_train, net_dataidx_map_test, alpha_dict, device=device)

                if args.log_flag:
                    logger.info('>> Global Model Train accuracy: %f' % train_acc)
                    logger.info('>> Global Model Test accuracy: %f' % test_acc)
                    logger.info('>> Test avg loss: %f' %test_avg_loss)

                results_dict['train_avg_loss'].append(train_avg_loss)
                results_dict['train_avg_acc'].append(train_acc)
                results_dict['test_avg_loss'].append(test_avg_loss)
                results_dict['test_avg_acc'].append(test_acc*100)

        save_path = Path("results_table/"+save_path)
        save_path.mkdir(parents=True, exist_ok=True)
  
        if args.use_hyperRod:
            accessories = args.alg + "-hyper-" + str(args.n_parties) + "-" + str(args.dataset) + "-" + args.partition + "-" + args.comment
        else:
            accessories = args.alg + "-linear-" + str(args.n_parties) + "-" + str(args.dataset) + "-" + args.partition + "-" + args.comment

        if args.save_model:
            logger.info("Saving model")
            outfile_gmodel = os.path.join(save_path, 'gmodel_1500.tar')
            torch.save({'epoch':args.comm_round+1, 'state':global_model.state_dict()}, outfile_gmodel)
            for ele in range(len(personalized_pred_list)):
                p_head = os.path.join(save_path, 'phead_1500_'+str(ele)+".tar")
                torch.save({'epoch':args.comm_round+1, 'state':personalized_pred_list[ele]}, p_head)

        json_file_opt = "results_"+accessories+".json"
        with open(str(save_path / json_file_opt), "w") as file:
            json.dump(results_dict, file, indent=4)

    elif args.alg == 'fedBN':
        if args.dataset == 'shakespeare':
            raise NotImplementedError("fedBN does not run on shakespeare.")
        if args.model != 'cnn-b':
            raise NotImplementedError("fedBN uses cnn with BN.")
        logger.info("Initializing nets")

        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()

        logger.info("Initializing Personalized BN layer")
        personalized_bn_list = init_personalized_parameters(args)

        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round: %d" %round)

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    node_weights = personalized_bn_list[idx]
                    nets[idx].load_state_dict(global_para)
                    nets[idx].load_state_dict(node_weights, strict=False)

            local_train_net_per(nets, selected, args, net_dataidx_map_train, net_dataidx_map_test, logger, device=device)

            # update global model
            total_data_points = sum([len(net_dataidx_map_train[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map_train[r]) / total_data_points for r in selected]

            if round == 0:
                for iidx in range(args.n_parties):
                    final_state = copy.deepcopy(nets[iidx].state_dict())
                    for ll in range(4):
                        personalized_bn_list[iidx]["bn"+str(ll+1)+".weight"] = copy.deepcopy(final_state["bn"+str(ll+1)+".weight"])
                        personalized_bn_list[iidx]["bn"+str(ll+1)+".bias"] = copy.deepcopy(final_state["bn"+str(ll+1)+".bias"])
           
            for idx in range(len(selected)):
                if round != 0:
                    final_state = copy.deepcopy(nets[selected[idx]].state_dict())
                    for ll in range(4):
                        personalized_bn_list[selected[idx]]["bn"+str(ll+1)+".weight"] = copy.deepcopy(final_state["bn"+str(ll+1)+".weight"])
                        personalized_bn_list[selected[idx]]["bn"+str(ll+1)+".bias"] = copy.deepcopy(final_state["bn"+str(ll+1)+".bias"])
                
                net_para = nets[selected[idx]].state_dict()               
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]

            global_model.load_state_dict(global_para)

            if (round+1)>=test_round and (round+1)%eval_step == 0: 
                
                train_results, train_avg_loss, train_acc, train_all_acc, test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_personally(
                personalized_bn_list, global_model, args, net_dataidx_map_train, net_dataidx_map_test, nets, device=device)

                if args.log_flag:
                    logger.info('>> Global Model Train accuracy: %f' % train_acc)
                    logger.info('>> Global Model Test accuracy: %f' % test_acc)
                    logger.info('>> Test avg loss: %f' %test_avg_loss)

                results_dict['train_avg_loss'].append(train_avg_loss)
                results_dict['train_avg_acc'].append(train_acc)
                results_dict['test_avg_loss'].append(test_avg_loss)
                results_dict['test_avg_acc'].append(test_acc*100)

        save_path = Path("results_table/"+save_path)
        save_path.mkdir(parents=True, exist_ok=True)
  
        accessories = args.alg + "-" + str(args.n_parties) + "-" + str(args.dataset) + "-" + args.partition + "-" + args.comment

        if args.save_model:
            logger.info("Saving model")
            outfile_gmodel = os.path.join(save_path, 'gmodel_1500.tar')
            torch.save({'epoch':args.comm_round+1, 'state':global_model.state_dict()}, outfile_gmodel)
            for ele in range(len(personalized_bn_list)):
                p_head = os.path.join(save_path, 'phead_1500_'+str(ele)+".tar")
                torch.save({'epoch':args.comm_round+1, 'state':personalized_bn_list[ele]}, p_head)

        json_file_opt = "results_"+accessories+".json"
        with open(str(save_path / json_file_opt), "w") as file:
            json.dump(results_dict, file, indent=4)

    elif args.alg == 'pfedKL':
        if args.model != 'cnn-b':
            raise NotImplementedError("pfedKL uses cnn-b with BN.")
        logger.info("Initializing nets")

        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        local_nets, _, _ = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()

        logger.info("Initializing Personalized BN layer")
        personalized_bn_list = init_personalized_parameters(args)

        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for a_iter in range(args.comm_round):
            logger.info("in comm round: %d" %a_iter)

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if a_iter == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    node_weights = personalized_bn_list[idx]
                    nets[idx].load_state_dict(global_para)
                    nets[idx].load_state_dict(node_weights, strict=False)

            local_train_net_per_kl(nets, local_nets, selected, args, net_dataidx_map_train, net_dataidx_map_test, a_iter, logger, device=device)

            # update global model
            total_data_points = sum([len(net_dataidx_map_train[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map_train[r]) / total_data_points for r in selected]

            if a_iter == 0:
                for iidx in range(args.n_parties):
                    final_state = copy.deepcopy(nets[iidx].state_dict())
                    for ll in range(4):
                        personalized_bn_list[iidx]["bn"+str(ll+1)+".weight"] = copy.deepcopy(final_state["bn"+str(ll+1)+".weight"])
                        personalized_bn_list[iidx]["bn"+str(ll+1)+".bias"] = copy.deepcopy(final_state["bn"+str(ll+1)+".bias"])
           
            for idx in range(len(selected)):
                if a_iter != 0:
                    final_state = copy.deepcopy(nets[selected[idx]].state_dict())
                    for ll in range(4):
                        personalized_bn_list[selected[idx]]["bn"+str(ll+1)+".weight"] = copy.deepcopy(final_state["bn"+str(ll+1)+".weight"])
                        personalized_bn_list[selected[idx]]["bn"+str(ll+1)+".bias"] = copy.deepcopy(final_state["bn"+str(ll+1)+".bias"])
                
                net_para = nets[selected[idx]].state_dict()               
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]

            global_model.load_state_dict(global_para)

            if (a_iter+1)>=test_round and (a_iter+1)%eval_step == 0: 
                
                train_results, train_avg_loss, train_acc, train_all_acc, test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_two_branch(
                personalized_bn_list, global_model, local_nets, args, net_dataidx_map_train, net_dataidx_map_test, device=device)

                if args.log_flag:
                    logger.info('>> Global Model Train accuracy: %f' % train_acc)
                    logger.info('>> Global Model Test accuracy: %f' % test_acc)
                    logger.info('>> Test avg loss: %f' %test_avg_loss)

                results_dict['train_avg_loss'].append(train_avg_loss)
                results_dict['train_avg_acc'].append(train_acc)
                results_dict['test_avg_loss'].append(test_avg_loss)
                results_dict['test_avg_acc'].append(test_acc*100)

        save_path = Path("results_table/"+save_path)
        save_path.mkdir(parents=True, exist_ok=True)
  
        accessories = args.alg + "-" + str(args.n_parties) + "-" + str(args.dataset) + "-" + args.partition + "-" + args.comment

        if args.save_model:
            logger.info("Saving model")
            outfile_gmodel = os.path.join(save_path, 'gmodel_1500.tar')
            torch.save({'epoch':args.comm_round+1, 'state':global_model.state_dict()}, outfile_gmodel)
            for ele in range(len(personalized_bn_list)):
                p_head = os.path.join(save_path, 'phead_1500_'+str(ele)+".tar")
                torch.save({'epoch':args.comm_round+1, 'state':personalized_bn_list[ele]}, p_head)

        json_file_opt = "results_"+accessories+".json"
        with open(str(save_path / json_file_opt), "w") as file:
            json.dump(results_dict, file, indent=4)

    elif args.alg == 'moon':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.n_parties, args, device='cpu')

        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 1, args, device='cpu')
        global_model = global_models[0]
        n_comm_rounds = args.comm_round

        # if args.server_momentum:
        #     moment_v = copy.deepcopy(global_model.state_dict())
        #     for key in moment_v:
        #         moment_v[key] = 0

        old_nets_pool = []
        # if args.load_pool_file:
        #     for nets_id in range(args.model_buffer_size):
        #         old_nets, _, _ = init_nets(args.net_config, args.n_parties, args, device='cpu')
        #         checkpoint = torch.load(args.load_pool_file)
        #         for net_id, net in old_nets.items():
        #             net.load_state_dict(checkpoint['pool' + str(nets_id) + '_'+'net'+str(net_id)])
        #         old_nets_pool.append(old_nets)
        # elif args.load_first_net:
        #     if len(old_nets_pool) < args.model_buffer_size:
        #         old_nets = copy.deepcopy(nets)
        #         for _, net in old_nets.items():
        #             net.eval()
        #             for param in net.parameters():
        #                 param.requires_grad = False

        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_model.eval()

            for param in global_model.parameters():
                param.requires_grad = False
            global_w = global_model.state_dict()

            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in selected}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)

            local_train_net_moon(nets_this_round, args, net_dataidx_map_train, net_dataidx_map_test, train_dl=train_dl, test_dl=test_dl, global_model = global_model, prev_model_pool=old_nets_pool, round=round, device=device)

            total_data_points = sum([len(net_dataidx_map_train[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map_train[r]) / total_data_points for r in party_list_this_round]


            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]

            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]

            global_model.load_state_dict(global_w)
            #summary(global_model.to(device), (3, 32, 32))

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))
            global_model.cuda()
            train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
            global_model.to('cpu')
            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Train loss: %f' % train_loss)


            if len(old_nets_pool) < args.model_buffer_size:
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                old_nets_pool.append(old_nets)
            elif args.pool_option == 'FIFO':
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                for i in range(args.model_buffer_size-2, -1, -1):
                    old_nets_pool[i] = old_nets_pool[i+1]
                old_nets_pool[args.model_buffer_size - 1] = old_nets

            mkdirs(args.modeldir+'fedcon/')
            if args.save_model:
                torch.save(global_model.state_dict(), args.modeldir+'fedcon/global_model_'+args.log_file_name+'.pth')
                torch.save(nets[0].state_dict(), args.modeldir+'fedcon/localmodel0'+args.log_file_name+'.pth')
                for nets_id, old_nets in enumerate(old_nets_pool):
                    torch.save({'pool'+ str(nets_id) + '_'+'net'+str(net_id): net.state_dict() for net_id, net in old_nets.items()}, args.modeldir+'fedcon/prev_model_pool_'+args.log_file_name+'.pth')


    elif args.alg == 'fedAP':
        if args.dataset == 'shakespeare':
            raise NotImplementedError("fedBN does not run on shakespeare.")
        if args.model != 'cnn-b':
            raise NotImplementedError("fedBN uses cnn with BN.")
        logger.info("Initializing nets")
        # /public/home/caizhy/work/Peer/NIID-Bench-main/results_table/fedBN-cnn-b-50-cifar10-noniid-labeldir_/

        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]
        paggre_models = copy.deepcopy(nets)

        global_para = global_model.state_dict()

        logger.info("Calculate client weights")
        temp_path = "fedBN-"+args.model+"-"+str(args.n_parties)+"-"+args.dataset+"-"+args.partition+args.comment
        temp_path = Path("results_table/"+temp_path)
        outfile_backbone = os.path.join(temp_path, 'gmodel_1500.tar')
        tmp_backbone = torch.load(outfile_backbone)
        backbone_para = tmp_backbone['state']
        spe_backbone = copy.deepcopy(global_model)
        spe_backbone.load_state_dict(backbone_para)
        spe_bn_list = init_personalized_parameters(args)
        for cc in range(args.n_parties):
            outfile_bn = os.path.join(temp_path, 'phead_1500_'+str(cc)+".tar")
            p_bn = torch.load(outfile_bn)['state']
            spe_bn_list[cc] = copy.deepcopy(p_bn)
        client_weights = set_client_weight(net_dataidx_map_train, net_dataidx_map_test, spe_backbone, spe_bn_list, device, args)
        print(client_weights)


        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round: %d" %round)

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
           
            local_train_net_per(nets, selected, args, net_dataidx_map_train, net_dataidx_map_test, logger, device=device)

            # update global model            
            for cl in range(args.n_parties):
                for key in global_model.state_dict().keys():
                    temp = torch.zeros_like(global_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(args.n_parties):
                        temp += client_weights[cl,client_idx] * nets[client_idx].state_dict()[key]
                    global_model.state_dict()[key].data.copy_(temp)
                    if 'bn' not in key:
                        paggre_models[cl].state_dict()[key].data.copy_(global_model.state_dict()[key])

            for client_idx in range(args.n_parties):
                nets[client_idx].load_state_dict(copy.deepcopy(paggre_models[client_idx].state_dict()))

            global_model.load_state_dict(global_para)

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

        save_path = Path("results_table/"+save_path)
        save_path.mkdir(parents=True, exist_ok=True)
  
        accessories = args.alg + "-" + str(args.n_parties) + "-" + str(args.dataset) + "-" + args.partition + "-" + args.comment

        if args.save_model:
            logger.info("Saving model")
            for node_idx in arr:
                outfile_vit = os.path.join(save_path, 'model_'+str(node_idx)+'_1500.tar')
                torch.save({'epoch':args.comm_round+1, 'state':nets[node_idx].state_dict()}, outfile_vit)

        json_file_opt = "results_"+accessories+".json"
        with open(str(save_path / json_file_opt), "w") as file:
            json.dump(results_dict, file, indent=4)

    elif args.alg == 'local_training':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        
        arr = np.arange(args.n_parties)
        for round in range(args.comm_round):
            logger.info("in comm round: %d" %round)
            logger.info("epochs: %d" %args.epochs)

            local_train_net_per(nets, arr, args, net_dataidx_map_train, net_dataidx_map_test, logger, device=device)
            
            if (round+1)>=test_round and (round+1)%eval_step == 0:
                train_results, train_avg_loss, train_acc, train_all_acc, test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_local(
                nets, args, net_dataidx_map_train, net_dataidx_map_test, device=device)

                if args.log_flag:
                    logger.info('>> Average Train accuracy: %f' % train_acc)
                    logger.info('>> Average Test accuracy: %f' % test_acc)
                    logger.info('>> Test avg loss: %f' %test_avg_loss)

                results_dict['train_avg_loss'].append(train_avg_loss)
                results_dict['train_avg_acc'].append(train_acc)
                results_dict['test_avg_loss'].append(test_avg_loss)
                results_dict['test_avg_acc'].append(test_acc*100)

        save_path = Path("results_table/"+save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        accessories = args.alg + "-" + str(args.n_parties) + "-" + str(args.dataset) + "-" + args.comment
        print("test_all_acc: ", test_all_acc)
        if args.save_model:
            logger.info("Saving model")
            for node_idx in arr:
                outfile_vit = os.path.join(save_path, 'Vit_'+str(node_idx)+'_1500.tar')
                torch.save({'epoch':args.comm_round+1, 'state':nets[node_idx].state_dict()}, outfile_vit)

        json_file_opt = "results_"+accessories+".json"
        with open(str(save_path / json_file_opt), "w") as file:
            json.dump(results_dict, file, indent=4)

    elif args.alg == 'fedcluster':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model1 = global_models[0]
        global_model2 = copy.deepcopy(global_model1)
        global_para1 = global_model1.state_dict()
        global_para2 = global_model2.state_dict()

        if args.is_same_initial:
            for net_id, net in nets.items():
                if net_id<int(args.n_parties/2):
                    net.load_state_dict(global_para1)
                else:
                    net.load_state_dict(global_para2)

        for round in range(args.comm_round):
            logger.info("in comm round: %d" %round)

            selected = np.arange(args.n_parties)
            global_para1 = global_model1.state_dict()
            global_para2 = global_model2.state_dict()

            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        if idx<int(args.n_parties/2):
                            nets[idx].load_state_dict(global_para1)
                        else:
                            nets[idx].load_state_dict(global_para2)
            else:
                for idx in selected:
                    if idx<int(args.n_parties/2):
                        nets[idx].load_state_dict(global_para1)
                    else:
                        nets[idx].load_state_dict(global_para2)

            local_train_net_per(nets, selected, args, net_dataidx_map_train, net_dataidx_map_test, logger, device=device)

            # update global model
            total_data_points_part1 = sum([len(net_dataidx_map_train[r]) for r in range(int(args.n_parties/2))])
            total_data_points_part2 = sum([len(net_dataidx_map_train[r]) for r in range(int(args.n_parties/2), args.n_parties)])
            fed_avg_freqs_part1 = [len(net_dataidx_map_train[r]) / total_data_points_part1 for r in range(int(args.n_parties/2))]
            fed_avg_freqs_part2 = [len(net_dataidx_map_train[r]) / total_data_points_part2 for r in range(int(args.n_parties/2), args.n_parties)]

            for idx in range(int(args.n_parties/2)):
                net_para = nets[idx].state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para1[key] = net_para[key] * fed_avg_freqs_part1[idx]
                else:
                    for key in net_para:
                        global_para1[key] += net_para[key] * fed_avg_freqs_part1[idx]
            global_model1.load_state_dict(global_para1)

            for idx in range(int(args.n_parties/2), args.n_parties):
                net_para = nets[idx].state_dict()
                if idx == int(args.n_parties/2):
                    for key in net_para:
                        global_para2[key] = net_para[key] * fed_avg_freqs_part2[idx-int(args.n_parties/2)]
                else:
                    for key in net_para:
                        global_para2[key] += net_para[key] * fed_avg_freqs_part2[idx-int(args.n_parties/2)]
            global_model2.load_state_dict(global_para2)

            if (round+1)>=test_round and (round+1)%eval_step == 0:
                # train_results, train_avg_loss, train_acc, train_all_acc = compute_accuracy_per_client_simple(global_model, args, net_dataidx_map_train, test=False, device=device)
                # test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_per_client_simple(global_model, args, net_dataidx_map_train, device=device)
                train_results, train_avg_loss, train_acc, train_all_acc, test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_per_client_cluster(
                global_model1, global_model2, args, net_dataidx_map_train, net_dataidx_map_test, nets, device=device)

                if args.log_flag:
                    logger.info('>> Global Model Train accuracy: %f' % train_acc)
                    logger.info('>> Global Model Test accuracy: %f' % test_acc)
                    logger.info('>> Test avg loss: %f' %test_avg_loss)

                results_dict['train_avg_loss'].append(train_avg_loss)
                results_dict['train_avg_acc'].append(train_acc)
                results_dict['test_avg_loss'].append(test_avg_loss)
                results_dict['test_avg_acc'].append(test_acc*100)

        save_path = Path("results_table/"+save_path)
        save_path.mkdir(parents=True, exist_ok=True)
  
        accessories = args.alg + "-" + str(args.n_parties) + "-" + str(args.dataset) + "-" + args.partition + "-" + args.comment
            
        print("test_all_acc: ", test_all_acc)
        if args.save_model:
            logger.info("Saving model")
            outfile_gmodel = os.path.join(save_path, 'gmodel_1500.tar')
            torch.save({'epoch':args.comm_round+1, 'state':global_model.state_dict()}, outfile_gmodel)

        json_file_opt = "results_"+accessories+".json"
        with open(str(save_path / json_file_opt), "w") as file:
            json.dump(results_dict, file, indent=4)

    elif args.alg == 'proto_cluster_ori':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        server_nets = copy.deepcopy(nets)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]
        global_para = global_model.state_dict()

        if args.dataset == 'cifar10':
        	class_number = 10
        elif args.dataset == 'cifar100':
        	class_number = 100

        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round: %d" %round)

            selected = np.arange(args.n_parties)
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    server_state = server_nets[idx].state_dict()
                    nets[idx].load_state_dict(server_state)
                        
            proto_dict = local_train_net_cluster(nets, selected, args, net_dataidx_map_train, net_dataidx_map_test, logger=logger, device=device)
            self_nid = -1
            similarity_dic = {}
            for idx in range(args.n_parties):
                similarity_vec = []
                i_prototype = proto_dict[idx]
                local_para = nets[idx].state_dict()
                for jdx in range(args.n_parties):
                    if idx == jdx:
                        similarity_vec.append(class_number * (math.e**2))
                        self_nid = idx
                    elif idx>jdx:
                    	similarity_vec.append(similarity_dic[(jdx,idx)])
                    else: 
                        j_prototype = proto_dict[jdx]
                        proto_sim = torch.exp(2*torch.cosine_similarity(i_prototype, j_prototype))
                        sim_sum = torch.sum(proto_sim)
                        # print(proto_sim)
                        # print(sim_sum)
                        similarity_vec.append(sim_sum)
                        similarity_dic[(idx,jdx)] = sim_sum
                # similarity_vec[self_nid] = class_number * (math.e**2)
                # print("before: ", similarity_vec)
                node_sim_sum = sum(similarity_vec)
                # print(node_sim_sum)
                similarity_vec = [x/node_sim_sum for x in similarity_vec]
                print("after: ",similarity_vec)
                # assert False

                for jdx in range(args.n_parties):
	                net_para = nets[jdx].state_dict()
	                if jdx == 0:
	                    for key in net_para:
	                        local_para[key] = net_para[key] * similarity_vec[jdx]
	                else:
	                    for key in net_para:
	                        local_para[key] += net_para[key] * similarity_vec[jdx]
                server_nets[idx].load_state_dict(local_para)

            if (round+1)>=test_round and (round+1)%eval_step == 0:
                # train_results, train_avg_loss, train_acc, train_all_acc = compute_accuracy_per_client_simple(global_model, args, net_dataidx_map_train, test=False, device=device)
                # test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_per_client_simple(global_model, args, net_dataidx_map_train, device=device)
                train_results, train_avg_loss, train_acc, train_all_acc, test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_local(
                server_nets, args, net_dataidx_map_train, net_dataidx_map_test, device=device)

                if args.log_flag:
                    logger.info('>> Global Model Train accuracy: %f' % train_acc)
                    logger.info('>> Global Model Test accuracy: %f' % test_acc)
                    logger.info('>> Test avg loss: %f' %test_avg_loss)

                results_dict['train_avg_loss'].append(train_avg_loss)
                results_dict['train_avg_acc'].append(train_acc)
                results_dict['test_avg_loss'].append(test_avg_loss)
                results_dict['test_avg_acc'].append(test_acc*100)

        save_path = Path("results_table/"+save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        accessories = args.alg + "-" + str(args.n_parties) + "-" + str(args.dataset) + "-" + args.partition + "-" + args.comment
            
        print("test_all_acc: ", test_all_acc)
        if args.save_model:
            logger.info("Saving model")
            outfile_gmodel = os.path.join(save_path, 'gmodel_1500.tar')
            torch.save({'epoch':args.comm_round+1, 'state':global_model.state_dict()}, outfile_gmodel)

        json_file_opt = "results_"+accessories+".json"
        with open(str(save_path / json_file_opt), "w") as file:
            json.dump(results_dict, file, indent=4)

    elif args.alg == 'proto_cluster':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        server_nets = copy.deepcopy(nets)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]
        global_para = global_model.state_dict()

        logger.info("Initializing Personalized Classification head")
        personalized_pred_list = init_personalized_parameters(args)

        if args.dataset == 'cifar10':
        	class_number = 10
        elif args.dataset == 'cifar100':
        	class_number = 100

        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round: %d" %round)

            selected = np.arange(args.n_parties)
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    server_state = server_nets[idx].state_dict()
                    nets[idx].load_state_dict(server_state)
                    node_weights = personalized_pred_list[idx]
                    nets[idx].load_state_dict(node_weights, strict=False)
                        
            proto_dict = local_train_net_cluster(nets, selected, args, net_dataidx_map_train, net_dataidx_map_test, logger=logger, device=device)

            if round == 0:
                for iidx in range(args.n_parties):
                    final_state = copy.deepcopy(nets[iidx].state_dict())
                    personalized_pred_list[iidx]["fc3.weight"] = copy.deepcopy(final_state["fc3.weight"])
                    personalized_pred_list[iidx]["fc3.bias"] = copy.deepcopy(final_state["fc3.bias"])

            self_nid = -1
            similarity_dic = {}
            for idx in range(args.n_parties):
                if round != 0:
                    final_state = copy.deepcopy(nets[idx].state_dict())
                    personalized_pred_list[idx]["fc3.weight"] = copy.deepcopy(final_state["fc3.weight"])
                    personalized_pred_list[idx]["fc3.bias"] = copy.deepcopy(final_state["fc3.bias"])

                similarity_vec = []
                i_prototype = proto_dict[idx]
                local_para = nets[idx].state_dict()
                for jdx in range(args.n_parties):
                    if idx == jdx:
                        similarity_vec.append(class_number * (math.e**2))
                        self_nid = idx
                    elif idx>jdx:
                    	similarity_vec.append(similarity_dic[(jdx,idx)])
                    else: 
                        j_prototype = proto_dict[jdx]
                        proto_sim = torch.exp(2*torch.cosine_similarity(i_prototype, j_prototype))
                        sim_sum = torch.sum(proto_sim)
                        # print(proto_sim)
                        # print(sim_sum)
                        similarity_vec.append(sim_sum)
                        similarity_dic[(idx,jdx)] = sim_sum
                # similarity_vec[self_nid] = class_number * (math.e**2)
                # print("before: ", similarity_vec)
                node_sim_sum = sum(similarity_vec)
                # print(node_sim_sum)
                similarity_vec = [x/node_sim_sum for x in similarity_vec]
                print("after: ",similarity_vec)
                # assert False

                for jdx in range(args.n_parties):
	                net_para = nets[jdx].state_dict()
	                if jdx == 0:
	                    for key in net_para:
	                        local_para[key] = net_para[key] * similarity_vec[jdx]
	                else:
	                    for key in net_para:
	                        local_para[key] += net_para[key] * similarity_vec[jdx]
                server_nets[idx].load_state_dict(local_para)

            if (round+1)>=test_round and (round+1)%eval_step == 0:
                # train_results, train_avg_loss, train_acc, train_all_acc = compute_accuracy_per_client_simple(global_model, args, net_dataidx_map_train, test=False, device=device)
                # test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_per_client_simple(global_model, args, net_dataidx_map_train, device=device)
                train_results, train_avg_loss, train_acc, train_all_acc, test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_local_per(
                personalized_pred_list, server_nets, args, net_dataidx_map_train, net_dataidx_map_test, device=device)

                if args.log_flag:
                    logger.info('>> Global Model Train accuracy: %f' % train_acc)
                    logger.info('>> Global Model Test accuracy: %f' % test_acc)
                    logger.info('>> Test avg loss: %f' %test_avg_loss)

                results_dict['train_avg_loss'].append(train_avg_loss)
                results_dict['train_avg_acc'].append(train_acc)
                results_dict['test_avg_loss'].append(test_avg_loss)
                results_dict['test_avg_acc'].append(test_acc*100)

        save_path = Path("results_table/"+save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        accessories = args.alg + "-" + str(args.n_parties) + "-" + str(args.dataset) + "-" + args.partition + "-" + args.comment
            
        print("test_all_acc: ", test_all_acc)
        if args.save_model:
            logger.info("Saving model")
            outfile_gmodel = os.path.join(save_path, 'gmodel_1500.tar')
            torch.save({'epoch':args.comm_round+1, 'state':global_model.state_dict()}, outfile_gmodel)

        json_file_opt = "results_"+accessories+".json"
        with open(str(save_path / json_file_opt), "w") as file:
            json.dump(results_dict, file, indent=4)

    elif args.alg == 'hyperVit_cluster':
        if args.model not in ["vit", "transformer"]:
            raise NotImplementedError("hyperVit only supports ViT and transformer")

        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        server_nets = copy.deepcopy(nets)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        if args.dataset == "cifar10":
            class_number = 10
        elif args.dataset == "cifar100":
            class_number = 100

        logger.info("Initializing hyper")
        hnet = init_hyper(args).to(device)

        optimizers = {
            'sgd': torch.optim.SGD(
                params=hnet.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3
            ),
            'adam': torch.optim.Adam(params=hnet.parameters(), lr=args.lr)
        }
        optimizer = optimizers['sgd']


        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round: %d" %round)

            hnet.train()
            grads_update = []
            selected = np.arange(args.n_parties)
            weights = hnet(torch.tensor([selected], dtype=torch.long).to(device), False)

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in range(args.n_parties):
                        node_weights = weights[idx]
                        nets[idx].load_state_dict(global_para)
                        nets[idx].load_state_dict(node_weights, strict=False)
            else:
                for idx in range(args.n_parties):
                    node_weights = weights[idx]
                    server_state = server_nets[idx].state_dict()
                    nets[idx].load_state_dict(server_state)
                    nets[idx].load_state_dict(node_weights, strict=False)

            proto_dict = local_train_net_cluster(nets, selected, args, net_dataidx_map_train, net_dataidx_map_test, logger=logger, device=device)
            self_nid = -1
            similarity_dic = {}
            total_data_points = sum([len(net_dataidx_map_train[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map_train[r]) / total_data_points for r in selected]
            
            # update global model
            for idx in range(args.n_parties):
                similarity_vec = []
                i_prototype = proto_dict[idx]
                local_para = nets[idx].state_dict()
                for jdx in range(args.n_parties):
                    if idx == jdx:
                        similarity_vec.append(class_number * (math.e**2))
                        self_nid = idx
                    elif idx>jdx:
                	    similarity_vec.append(similarity_dic[(jdx,idx)])
                    else: 
                        j_prototype = proto_dict[jdx]
                        proto_sim = torch.exp(2*torch.cosine_similarity(i_prototype, j_prototype))
                        sim_sum = torch.sum(proto_sim)
                        # print(proto_sim)
                        # print(sim_sum)
                        similarity_vec.append(sim_sum)
                        similarity_dic[(idx,jdx)] = sim_sum
                # similarity_vec[self_nid] = class_number * (math.e**2)
                # print("before: ", similarity_vec)
                node_sim_sum = sum(similarity_vec)
                # print(node_sim_sum)
                similarity_vec = [x/node_sim_sum for x in similarity_vec]
                # print("after: ",similarity_vec)
                # assert False

                for jdx in range(args.n_parties):
                    net_para = nets[jdx].state_dict()
                    if jdx == 0:
                        for key in net_para:
                            local_para[key] = net_para[key] * similarity_vec[jdx]
                    else:
                        for key in net_para:
                            local_para[key] += net_para[key] * similarity_vec[jdx]
                server_nets[idx].load_state_dict(local_para)


                final_state = nets[idx].state_dict()
                net_para = nets[idx].state_dict()

                node_weights = weights[idx]
                inner_state = OrderedDict({k: tensor.data for k, tensor in node_weights.items()})
                delta_theta = OrderedDict({k: inner_state[k] - final_state[k] for k in node_weights.keys()})
                hnet_grads = torch.autograd.grad(
                    list(node_weights.values()), hnet.parameters(), grad_outputs=list(delta_theta.values()), retain_graph=True
                )

                if idx == 0:
                    grads_update = [fed_avg_freqs[idx]*x  for x in hnet_grads]
                else:
                    for g in range(len(hnet_grads)):
                        grads_update[g] += fed_avg_freqs[idx] * hnet_grads[g]

            optimizer.zero_grad()
            for p, g in zip(hnet.parameters(), grads_update):
                p.grad = g
            optimizer.step()

            if (round+1)>=test_round and (round+1)%eval_step == 0:
                # train_results, train_avg_loss, train_acc, train_all_acc = compute_accuracy_per_client(hnet, global_model, args, net_dataidx_map_train,test=False, device=device)
                # test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_per_client(hnet, global_model, args, net_dataidx_map_train, device=device)
                train_results, train_avg_loss, train_acc, train_all_acc, test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_per_client_proto_cluster(
                hnet, nets, global_model, args, net_dataidx_map_train, net_dataidx_map_test, class_number, device=device)
                
                if args.log_flag:
                    logger.info('>> Global Model Train accuracy: %f' % train_acc)
                    logger.info('>> Global Model Test accuracy: %f' % test_acc)
                    logger.info('>> Test avg loss: %f' %test_avg_loss)

                results_dict['train_avg_loss'].append(train_avg_loss)
                results_dict['train_avg_acc'].append(train_acc)
                results_dict['test_all_acc'] = test_all_acc
                results_dict['test_avg_loss'].append(test_avg_loss)
                results_dict['test_avg_acc'].append(test_acc*100)

        save_path = Path("results_table/"+save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        accessories = args.alg + "-" + str(args.n_parties) + "-" + str(args.dataset) + "-" + args.partition + "-" + args.comment
        
        if args.save_model:
            logger.info("Saving model")
            outfile_hp = os.path.join(save_path,  'HY_1500.tar')
            outfile_vit = os.path.join(save_path, 'Vit_1500.tar')
            torch.save({'epoch':args.comm_round+1, 'state':hnet.state_dict()}, outfile_hp)
            torch.save({'epoch':args.comm_round+1, 'state':global_model.state_dict()}, outfile_vit)

        json_file_opt = "results_"+accessories+".json"
        with open(str(save_path / json_file_opt), "w") as file:
            json.dump(results_dict, file, indent=4)


    acc_all  = np.asarray(results_dict['test_avg_acc'])
    logger.info("Accuracy Record: ")
    logger.info(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std  = np.std(acc_all)
    logger.info('Test Acc = %4.2f%% +- %4.2f%%' %(acc_mean, acc_std))
    if args.show_all_accuracy:
        logger.info("Accuracy in each client: ")
        logger.info(results_dict['test_all_acc'])