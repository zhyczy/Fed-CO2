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
from models.cnn import CNNTarget, CNN_B
from utils import *
from methods.method import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='cnn-b', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='noniid-labeldir', help='the data partitioning strategy')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=1, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=10,  help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fed-co2',
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
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox or moon')
    parser.add_argument('--noise', type=float, default=0, help='how much noise we add to some party')
    parser.add_argument('--noise_type', type=str, default='None', help='Different level of noise or different space of noise')
    parser.add_argument('--rho', type=float, default=0, help='Parameter controlling the momentum SGD')
    parser.add_argument('--sample', type=float, default=0.4, help='Sample ratio for each communication round')
    parser.add_argument('--train_acc_pre', action='store_true')
    parser.add_argument('--eval_step', type=int, default=5)
    parser.add_argument('--test_round', type=int, default=1300)
    parser.add_argument('--log_flag', default=True)
    parser.add_argument("--save_model", action='store_true')
    parser.add_argument("--comment", default="_")
    parser.add_argument("--definite_selection", action='store_true')
    parser.add_argument("--show_all_accuracy", action='store_true')

    """
    Used for moon
    temperature is 0.5 by default
    """
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')


    args = parser.parse_args()
    return args


def init_nets(net_configs, dropout_p, n_parties, args):

    nets = {net_i: None for net_i in range(n_parties)}
    device = torch.device(args.device)

    for net_i in range(n_parties):
        if args.model == "cnn":
            if args.dataset == "cifar10":
                net = CNNTarget(n_kernels=16)
            elif args.dataset == "cifar100":
                net = CNNTarget(n_kernels=16, out_dim=100)

        elif args.model == "cnn-b":
            if args.dataset == "cifar10":
                net = CNN_B(n_kernels=16)
            elif args.dataset == "cifar100":
                net = CNN_B(n_kernels=16, out_dim=100)

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


def init_personalized_parameters(args, client_number=None):
    personalized_pred_list = []
    if args.dataset == "cifar10":
        class_num = 10
    elif args.dataset == "cifar100":
        class_num = 100

    if args.alg == 'fedrod':
        if args.model == "cnn":
            dim = 84
            for nndx in range(args.n_parties):
                p_class = nn.Linear(dim, class_num).to(args.device)
                personalized_pred_list.append(p_class)

    elif args.alg == 'fedper':
        if args.model == 'cnn':
            dim = 84
            for nndx in range(args.n_parties):
                para_dict = OrderedDict()
                para_dict["fc3.weight"] = None
                para_dict["fc3.bias"] = None
                personalized_pred_list.append(para_dict)
        elif args.model == 'cnn-b':
            dim = 84
            for nndx in range(args.n_parties):
                para_dict = OrderedDict()
                para_dict["fc3.weight"] = None
                para_dict["fc3.bias"] = None
                personalized_pred_list.append(para_dict)

    elif args.alg in ['fedbn', 'fed-co2']:
        for nndx in range(args.n_parties):
            bn_dict = OrderedDict()
            for ll in range(4):
                bn_dict["bn"+str(ll+1)+".weight"]=None
                bn_dict["bn"+str(ll+1)+".bias"]=None
            personalized_pred_list.append(bn_dict)

    return personalized_pred_list


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
    logging.info("Beta: %f" % args.beta)
    logging.info("Sample rate: %f" % args.sample)
    logging.info("Print Accuracy on training set: %s" % args.train_acc_pre)
    logging.info("Save model: %s" % args.save_model)
    logging.info("Total running round: %s" % args.comm_round)
    logging.info("Test round fequency: %d" % args.eval_step)
    logging.info("Show every client's accuracy: %s" %args.show_all_accuracy)
    if args.alg == "fedprox":
        logging.info("mu value: %f" %args.mu)
    if args.test_round<=1:
        raise NotImplementedError("test round should be larger than 1")

    save_path = args.alg+"-"+args.model+"-"+str(args.n_parties)+"-"+args.dataset+"-"+args.partition
    mkdirs(args.modeldir)
    device = torch.device(args.device)

    mkdirs(args.logdir)
    if args.log_file_name is None:
        argument_path= args.alg + " " + args.model + " " + '-experiment_arguments-%s.json ' % datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
    else:
        argument_path=args.log_file_name+'.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)

    if args.log_file_name is None:
        args.log_file_name = args.model + " " + '-experiment_log-%s ' % (datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S"))
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

        accessories = args.alg + "-" + str(args.n_parties) + "-" + str(args.dataset) + "-" + args.partition + "-" + args.comment  
        if args.save_model:
            logger.info("Saving model")
            outfile_gmodel = os.path.join(save_path, 'gmodel_1500.tar')
            torch.save({'epoch':args.comm_round+1, 'state':global_model.state_dict()}, outfile_gmodel)

        json_file_opt = "results_"+accessories+".json"
        with open(str(save_path / json_file_opt), "w") as file:
            json.dump(results_dict, file, indent=4)

    elif args.alg == 'fedprox':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()

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
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net_fedprox(nets, selected, global_model, args, net_dataidx_map_train, net_dataidx_map_test, logger, device=device)
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

        accessories = args.alg + "-" + str(args.n_parties) + "-" + str(args.dataset) + "-" + args.comment
        if args.save_model:
            logger.info("Saving model")
            outfile_gmodel = os.path.join(save_path, 'gmodel_1500.tar')
            torch.save({'epoch':args.comm_round+1, 'state':global_model.state_dict()}, outfile_gmodel)

        json_file_opt = "results_"+accessories+".json"
        with open(str(save_path / json_file_opt), "w") as file:
            json.dump(results_dict, file, indent=4)

    elif args.alg == 'fedper':
        if args.model != 'cnn':
            raise NotImplementedError("fedper uses cnn as backbone")
        logger.info("Initializing nets")

        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()

        logger.info("Initializing Personalized Classification head")
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

            local_train_net_per(nets, selected, args, net_dataidx_map_train, net_dataidx_map_test, logger, device=device)

            # update global model
            total_data_points = sum([len(net_dataidx_map_train[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map_train[r]) / total_data_points for r in selected]

            if round == 0:
                for iidx in range(args.n_parties):
                    final_state = copy.deepcopy(nets[iidx].state_dict())
                    personalized_pred_list[iidx]["fc3.weight"] = copy.deepcopy(final_state["fc3.weight"])
                    personalized_pred_list[iidx]["fc3.bias"] = copy.deepcopy(final_state["fc3.bias"])
           
            for idx in range(len(selected)):
                if round != 0:
                    final_state = copy.deepcopy(nets[selected[idx]].state_dict())
                    personalized_pred_list[selected[idx]]["fc3.weight"] = copy.deepcopy(final_state["fc3.weight"])
                    personalized_pred_list[selected[idx]]["fc3.bias"] = copy.deepcopy(final_state["fc3.bias"])
                
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

    elif args.alg == 'fedrod':
        if args.model != 'cnn':
            raise NotImplementedError("fedrod uses cnn as backbone")
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
        
        logger.info("Initializing Personalized Classification head")
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
                    nets[idx].load_state_dict(global_para)

            update_dict = local_train_net_fedrod(nets, selected, personalized_pred_list, args, 
            net_dataidx_map_train, net_dataidx_map_test, logger, alpha=alpha_dict, device=device)

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

    elif args.alg == 'fedbn':
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

    elif args.alg == 'fed-co2':
        if args.model != 'cnn-b':
            raise NotImplementedError("fed-co2 uses cnn-b with BN.")
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

            local_train_net_per_2branch(nets, local_nets, selected, args, net_dataidx_map_train, net_dataidx_map_test, a_iter, logger, device=device)

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
                
                train_results, train_avg_loss, train_acc, train_all_acc, test_results, test_avg_loss, test_acc, test_all_acc, test_g_avg_loss, test_g_acc, test_g_all_acc, test_p_avg_loss, test_p_acc, test_p_all_acc = compute_accuracy_two_branch(
                personalized_bn_list, global_model, local_nets, args, net_dataidx_map_train, net_dataidx_map_test, device=device)

                if args.log_flag:
                    logger.info('>> Global Model Train accuracy: %f' % train_acc)
                    logger.info('>> Global Model Test accuracy: %f' % test_acc)
                    logger.info('>> Global Model G Test accuracy: %f' % test_g_acc)
                    logger.info('>> Global Model P Test accuracy: %f' % test_p_acc)
                    logger.info('>> Test avg loss: %f' %test_avg_loss)

                results_dict['train_avg_loss'].append(train_avg_loss)
                results_dict['train_avg_acc'].append(train_acc)
                results_dict['test_avg_loss'].append(test_avg_loss)
                results_dict['test_avg_acc'].append(test_acc*100)
                results_dict['test_g_avg_loss'].append(test_g_avg_loss)
                results_dict['test_g_avg_acc'].append(test_g_acc*100)
                results_dict['test_p_avg_loss'].append(test_p_avg_loss)
                results_dict['test_p_avg_acc'].append(test_p_acc*100)

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
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0].to(device)
        n_comm_rounds = args.comm_round
        old_nets = copy.deepcopy(nets)
        for _, net in old_nets.items():
            net.eval()
            for param in net.parameters():
                param.requires_grad = False

        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_model.eval()
            for param in global_model.parameters():
                param.requires_grad = False
            global_w = global_model.state_dict()

            nets_this_round = {k: nets[k] for k in selected}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)

            local_train_net_moon(nets_this_round, args, net_dataidx_map_train, net_dataidx_map_test, global_model, old_nets, device=device)

            total_data_points = sum([len(net_dataidx_map_train[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map_train[r]) / total_data_points for r in selected]

            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]
            global_model.load_state_dict(global_w)

            if (round+1)>=test_round and (round+1)%eval_step == 0:
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

        if args.save_model:
            logger.info("Saving model")
            outfile_gmodel = os.path.join(save_path, 'gmodel_1500.tar')
            torch.save({'epoch':args.comm_round+1, 'state':global_model.state_dict()}, outfile_gmodel)

        accessories = args.alg + "-" + str(args.n_parties) + "-" + str(args.dataset) + "-" + args.partition + "-" + args.comment

        json_file_opt = "results_"+accessories+".json"
        with open(str(save_path / json_file_opt), "w") as file:
            json.dump(results_dict, file, indent=4)

    elif args.alg == 'singleset':
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

    acc_all  = np.asarray(results_dict['test_avg_acc'])
    logger.info("Accuracy Record: ")
    logger.info(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std  = np.std(acc_all)
    logger.info('Test Acc = %4.2f%% +- %4.2f%%' %(acc_mean, acc_std))

    if args.alg == 'fed-co2':
        acc_all  = np.asarray(results_dict['test_g_avg_acc'])
        logger.info("Accuracy G Record: ")
        logger.info(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        logger.info('Test Acc = %4.2f%% +- %4.2f%%' %(acc_mean, acc_std))

        acc_all  = np.asarray(results_dict['test_p_avg_acc'])
        logger.info("Accuracy P Record: ")
        logger.info(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        logger.info('Test Acc = %4.2f%% +- %4.2f%%' %(acc_mean, acc_std))
    if args.show_all_accuracy:
        logger.info("Accuracy in each client: ")
        logger.info(test_all_acc)
        if args.alg == 'fed-co2':
            logger.info("G Accuracy in each client: ")
            logger.info(test_g_all_acc)
            logger.info("P Accuracy in each client: ")
            logger.info(test_p_all_acc)
