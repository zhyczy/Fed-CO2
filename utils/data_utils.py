import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import numpy as np
import torch
import copy
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os

def prepare_data_domainNet(args):
    data_base_path = 'data'
    transform_train = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30,30)),
            transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.ToTensor(),
    ])

    # clipart
    clipart_trainset = DomainNetDataset(data_base_path, 'clipart', transform=transform_train)
    clipart_testset = DomainNetDataset(data_base_path, 'clipart', transform=transform_test, train=False)
    # infograph
    infograph_trainset = DomainNetDataset(data_base_path, 'infograph', transform=transform_train)
    infograph_testset = DomainNetDataset(data_base_path, 'infograph', transform=transform_test, train=False)
    # painting
    painting_trainset = DomainNetDataset(data_base_path, 'painting', transform=transform_train)
    painting_testset = DomainNetDataset(data_base_path, 'painting', transform=transform_test, train=False)
    # quickdraw
    quickdraw_trainset = DomainNetDataset(data_base_path, 'quickdraw', transform=transform_train)
    quickdraw_testset = DomainNetDataset(data_base_path, 'quickdraw', transform=transform_test, train=False)
    # real
    real_trainset = DomainNetDataset(data_base_path, 'real', transform=transform_train)
    real_testset = DomainNetDataset(data_base_path, 'real', transform=transform_test, train=False)
    # sketch
    sketch_trainset = DomainNetDataset(data_base_path, 'sketch', transform=transform_train)
    sketch_testset = DomainNetDataset(data_base_path, 'sketch', transform=transform_test, train=False)

    min_data_len = min(len(clipart_trainset), len(infograph_trainset), len(painting_trainset), len(quickdraw_trainset), len(real_trainset), len(sketch_trainset))
    # print(len(clipart_trainset), len(infograph_trainset), len(painting_trainset), len(quickdraw_trainset), len(real_trainset), len(sketch_trainset))
    # print("min_data_len: ", min_data_len)
    val_len = int(min_data_len * 0.05)
    min_data_len = int(min_data_len * 0.05)
    # min_data_len = 500
    print("Train data number: ", min_data_len)

    clipart_valset   = torch.utils.data.Subset(clipart_trainset, list(range(len(clipart_trainset)))[-val_len:])
    clipart_trainset = torch.utils.data.Subset(clipart_trainset, list(range(min_data_len)))
    
    infograph_valset   = torch.utils.data.Subset(infograph_trainset, list(range(len(infograph_trainset)))[-val_len:])
    infograph_trainset = torch.utils.data.Subset(infograph_trainset, list(range(min_data_len)))
    
    painting_valset   = torch.utils.data.Subset(painting_trainset, list(range(len(painting_trainset)))[-val_len:])
    painting_trainset = torch.utils.data.Subset(painting_trainset, list(range(min_data_len)))

    quickdraw_valset   = torch.utils.data.Subset(quickdraw_trainset, list(range(len(quickdraw_trainset)))[-val_len:])
    quickdraw_trainset = torch.utils.data.Subset(quickdraw_trainset, list(range(min_data_len)))

    real_valset   = torch.utils.data.Subset(real_trainset, list(range(len(real_trainset)))[-val_len:])
    real_trainset = torch.utils.data.Subset(real_trainset, list(range(min_data_len)))

    sketch_valset   = torch.utils.data.Subset(sketch_trainset, list(range(len(sketch_trainset)))[-val_len:])
    sketch_trainset = torch.utils.data.Subset(sketch_trainset, list(range(min_data_len)))

    clipart_train_loader = torch.utils.data.DataLoader(clipart_trainset, batch_size=32, shuffle=True)
    clipart_val_loader   = torch.utils.data.DataLoader(clipart_valset, batch_size=32, shuffle=False)
    clipart_test_loader  = torch.utils.data.DataLoader(clipart_testset, batch_size=32, shuffle=False)

    infograph_train_loader = torch.utils.data.DataLoader(infograph_trainset, batch_size=32, shuffle=True)
    infograph_val_loader = torch.utils.data.DataLoader(infograph_valset, batch_size=32, shuffle=False)
    infograph_test_loader = torch.utils.data.DataLoader(infograph_testset, batch_size=32, shuffle=False)

    painting_train_loader = torch.utils.data.DataLoader(painting_trainset, batch_size=32, shuffle=True)
    painting_val_loader = torch.utils.data.DataLoader(painting_valset, batch_size=32, shuffle=False)
    painting_test_loader = torch.utils.data.DataLoader(painting_testset, batch_size=32, shuffle=False)

    quickdraw_train_loader = torch.utils.data.DataLoader(quickdraw_trainset, batch_size=32, shuffle=True)
    quickdraw_val_loader = torch.utils.data.DataLoader(quickdraw_valset, batch_size=32, shuffle=False)
    quickdraw_test_loader = torch.utils.data.DataLoader(quickdraw_testset, batch_size=32, shuffle=False)

    real_train_loader = torch.utils.data.DataLoader(real_trainset, batch_size=32, shuffle=True)
    real_val_loader = torch.utils.data.DataLoader(real_valset, batch_size=32, shuffle=False)
    real_test_loader = torch.utils.data.DataLoader(real_testset, batch_size=32, shuffle=False)

    sketch_train_loader = torch.utils.data.DataLoader(sketch_trainset, batch_size=32, shuffle=True)
    sketch_val_loader = torch.utils.data.DataLoader(sketch_valset, batch_size=32, shuffle=False)
    sketch_test_loader = torch.utils.data.DataLoader(sketch_testset, batch_size=32, shuffle=False)
    
    train_loaders = [clipart_train_loader, infograph_train_loader, painting_train_loader, quickdraw_train_loader, real_train_loader, sketch_train_loader]
    val_loaders = [clipart_val_loader, infograph_val_loader, painting_val_loader, quickdraw_val_loader, real_val_loader, sketch_val_loader]
    test_loaders = [clipart_test_loader, infograph_test_loader, painting_test_loader, quickdraw_test_loader, real_test_loader, sketch_test_loader]
    
    return train_loaders, val_loaders, test_loaders


def prepare_data_domainNet_partition_train(args):
    data_base_path = 'data'
    transform_train = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30,30)),
            transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.ToTensor(),
    ])

    print("Clipart: ")
    net_dataidx_map_train, _, train_ratio, _ = Dataset_partition_domainnet('clipart', args.beta, split_test=False, n_parties=args.divide, min_require_size=args.min_img_num)
    net_dataidx_map_test = Adjust_test_dataset_domainnet('clipart', train_ratio[0])
    clipart_trainset = DomainNetDataset_sub(data_base_path, 'clipart', net_dataidx_map_train[0], transform=transform_train)
    clipart_testset = DomainNetDataset_sub(data_base_path, 'clipart', net_dataidx_map_test, transform=transform_test, train=False)

    print("Infograph: ")
    net_dataidx_map_train, _, train_ratio, _ = Dataset_partition_domainnet('infograph', args.beta, split_test=False, n_parties=args.divide, min_require_size=args.min_img_num)
    net_dataidx_map_test = Adjust_test_dataset_domainnet('infograph', train_ratio[0])
    infograph_trainset = DomainNetDataset_sub(data_base_path, 'infograph', net_dataidx_map_train[0], transform=transform_train)
    infograph_testset = DomainNetDataset_sub(data_base_path, 'infograph', net_dataidx_map_test, transform=transform_test, train=False)
    
    print("Painting: ")
    net_dataidx_map_train, _, train_ratio, _ = Dataset_partition_domainnet('painting', args.beta, split_test=False, n_parties=args.divide, min_require_size=args.min_img_num)
    if args.divide>=3:
        net_dataidx_map_test = Adjust_test_dataset_domainnet('painting', train_ratio[2])
        painting_trainset = DomainNetDataset_sub(data_base_path, 'painting', net_dataidx_map_train[2], transform=transform_train)
    else:
        net_dataidx_map_test = Adjust_test_dataset_domainnet('painting', train_ratio[1])
        painting_trainset = DomainNetDataset_sub(data_base_path, 'painting', net_dataidx_map_train[1], transform=transform_train)
    painting_testset = DomainNetDataset_sub(data_base_path, 'painting', net_dataidx_map_test, transform=transform_test, train=False)
    
    print("Quickdraw: ")
    net_dataidx_map_train, _, train_ratio, _ = Dataset_partition_domainnet('quickdraw', args.beta, split_test=False, n_parties=args.divide, min_require_size=args.min_img_num)
    net_dataidx_map_test = Adjust_test_dataset_domainnet('quickdraw', train_ratio[1])
    quickdraw_trainset = DomainNetDataset_sub(data_base_path, 'quickdraw', net_dataidx_map_train[1], transform=transform_train)
    quickdraw_testset = DomainNetDataset_sub(data_base_path, 'quickdraw', net_dataidx_map_test, transform=transform_test, train=False)
    
    print("Real")
    net_dataidx_map_train, _, train_ratio, _ = Dataset_partition_domainnet('real', args.beta, split_test=False, n_parties=args.divide, min_require_size=args.min_img_num)
    net_dataidx_map_test = Adjust_test_dataset_domainnet('real', train_ratio[0])
    real_trainset = DomainNetDataset_sub(data_base_path, 'real', net_dataidx_map_train[0], transform=transform_train)
    real_testset = DomainNetDataset_sub(data_base_path, 'real', net_dataidx_map_test, transform=transform_test, train=False)
    
    print("Sketch")
    net_dataidx_map_train, _, train_ratio, _ = Dataset_partition_domainnet('sketch', args.beta, split_test=False, n_parties=args.divide, min_require_size=args.min_img_num)
    net_dataidx_map_test = Adjust_test_dataset_domainnet('sketch', train_ratio[0])
    sketch_trainset = DomainNetDataset_sub(data_base_path, 'sketch', net_dataidx_map_train[0], transform=transform_train)
    sketch_testset = DomainNetDataset_sub(data_base_path, 'sketch', net_dataidx_map_test, transform=transform_test, train=False)

    data_num_list = [len(clipart_trainset), len(infograph_trainset), len(painting_trainset), len(quickdraw_trainset), len(real_trainset), len(sketch_trainset)]
    min_data_len = min(len(clipart_trainset), len(infograph_trainset), len(painting_trainset), len(quickdraw_trainset), len(real_trainset), len(sketch_trainset))
    print((len(clipart_trainset), len(infograph_trainset), len(painting_trainset), len(quickdraw_trainset), len(real_trainset), len(sketch_trainset)))

    unq, unq_cnt = np.unique([clipart_trainset[x][1] for x in range(len(clipart_trainset))], return_counts=True)
    print("Train Clipart: ", {unq[i]: unq_cnt[i] for i in range(len(unq))})
    unq, unq_cnt = np.unique([clipart_testset[x][1] for x in range(len(clipart_testset))], return_counts=True)
    print("Test Clipart: ", {unq[i]: unq_cnt[i] for i in range(len(unq))})

    unq, unq_cnt = np.unique([infograph_trainset[x][1] for x in range(len(infograph_trainset))], return_counts=True)
    print("Train Infograph: ", {unq[i]: unq_cnt[i] for i in range(len(unq))})
    unq, unq_cnt = np.unique([infograph_testset[x][1] for x in range(len(infograph_testset))], return_counts=True)
    print("Test Infograph: ", {unq[i]: unq_cnt[i] for i in range(len(unq))})

    unq, unq_cnt = np.unique([painting_trainset[x][1] for x in range(len(painting_trainset))], return_counts=True)
    print("Train Painting: ", {unq[i]: unq_cnt[i] for i in range(len(unq))})
    unq, unq_cnt = np.unique([painting_testset[x][1] for x in range(len(painting_testset))], return_counts=True)
    print("Test Painting: ", {unq[i]: unq_cnt[i] for i in range(len(unq))})

    unq, unq_cnt = np.unique([quickdraw_trainset[x][1] for x in range(len(quickdraw_trainset))], return_counts=True)
    print("Train Quickdraw: ", {unq[i]: unq_cnt[i] for i in range(len(unq))})
    unq, unq_cnt = np.unique([quickdraw_testset[x][1] for x in range(len(quickdraw_testset))], return_counts=True)
    print("Test Quickdraw: ", {unq[i]: unq_cnt[i] for i in range(len(unq))})

    unq, unq_cnt = np.unique([real_trainset[x][1] for x in range(len(real_trainset))], return_counts=True)
    print("Train Real: ", {unq[i]: unq_cnt[i] for i in range(len(unq))})
    unq, unq_cnt = np.unique([real_testset[x][1] for x in range(len(real_testset))], return_counts=True)
    print("Test Real: ", {unq[i]: unq_cnt[i] for i in range(len(unq))})

    unq, unq_cnt = np.unique([sketch_trainset[x][1] for x in range(len(sketch_trainset))], return_counts=True)
    print("Train Sketch: ", {unq[i]: unq_cnt[i] for i in range(len(unq))})
    unq, unq_cnt = np.unique([sketch_testset[x][1] for x in range(len(sketch_testset))], return_counts=True)
    print("Test Sketch: ", {unq[i]: unq_cnt[i] for i in range(len(unq))})

    clipart_train_loader = torch.utils.data.DataLoader(clipart_trainset, batch_size=32, shuffle=True)
    clipart_test_loader  = torch.utils.data.DataLoader(clipart_testset, batch_size=32, shuffle=False)

    infograph_train_loader = torch.utils.data.DataLoader(infograph_trainset, batch_size=32, shuffle=True)
    infograph_test_loader = torch.utils.data.DataLoader(infograph_testset, batch_size=32, shuffle=False)

    painting_train_loader = torch.utils.data.DataLoader(painting_trainset, batch_size=32, shuffle=True)
    painting_test_loader = torch.utils.data.DataLoader(painting_testset, batch_size=32, shuffle=False)

    quickdraw_train_loader = torch.utils.data.DataLoader(quickdraw_trainset, batch_size=32, shuffle=True)
    quickdraw_test_loader = torch.utils.data.DataLoader(quickdraw_testset, batch_size=32, shuffle=False)

    real_train_loader = torch.utils.data.DataLoader(real_trainset, batch_size=32, shuffle=True)
    real_test_loader = torch.utils.data.DataLoader(real_testset, batch_size=32, shuffle=False)

    sketch_train_loader = torch.utils.data.DataLoader(sketch_trainset, batch_size=32, shuffle=True)
    sketch_test_loader = torch.utils.data.DataLoader(sketch_testset, batch_size=32, shuffle=False)
    
    train_loaders = [clipart_train_loader, infograph_train_loader, painting_train_loader, quickdraw_train_loader, real_train_loader, sketch_train_loader]
    test_loaders = [clipart_test_loader, infograph_test_loader, painting_test_loader, quickdraw_test_loader, real_test_loader, sketch_test_loader]
    return train_loaders, test_loaders, data_num_list


def prepare_data_office(args):
    data_base_path = 'data'
    transform_office = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30,30)),
            transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.ToTensor(),
    ])

    # amazon
    amazon_trainset = OfficeDataset(data_base_path, 'amazon', transform=transform_office)
    amazon_testset = OfficeDataset(data_base_path, 'amazon', transform=transform_test, train=False)
    # caltech
    caltech_trainset = OfficeDataset(data_base_path, 'caltech', transform=transform_office)
    caltech_testset = OfficeDataset(data_base_path, 'caltech', transform=transform_test, train=False)
    # dslr
    dslr_trainset = OfficeDataset(data_base_path, 'dslr', transform=transform_office)
    dslr_testset = OfficeDataset(data_base_path, 'dslr', transform=transform_test, train=False)
    # webcam
    webcam_trainset = OfficeDataset(data_base_path, 'webcam', transform=transform_office)
    webcam_testset = OfficeDataset(data_base_path, 'webcam', transform=transform_test, train=False)

    min_data_len = min(len(amazon_trainset), len(caltech_trainset), len(dslr_trainset), len(webcam_trainset))
    val_len = int(min_data_len * 0.4)
    min_data_len = int(min_data_len * 0.5)

    print("Train data number: ", min_data_len)

    amazon_valset = torch.utils.data.Subset(amazon_trainset, list(range(len(amazon_trainset)))[-val_len:]) 
    amazon_trainset = torch.utils.data.Subset(amazon_trainset, list(range(min_data_len)))

    caltech_valset = torch.utils.data.Subset(caltech_trainset, list(range(len(caltech_trainset)))[-val_len:]) 
    caltech_trainset = torch.utils.data.Subset(caltech_trainset, list(range(min_data_len)))

    dslr_valset = torch.utils.data.Subset(dslr_trainset, list(range(len(dslr_trainset)))[-val_len:]) 
    dslr_trainset = torch.utils.data.Subset(dslr_trainset, list(range(min_data_len)))

    webcam_valset = torch.utils.data.Subset(webcam_trainset, list(range(len(webcam_trainset)))[-val_len:]) 
    webcam_trainset = torch.utils.data.Subset(webcam_trainset, list(range(min_data_len)))

    amazon_train_loader = torch.utils.data.DataLoader(amazon_trainset, batch_size=args.batch, shuffle=True)
    amazon_val_loader = torch.utils.data.DataLoader(amazon_valset, batch_size=args.batch, shuffle=False)
    amazon_test_loader = torch.utils.data.DataLoader(amazon_testset, batch_size=args.batch, shuffle=False)

    caltech_train_loader = torch.utils.data.DataLoader(caltech_trainset, batch_size=args.batch, shuffle=True)
    caltech_val_loader = torch.utils.data.DataLoader(caltech_valset, batch_size=args.batch, shuffle=False)
    caltech_test_loader = torch.utils.data.DataLoader(caltech_testset, batch_size=args.batch, shuffle=False)

    dslr_train_loader = torch.utils.data.DataLoader(dslr_trainset, batch_size=args.batch, shuffle=True)
    dslr_val_loader = torch.utils.data.DataLoader(dslr_valset, batch_size=args.batch, shuffle=False)
    dslr_test_loader = torch.utils.data.DataLoader(dslr_testset, batch_size=args.batch, shuffle=False)

    webcam_train_loader = torch.utils.data.DataLoader(webcam_trainset, batch_size=args.batch, shuffle=True)
    webcam_val_loader = torch.utils.data.DataLoader(webcam_valset, batch_size=args.batch, shuffle=False)
    webcam_test_loader = torch.utils.data.DataLoader(webcam_testset, batch_size=args.batch, shuffle=False)
    
    train_loaders = [amazon_train_loader, caltech_train_loader, dslr_train_loader, webcam_train_loader]
    val_loaders = [amazon_val_loader, caltech_val_loader, dslr_val_loader, webcam_val_loader]
    test_loaders = [amazon_test_loader, caltech_test_loader, dslr_test_loader, webcam_test_loader]

    return train_loaders, val_loaders, test_loaders


def prepare_data_office_partition_train(args):
    data_base_path = 'data'
    transform_train = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30,30)),
            transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.ToTensor(),
    ])

    K = 10
    # amazon
    print("Amazon: ")
    net_dataidx_map_train, _, train_ratio, _ = Dataset_partition_office('amazon', args.beta, split_test=False, n_parties=args.divide, min_require_size=args.min_img_num)
    net_dataidx_map_test = Adjust_test_dataset_office('amazon', train_ratio[0])
    amazon_trainset_all = OfficeDataset_sub(data_base_path, 'amazon', net_dataidx_map_train[0], transform=transform_train)
    proportions = np.asarray([1/6, 5/6])
    all_labels = [amazon_trainset_all[idx][1] for idx in range(len(amazon_trainset_all))]
    all_labels = np.asarray(all_labels)
    split_label = [[], []]
    for k in range(K):
        all_idx_k = np.where(all_labels == k)[0]
        np.random.shuffle(all_idx_k)
        proportions_all = (np.cumsum(proportions) * len(all_idx_k)).astype(int)[:-1]
        split_label = [idx_j + idx.tolist() for idx_j, idx in zip(split_label, np.split(all_idx_k, proportions_all))]
    amazon_trainset = torch.utils.data.Subset(amazon_trainset_all, split_label[0])
    amazon_testset = OfficeDataset_sub(data_base_path, 'amazon', net_dataidx_map_test, transform=transform_test, train=False)

    # caltech
    print("Caltech: ")
    net_dataidx_map_train, _, train_ratio, _ = Dataset_partition_office('caltech', args.beta, split_test=False, n_parties=args.divide, min_require_size=args.min_img_num)
    net_dataidx_map_test = Adjust_test_dataset_office('caltech', train_ratio[0])
    caltech_trainset_all = OfficeDataset_sub(data_base_path, 'caltech', net_dataidx_map_train[0], transform=transform_train)
    proportions = np.asarray([1/7, 6/7])
    all_labels = [caltech_trainset_all[idx][1] for idx in range(len(caltech_trainset_all))]
    all_labels = np.asarray(all_labels)
    split_label = [[], []]
    for k in range(K):
        all_idx_k = np.where(all_labels == k)[0]
        np.random.shuffle(all_idx_k)
        proportions_all = (np.cumsum(proportions) * len(all_idx_k)).astype(int)[:-1]
        split_label = [idx_j + idx.tolist() for idx_j, idx in zip(split_label, np.split(all_idx_k, proportions_all))]
    caltech_trainset = torch.utils.data.Subset(caltech_trainset_all, split_label[0])
    caltech_testset = OfficeDataset_sub(data_base_path, 'caltech', net_dataidx_map_test, transform=transform_test, train=False)
    
    # dslr
    print("dslr: ")
    net_dataidx_map_train, _, train_ratio, _ = Dataset_partition_office('dslr', args.beta, split_test=False, n_parties=args.divide, min_require_size=args.min_img_num)
    if args.beta in {0.3, 1.0}:
        dslr_trainset = OfficeDataset_sub(data_base_path, 'dslr', net_dataidx_map_train[1], transform=transform_train)
    else:
        dslr_trainset = OfficeDataset_sub(data_base_path, 'dslr', net_dataidx_map_train[0], transform=transform_train)
    dslr_testset = OfficeDataset(data_base_path, 'dslr', transform=transform_test, train=False)
    
    # webcam
    print("Webcam: ")
    net_dataidx_map_train, _, train_ratio, _ = Dataset_partition_office('webcam', args.beta, split_test=False, n_parties=args.divide, min_require_size=args.min_img_num)
    webcam_trainset_all = OfficeDataset_sub(data_base_path, 'webcam', net_dataidx_map_train[0], transform=transform_train)
    proportions = np.asarray([0.5, 0.5])
    all_labels = [webcam_trainset_all[idx][1] for idx in range(len(webcam_trainset_all))]
    all_labels = np.asarray(all_labels)
    split_label = [[], []]
    for k in range(K):
        all_idx_k = np.where(all_labels == k)[0]
        np.random.shuffle(all_idx_k)
        proportions_all = (np.cumsum(proportions) * len(all_idx_k)).astype(int)[:-1]
        split_label = [idx_j + idx.tolist() for idx_j, idx in zip(split_label, np.split(all_idx_k, proportions_all))]
    webcam_trainset = torch.utils.data.Subset(webcam_trainset_all, split_label[0])
    webcam_testset = OfficeDataset(data_base_path, 'webcam', transform=transform_test, train=False)

    data_num_list = [len(amazon_trainset), len(caltech_trainset), len(dslr_trainset), len(webcam_trainset)]
    min_data_len = min(len(amazon_trainset), len(caltech_trainset), len(dslr_trainset), len(webcam_trainset))
    print((len(amazon_trainset), len(caltech_trainset), len(dslr_trainset), len(webcam_trainset)))

    unq, unq_cnt = np.unique([amazon_trainset[x][1] for x in range(len(amazon_trainset))], return_counts=True)
    print("Train Amazon: ", {unq[i]: unq_cnt[i] for i in range(len(unq))})
    unq, unq_cnt = np.unique([amazon_testset[x][1] for x in range(len(amazon_testset))], return_counts=True)
    print("Test Amazon: ", {unq[i]: unq_cnt[i] for i in range(len(unq))})

    unq, unq_cnt = np.unique([caltech_trainset[x][1] for x in range(len(caltech_trainset))], return_counts=True)
    print("Train Caltech: ", {unq[i]: unq_cnt[i] for i in range(len(unq))})
    unq, unq_cnt = np.unique([caltech_testset[x][1] for x in range(len(caltech_testset))], return_counts=True)
    print("Test Caltech: ", {unq[i]: unq_cnt[i] for i in range(len(unq))})

    unq, unq_cnt = np.unique([dslr_trainset[x][1] for x in range(len(dslr_trainset))], return_counts=True)
    print("Train dslr: ", {unq[i]: unq_cnt[i] for i in range(len(unq))})
    unq, unq_cnt = np.unique([dslr_testset[x][1] for x in range(len(dslr_testset))], return_counts=True)
    print("Test dslr: ", {unq[i]: unq_cnt[i] for i in range(len(unq))})

    unq, unq_cnt = np.unique([webcam_trainset[x][1] for x in range(len(webcam_trainset))], return_counts=True)
    print("Train webcam: ", {unq[i]: unq_cnt[i] for i in range(len(unq))})
    unq, unq_cnt = np.unique([webcam_testset[x][1] for x in range(len(webcam_testset))], return_counts=True)
    print("Test webcam: ", {unq[i]: unq_cnt[i] for i in range(len(unq))})


    amazon_train_loader = torch.utils.data.DataLoader(amazon_trainset, batch_size=32, shuffle=True)
    amazon_test_loader  = torch.utils.data.DataLoader(amazon_testset, batch_size=32, shuffle=False)

    caltech_train_loader = torch.utils.data.DataLoader(caltech_trainset, batch_size=32, shuffle=True)
    caltech_test_loader = torch.utils.data.DataLoader(caltech_testset, batch_size=32, shuffle=False)

    dslr_train_loader = torch.utils.data.DataLoader(dslr_trainset, batch_size=32, shuffle=True)
    dslr_test_loader = torch.utils.data.DataLoader(dslr_testset, batch_size=32, shuffle=False)

    webcam_train_loader = torch.utils.data.DataLoader(webcam_trainset, batch_size=32, shuffle=True)
    webcam_test_loader = torch.utils.data.DataLoader(webcam_testset, batch_size=32, shuffle=False)
    
    train_loaders = [amazon_train_loader, caltech_train_loader, dslr_train_loader, webcam_train_loader]
    test_loaders = [amazon_test_loader, caltech_test_loader, dslr_test_loader, webcam_test_loader]
    return train_loaders, test_loaders, data_num_list


def prepare_data_digits(args):
    # Prepare data
    transform_mnist = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_svhn = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_usps = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_synth = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_mnistm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # MNIST
    mnist_trainset     = DigitsDataset(data_path="data/digits/MNIST", channels=1, percent=args.percent, train=True,  transform=transform_mnist)
    mnist_testset      = DigitsDataset(data_path="data/digits/MNIST", channels=1, percent=args.percent, train=False, transform=transform_mnist)

    # SVHN
    svhn_trainset      = DigitsDataset(data_path='data/digits/SVHN', channels=3, percent=args.percent,  train=True,  transform=transform_svhn)
    svhn_testset       = DigitsDataset(data_path='data/digits/SVHN', channels=3, percent=args.percent,  train=False, transform=transform_svhn)

    # USPS
    usps_trainset      = DigitsDataset(data_path='data/digits/USPS', channels=1, percent=args.percent,  train=True,  transform=transform_usps)
    usps_testset       = DigitsDataset(data_path='data/digits/USPS', channels=1, percent=args.percent,  train=False, transform=transform_usps)

    # Synth Digits
    synth_trainset     = DigitsDataset(data_path='data/digits/SynthDigits/', channels=3, percent=args.percent,  train=True,  transform=transform_synth)
    synth_testset      = DigitsDataset(data_path='data/digits/SynthDigits/', channels=3, percent=args.percent,  train=False, transform=transform_synth)

    # MNIST-M
    mnistm_trainset    = DigitsDataset(data_path='data/digits/MNIST_M/', channels=3, percent=args.percent,  train=True,  transform=transform_mnistm)
    mnistm_testset     = DigitsDataset(data_path='data/digits/MNIST_M/', channels=3, percent=args.percent,  train=False, transform=transform_mnistm)

    min_data_len = min(len(mnist_trainset), len(svhn_trainset), len(usps_trainset), len(synth_trainset), len(mnistm_trainset))
    print("Train data number: ", min_data_len)

    mnist_train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=args.batch, shuffle=True)
    mnist_test_loader  = torch.utils.data.DataLoader(mnist_testset, batch_size=args.batch, shuffle=False)
    svhn_train_loader = torch.utils.data.DataLoader(svhn_trainset, batch_size=args.batch,  shuffle=True)
    svhn_test_loader = torch.utils.data.DataLoader(svhn_testset, batch_size=args.batch, shuffle=False)
    usps_train_loader = torch.utils.data.DataLoader(usps_trainset, batch_size=args.batch,  shuffle=True)
    usps_test_loader = torch.utils.data.DataLoader(usps_testset, batch_size=args.batch, shuffle=False)
    synth_train_loader = torch.utils.data.DataLoader(synth_trainset, batch_size=args.batch,  shuffle=True)
    synth_test_loader = torch.utils.data.DataLoader(synth_testset, batch_size=args.batch, shuffle=False)
    mnistm_train_loader = torch.utils.data.DataLoader(mnistm_trainset, batch_size=args.batch,  shuffle=True)
    mnistm_test_loader = torch.utils.data.DataLoader(mnistm_testset, batch_size=args.batch, shuffle=False)

    train_loaders = [mnist_train_loader, svhn_train_loader, usps_train_loader, synth_train_loader, mnistm_train_loader]
    test_loaders  = [mnist_test_loader, svhn_test_loader, usps_test_loader, synth_test_loader, mnistm_test_loader]

    return train_loaders, test_loaders


def prepare_data_digits_partition_train(args):
    # Prepare data
    transform_mnist = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_svhn = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_usps = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_synth = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_mnistm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # MNIST
    print("MNIST: ")
    net_dataidx_map_train, _, train_ratio, _ = Dataset_partition_digits("data/digits/MNIST", args.beta, split_test=False, n_parties=args.divide, min_require_size=args.min_img_num)
    net_dataidx_map_test = Adjust_test_dataset_digits("data/digits/MNIST", train_ratio[0])
    mnist_trainset     = DigitsDataset_sub("data/digits/MNIST", 1, net_dataidx_map_train[0], percent=0.5, train=True,  transform=transform_mnist)
    mnist_testset      = DigitsDataset_sub("data/digits/MNIST", 1, net_dataidx_map_test, percent=args.percent, train=False, transform=transform_mnist)

    # SVHN
    print("SVHN: ")
    net_dataidx_map_train, _, train_ratio, _ = Dataset_partition_digits("data/digits/SVHN", args.beta, split_test=False, n_parties=args.divide, min_require_size=args.min_img_num)
    net_dataidx_map_test = Adjust_test_dataset_digits("data/digits/SVHN", train_ratio[0])
    svhn_trainset      = DigitsDataset_sub('data/digits/SVHN', 3, net_dataidx_map_train[0], percent=0.5,  train=True,  transform=transform_svhn)
    svhn_testset       = DigitsDataset_sub('data/digits/SVHN', 3, net_dataidx_map_test, percent=args.percent,  train=False, transform=transform_svhn)

    # USPS
    print("USPS: ")
    net_dataidx_map_train, _, train_ratio, _ = Dataset_partition_digits("data/digits/USPS", args.beta, split_test=False, n_parties=args.divide, min_require_size=args.min_img_num)
    net_dataidx_map_test = Adjust_test_dataset_digits('data/digits/USPS', train_ratio[0])
    usps_trainset      = DigitsDataset_sub('data/digits/USPS', 1, net_dataidx_map_train[0], percent=0.5, train=True,  transform=transform_usps)
    usps_testset       = DigitsDataset_sub('data/digits/USPS', 1, net_dataidx_map_test, percent=args.percent,  train=False, transform=transform_usps)

    # Synth Digits
    print("Synth Digits: ")
    net_dataidx_map_train, _, train_ratio, _ = Dataset_partition_digits("data/digits/SynthDigits/", args.beta, split_test=False, n_parties=args.divide, min_require_size=args.min_img_num)
    net_dataidx_map_test = Adjust_test_dataset_digits('data/digits/SynthDigits/', train_ratio[0])
    synth_trainset     = DigitsDataset_sub('data/digits/SynthDigits/', 3, net_dataidx_map_train[0], percent=0.5,  train=True,  transform=transform_synth)
    synth_testset      = DigitsDataset_sub('data/digits/SynthDigits/', 3, net_dataidx_map_test, percent=args.percent,  train=False, transform=transform_synth)

    # MNIST-M
    print("MNIST-M: ")
    net_dataidx_map_train, _, train_ratio, _ = Dataset_partition_digits("data/digits/MNIST_M/", args.beta, split_test=False, n_parties=args.divide, min_require_size=args.min_img_num)
    net_dataidx_map_test = Adjust_test_dataset_digits("data/digits/MNIST_M/", train_ratio[0])
    mnistm_trainset    = DigitsDataset_sub('data/digits/MNIST_M/', 3, net_dataidx_map_train[0],  percent=0.5,  train=True,  transform=transform_mnistm)
    mnistm_testset     = DigitsDataset_sub('data/digits/MNIST_M/', 3, net_dataidx_map_test, percent=args.percent,  train=False, transform=transform_mnistm)

    data_num_list = [len(mnist_trainset), len(svhn_trainset), len(usps_trainset), len(synth_trainset), len(mnistm_trainset)]
    min_data_len = min(len(mnist_trainset), len(svhn_trainset), len(usps_trainset), len(synth_trainset), len(mnistm_trainset))
    print(len(mnist_trainset), len(svhn_trainset), len(usps_trainset), len(synth_trainset), len(mnistm_trainset))

    unq, unq_cnt = np.unique([mnist_trainset[x][1] for x in range(len(mnist_trainset))], return_counts=True)
    print("Train mnist: ", {unq[i]: unq_cnt[i] for i in range(len(unq))})
    unq, unq_cnt = np.unique([mnist_testset[x][1] for x in range(len(mnist_testset))], return_counts=True)
    print("Test mnist: ", {unq[i]: unq_cnt[i] for i in range(len(unq))})

    unq, unq_cnt = np.unique([svhn_trainset[x][1] for x in range(len(svhn_trainset))], return_counts=True)
    print("Train svhn: ", {unq[i]: unq_cnt[i] for i in range(len(unq))})
    unq, unq_cnt = np.unique([svhn_testset[x][1] for x in range(len(svhn_testset))], return_counts=True)
    print("Test svhn: ", {unq[i]: unq_cnt[i] for i in range(len(unq))})

    unq, unq_cnt = np.unique([usps_trainset[x][1] for x in range(len(usps_trainset))], return_counts=True)
    print("Train usps: ", {unq[i]: unq_cnt[i] for i in range(len(unq))})
    unq, unq_cnt = np.unique([usps_testset[x][1] for x in range(len(usps_testset))], return_counts=True)
    print("Test usps: ", {unq[i]: unq_cnt[i] for i in range(len(unq))})

    unq, unq_cnt = np.unique([synth_trainset[x][1] for x in range(len(synth_trainset))], return_counts=True)
    print("Train synth: ", {unq[i]: unq_cnt[i] for i in range(len(unq))})
    unq, unq_cnt = np.unique([synth_testset[x][1] for x in range(len(synth_testset))], return_counts=True)
    print("Test synth: ", {unq[i]: unq_cnt[i] for i in range(len(unq))})

    unq, unq_cnt = np.unique([mnistm_trainset[x][1] for x in range(len(mnistm_trainset))], return_counts=True)
    print("Train mnistm: ", {unq[i]: unq_cnt[i] for i in range(len(unq))})
    unq, unq_cnt = np.unique([mnistm_testset[x][1] for x in range(len(mnistm_testset))], return_counts=True)
    print("Test mnistm: ", {unq[i]: unq_cnt[i] for i in range(len(unq))})
    # assert False

    mnist_train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=args.batch, shuffle=True)
    mnist_test_loader  = torch.utils.data.DataLoader(mnist_testset, batch_size=args.batch, shuffle=False)
    svhn_train_loader = torch.utils.data.DataLoader(svhn_trainset, batch_size=args.batch,  shuffle=True)
    svhn_test_loader = torch.utils.data.DataLoader(svhn_testset, batch_size=args.batch, shuffle=False)
    usps_train_loader = torch.utils.data.DataLoader(usps_trainset, batch_size=args.batch,  shuffle=True)
    usps_test_loader = torch.utils.data.DataLoader(usps_testset, batch_size=args.batch, shuffle=False)
    synth_train_loader = torch.utils.data.DataLoader(synth_trainset, batch_size=args.batch,  shuffle=True)
    synth_test_loader = torch.utils.data.DataLoader(synth_testset, batch_size=args.batch, shuffle=False)
    mnistm_train_loader = torch.utils.data.DataLoader(mnistm_trainset, batch_size=args.batch,  shuffle=True)
    mnistm_test_loader = torch.utils.data.DataLoader(mnistm_testset, batch_size=args.batch, shuffle=False)

    train_loaders = [mnist_train_loader, svhn_train_loader, usps_train_loader, synth_train_loader, mnistm_train_loader]
    test_loaders  = [mnist_test_loader, svhn_test_loader, usps_test_loader, synth_test_loader, mnistm_test_loader]
    return train_loaders, test_loaders, data_num_list


class DigitsDataset(Dataset):
    def __init__(self, data_path, channels, percent=0.1, filename=None, train=True, transform=None):
        if filename is None:
            if train:
                if percent >= 0.1:
                    for part in range(int(percent*10)):
                        if part == 0:
                            self.images, self.labels = np.load(os.path.join(data_path, 'partitions/train_part{}.pkl'.format(part)), allow_pickle=True)
                        else:
                            images, labels = np.load(os.path.join(data_path, 'partitions/train_part{}.pkl'.format(part)), allow_pickle=True)
                            self.images = np.concatenate([self.images,images], axis=0)
                            self.labels = np.concatenate([self.labels,labels], axis=0)
                else:
                    self.images, self.labels = np.load(os.path.join(data_path, 'partitions/train_part0.pkl'), allow_pickle=True)
                    data_len = int(self.images.shape[0] * percent*10)
                    self.images = self.images[:data_len]
                    self.labels = self.labels[:data_len]
            else:
                self.images, self.labels = np.load(os.path.join(data_path, 'test.pkl'), allow_pickle=True)
        else:
            self.images, self.labels = np.load(os.path.join(data_path, filename), allow_pickle=True)

        self.transform = transform
        self.channels = channels
        self.labels = self.labels.astype(np.long).squeeze()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.channels == 1:
            image = Image.fromarray(image, mode='L')
        elif self.channels == 3:
            image = Image.fromarray(image, mode='RGB')
        else:
            raise ValueError("{} channel is not allowed.".format(self.channels))

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class OfficeDataset(Dataset):
    def __init__(self, base_path, site, train=True, transform=None):
        if train:
            self.paths, self.text_labels = np.load('data/office_caltech_10/{}_train.pkl'.format(site), allow_pickle=True)
        else:
            self.paths, self.text_labels = np.load('data/office_caltech_10/{}_test.pkl'.format(site), allow_pickle=True)
            
        label_dict={'back_pack':0, 'bike':1, 'calculator':2, 'headphones':3, 'keyboard':4, 'laptop_computer':5, 'monitor':6, 'mouse':7, 'mug':8, 'projector':9}
        self.labels = [label_dict[text] for text in self.text_labels]
        self.transform = transform
        self.base_path = base_path if base_path is not None else '../data'

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.labels[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class DomainNetDataset(Dataset):
    def __init__(self, base_path, site, train=True, transform=None):
        if train:
            self.paths, self.text_labels = np.load('data/DomainNet/{}_train.pkl'.format(site), allow_pickle=True)
        else:
            self.paths, self.text_labels = np.load('data/DomainNet/{}_test.pkl'.format(site), allow_pickle=True)
            
        label_dict = {'bird':0, 'feather':1, 'headphones':2, 'ice_cream':3, 'teapot':4, 'tiger':5, 'whale':6, 'windmill':7, 'wine_glass':8, 'zebra':9}     
        self.labels = [label_dict[text] for text in self.text_labels]
        self.transform = transform
        self.base_path = base_path if base_path is not None else '../data'

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.labels[idx]
        image = Image.open(img_path)
        
        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    return net_cls_counts


def Dataset_partition_domainnet(site, beta, split_test=True, n_parties=5, min_require_size=2):
    min_size = 0
    K = 10
    # np.random.seed(2023)

    _, train_text_labels = np.load('data/DomainNet/{}_train.pkl'.format(site), allow_pickle=True)    
    label_dict = {'bird':0, 'feather':1, 'headphones':2, 'ice_cream':3, 'teapot':4, 'tiger':5, 'whale':6, 'windmill':7, 'wine_glass':8, 'zebra':9}     
    train_labels = np.asarray([label_dict[text] for text in train_text_labels])
    N_train = train_labels.shape[0]
    net_dataidx_map_train = {}

    if split_test:
        _, test_text_labels = np.load('data/DomainNet/{}_test.pkl'.format(site), allow_pickle=True)
        test_labels = np.asarray([label_dict[text] for text in test_text_labels])
        N_test = test_labels.shape[0]
        net_dataidx_map_test = {}

    while min_size < min_require_size:
        idx_batch_train = [[] for _ in range(n_parties)]
        if split_test:
            idx_batch_test = [[] for _ in range(n_parties)]
        for k in range(K):
            if k == 0:
                min_size_last = 10000
            train_idx_k = np.where(train_labels == k)[0]
            np.random.shuffle(train_idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            proportions = np.array([p * (len(idx_j) < N_train / n_parties) for p, idx_j in zip(proportions, idx_batch_train)])
            proportions = proportions / proportions.sum()
            proportions_train = (np.cumsum(proportions) * len(train_idx_k)).astype(int)[:-1]
            train_part_list = np.split(train_idx_k, proportions_train)
            idx_batch_train = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_train, train_part_list)]
            min_size_train = min([len(idx_j) for idx_j in train_part_list])

            if split_test:
                test_idx_k = np.where(test_labels == k)[0]
                np.random.shuffle(test_idx_k)
                proportions_test = (np.cumsum(proportions) * len(test_idx_k)).astype(int)[:-1]
                test_part_list = np.split(test_idx_k, proportions_test)
                idx_batch_test = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_test, test_part_list)]   
                min_size_test = min([len(idx_j) for idx_j in test_part_list])
                min_size = min(min_size_train, min_size_test, min_size_last)
            else:
                min_size = min(min_size_train, min_size_last)
            min_size_last = min_size

    for j in range(n_parties):
        np.random.shuffle(idx_batch_train[j])
        net_dataidx_map_train[j] = idx_batch_train[j]
        if split_test:
            np.random.shuffle(idx_batch_test[j])
            net_dataidx_map_test[j] = idx_batch_test[j]

    traindata_cls_counts = record_net_data_stats(train_labels, net_dataidx_map_train)
    print("Training data split: ",traindata_cls_counts)
    if split_test:
        testdata_cls_counts = record_net_data_stats(test_labels, net_dataidx_map_test)
        print("Testing data split: ",testdata_cls_counts)
        return net_dataidx_map_train, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts
    else:
        return net_dataidx_map_train, None, traindata_cls_counts, None


def Dataset_partition_digits(data_path, beta, split_test=True, n_parties=5, min_require_size=2):
    min_size = 0
    K = 10
    # np.random.seed(2023)

    for part in range(n_parties):
        if part == 0:
            _, train_labels = np.load(os.path.join(data_path, 'partitions/train_part{}.pkl'.format(part)), allow_pickle=True)
        else:
            _, labels = np.load(os.path.join(data_path, 'partitions/train_part{}.pkl'.format(part)), allow_pickle=True)
            train_labels = np.concatenate([train_labels, labels], axis=0)
    N_train = train_labels.shape[0]
    net_dataidx_map_train = {}

    if split_test:
        _, test_labels = np.load(os.path.join(data_path, 'test.pkl'), allow_pickle=True)
        N_test = test_labels.shape[0]
        net_dataidx_map_test = {}

    while min_size < min_require_size:
        idx_batch_train = [[] for _ in range(n_parties)]
        if split_test:
            idx_batch_test = [[] for _ in range(n_parties)]
        for k in range(K):
            if k == 0:
                min_size_last = 10000
            train_idx_k = np.where(train_labels == k)[0]
            np.random.shuffle(train_idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            proportions = np.array([p * (len(idx_j) < N_train / n_parties) for p, idx_j in zip(proportions, idx_batch_train)])
            proportions = proportions / proportions.sum()
            proportions_train = (np.cumsum(proportions) * len(train_idx_k)).astype(int)[:-1]
            train_part_list = np.split(train_idx_k, proportions_train)
            idx_batch_train = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_train, train_part_list)]
            min_size_train = min([len(idx_j) for idx_j in train_part_list])

            if split_test:
                test_idx_k = np.where(test_labels == k)[0]
                np.random.shuffle(test_idx_k)
                proportions_test = (np.cumsum(proportions) * len(test_idx_k)).astype(int)[:-1]
                test_part_list = np.split(test_idx_k, proportions_test)
                idx_batch_test = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_test, test_part_list)]   
                min_size_test = min([len(idx_j) for idx_j in test_part_list])
                min_size = min(min_size_train, min_size_test, min_size_last)
            else:
                min_size = min(min_size_train, min_size_last)
            min_size_last = min_size

    for j in range(n_parties):
        np.random.shuffle(idx_batch_train[j])
        net_dataidx_map_train[j] = idx_batch_train[j]
        if split_test:
            np.random.shuffle(idx_batch_test[j])
            net_dataidx_map_test[j] = idx_batch_test[j]

    traindata_cls_counts = record_net_data_stats(train_labels, net_dataidx_map_train)
    print("Training data split: ",traindata_cls_counts)
    if split_test:
        testdata_cls_counts = record_net_data_stats(test_labels, net_dataidx_map_test)
        print("Testing data split: ",testdata_cls_counts)
        return net_dataidx_map_train, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts
    else:
        return net_dataidx_map_train, None, traindata_cls_counts, None


def Dataset_partition_office(site, beta, split_test=True, n_parties=5, min_require_size=2):
    min_size = 0
    K = 10
    # np.random.seed(2023)

    _, train_text_labels = np.load('data/office_caltech_10/{}_train.pkl'.format(site), allow_pickle=True)    
    label_dict={'back_pack':0, 'bike':1, 'calculator':2, 'headphones':3, 'keyboard':4, 'laptop_computer':5, 'monitor':6, 'mouse':7, 'mug':8, 'projector':9}     
    train_labels = np.asarray([label_dict[text] for text in train_text_labels])
    N_train = train_labels.shape[0]
    net_dataidx_map_train = {}

    if split_test:
        _, test_text_labels = np.load('data/office_caltech_10/{}_test.pkl'.format(site), allow_pickle=True)
        test_labels = np.asarray([label_dict[text] for text in test_text_labels])
        N_test = test_labels.shape[0]
        net_dataidx_map_test = {}

    while min_size < min_require_size:
        idx_batch_train = [[] for _ in range(n_parties)]
        if split_test:
            idx_batch_test = [[] for _ in range(n_parties)]
        for k in range(K):
            if k == 0:
                min_size_last = 10000
            train_idx_k = np.where(train_labels == k)[0]
            np.random.shuffle(train_idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            proportions = np.array([p * (len(idx_j) < N_train / n_parties) for p, idx_j in zip(proportions, idx_batch_train)])
            proportions = proportions / proportions.sum()
            proportions_train = (np.cumsum(proportions) * len(train_idx_k)).astype(int)[:-1]
            train_part_list = np.split(train_idx_k, proportions_train)
            idx_batch_train = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_train, train_part_list)]
            min_size_train = min([len(idx_j) for idx_j in train_part_list])

            if split_test:
                test_idx_k = np.where(test_labels == k)[0]
                np.random.shuffle(test_idx_k)
                proportions_test = (np.cumsum(proportions) * len(test_idx_k)).astype(int)[:-1]
                test_part_list = np.split(test_idx_k, proportions_test)
                idx_batch_test = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_test, test_part_list)]   
                min_size_test = min([len(idx_j) for idx_j in test_part_list])
                min_size = min(min_size_train, min_size_test, min_size_last)
            else:
                min_size = min(min_size_train, min_size_last)
            min_size_last = min_size

    for j in range(n_parties):
        np.random.shuffle(idx_batch_train[j])
        net_dataidx_map_train[j] = idx_batch_train[j]
        if split_test:
            np.random.shuffle(idx_batch_test[j])
            net_dataidx_map_test[j] = idx_batch_test[j]

    traindata_cls_counts = record_net_data_stats(train_labels, net_dataidx_map_train)
    print("Training data split: ",traindata_cls_counts)
    if split_test:
        testdata_cls_counts = record_net_data_stats(test_labels, net_dataidx_map_test)
        print("Testing data split: ",testdata_cls_counts)
        return net_dataidx_map_train, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts
    else:
        return net_dataidx_map_train, None, traindata_cls_counts, None


def Adjust_test_dataset_domainnet(site, class_ratio):
    c_num = 10
    label_dict = {'bird':0, 'feather':1, 'headphones':2, 'ice_cream':3, 'teapot':4, 'tiger':5, 'whale':6, 'windmill':7, 'wine_glass':8, 'zebra':9}     
    _, test_text_labels = np.load('data/DomainNet/{}_test.pkl'.format(site), allow_pickle=True)
    test_labels = np.asarray([label_dict[text] for text in test_text_labels])
    unq, unq_cnt = np.unique([test_labels[x] for x in range(len(test_labels))], return_counts=True)
    test_class_ratio = {unq[i]: unq_cnt[i] for i in range(len(unq))}
    times = [test_class_ratio[x]/class_ratio[x] for x in range(c_num)]
    min_time = min(times)
    right_class_num = [int(min_time*class_ratio[x]) for x in range(c_num)]
    idx = []
    for k in range(c_num):
        test_idx_k = np.where(test_labels == k)[0]
        np.random.shuffle(test_idx_k)
        idx = idx + test_idx_k[:right_class_num[k]].tolist()
    unq, unq_cnt = np.unique([test_labels[x] for x in idx], return_counts=True)
    test_class_ratio = {unq[i]: unq_cnt[i] for i in range(len(unq))}
    return idx


def Adjust_test_dataset_digits(data_path, class_ratio):
    c_num = 10
    _, test_labels = np.load(os.path.join(data_path, 'test.pkl'), allow_pickle=True)
    unq, unq_cnt = np.unique([test_labels[x] for x in range(len(test_labels))], return_counts=True)
    test_class_ratio = {unq[i]: unq_cnt[i] for i in range(len(unq))}
    times = [test_class_ratio[x]/class_ratio[x] for x in range(c_num)]
    min_time = min(times)
    right_class_num = [int(min_time*class_ratio[x]) for x in range(c_num)]
    idx = []
    for k in range(c_num):
        test_idx_k = np.where(test_labels == k)[0]
        np.random.shuffle(test_idx_k)
        idx = idx + test_idx_k[:right_class_num[k]].tolist()
    unq, unq_cnt = np.unique([test_labels[x] for x in idx], return_counts=True)
    test_class_ratio = {unq[i]: unq_cnt[i] for i in range(len(unq))}
    return idx


def Adjust_test_dataset_office(site, class_ratio):
    c_num = 10
    label_dict={'back_pack':0, 'bike':1, 'calculator':2, 'headphones':3, 'keyboard':4, 'laptop_computer':5, 'monitor':6, 'mouse':7, 'mug':8, 'projector':9}
    _, test_text_labels = np.load('data/office_caltech_10/{}_test.pkl'.format(site), allow_pickle=True)
    test_labels = np.asarray([label_dict[text] for text in test_text_labels])
    unq, unq_cnt = np.unique([test_labels[x] for x in range(len(test_labels))], return_counts=True)
    test_class_ratio = {unq[i]: unq_cnt[i] for i in range(len(unq))}
    times = [test_class_ratio[x]/class_ratio[x] for x in range(c_num)]
    min_time = min(times)
    right_class_num = [int(min_time*class_ratio[x]) for x in range(c_num)]
    idx = []
    for k in range(c_num):
        test_idx_k = np.where(test_labels == k)[0]
        np.random.shuffle(test_idx_k)
        idx = idx + test_idx_k[:right_class_num[k]].tolist()
    unq, unq_cnt = np.unique([test_labels[x] for x in idx], return_counts=True)
    test_class_ratio = {unq[i]: unq_cnt[i] for i in range(len(unq))}
    return idx


class DigitsDataset_sub(Dataset):
    def __init__(self, data_path, channels, net_dataidx_map, percent=0.1, filename=None, train=True, transform=None):
        if filename is None:
            if train:
                if percent >= 0.1:
                    for part in range(int(percent*10)):
                        if part == 0:
                            self.images, self.labels = np.load(os.path.join(data_path, 'partitions/train_part{}.pkl'.format(part)), allow_pickle=True)
                        else:
                            images, labels = np.load(os.path.join(data_path, 'partitions/train_part{}.pkl'.format(part)), allow_pickle=True)
                            self.images = np.concatenate([self.images,images], axis=0)
                            self.labels = np.concatenate([self.labels,labels], axis=0)
                else:
                    self.images, self.labels = np.load(os.path.join(data_path, 'partitions/train_part0.pkl'), allow_pickle=True)
                    data_len = int(self.images.shape[0] * percent*10)
                    self.images = self.images[:data_len]
                    self.labels = self.labels[:data_len]
            else:
                self.images, self.labels = np.load(os.path.join(data_path, 'test.pkl'), allow_pickle=True)
        else:
            self.images, self.labels = np.load(os.path.join(data_path, filename), allow_pickle=True)

        self.images = self.images[net_dataidx_map]
        self.labels = self.labels[net_dataidx_map]
        self.transform = transform
        self.channels = channels
        self.labels = self.labels.astype(np.long).squeeze()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.channels == 1:
            image = Image.fromarray(image, mode='L')
        elif self.channels == 3:
            image = Image.fromarray(image, mode='RGB')
        else:
            raise ValueError("{} channel is not allowed.".format(self.channels))

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class OfficeDataset_sub(Dataset):
    def __init__(self, base_path, site, net_dataidx_map, train=True, transform=None):
        if train:
            self.paths, self.text_labels = np.load('data/office_caltech_10/{}_train.pkl'.format(site), allow_pickle=True)
        else:
            self.paths, self.text_labels = np.load('data/office_caltech_10/{}_test.pkl'.format(site), allow_pickle=True)
            
        label_dict={'back_pack':0, 'bike':1, 'calculator':2, 'headphones':3, 'keyboard':4, 'laptop_computer':5, 'monitor':6, 'mouse':7, 'mug':8, 'projector':9}
        self.labels = np.asarray([label_dict[text] for text in self.text_labels])
        self.labels = self.labels[net_dataidx_map]
        self.transform = transform
        self.base_path = base_path if base_path is not None else '../data'

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.labels[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class DomainNetDataset_sub(Dataset):
    def __init__(self, base_path, site, net_dataidx_map, train=True, transform=None):
        if train:
            self.paths, self.text_labels = np.load('data/DomainNet/{}_train.pkl'.format(site), allow_pickle=True)
        else:
            self.paths, self.text_labels = np.load('data/DomainNet/{}_test.pkl'.format(site), allow_pickle=True)
            
        label_dict = {'bird':0, 'feather':1, 'headphones':2, 'ice_cream':3, 'teapot':4, 'tiger':5, 'whale':6, 'windmill':7, 'wine_glass':8, 'zebra':9}     
        self.labels = np.asarray([label_dict[text] for text in self.text_labels])
        self.labels = self.labels[net_dataidx_map]
        self.transform = transform
        self.base_path = base_path if base_path is not None else '../data'

    def second_divide(self, partitions):
        self.labels = self.labels[partitions]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.labels[idx]
        image = Image.open(img_path)
        
        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

