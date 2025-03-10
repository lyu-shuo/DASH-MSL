import torch
from torchvision import datasets, transforms
import torch.nn as nn
import random
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
from torch_geometric.transforms import NormalizeFeatures
import numpy as np


stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
cifar10_transform = transforms.Compose([transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         transforms.Normalize(*stats,inplace=True)])
cifar10_transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])

cifar100_transform_train = transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
cifar100_transform_test = transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


svhn_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

cinic10_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


imagenette_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomRotation(10),  # 随机旋转
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 颜色抖动
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_clients_trainsets(dataset, NC, batch_size, shuffle=True):
    if dataset == 'mnist':
        trainset = datasets.MNIST('data/', download=True, train=True, transform=transforms.ToTensor())
    elif dataset == 'f_mnist':
        trainset = datasets.FashionMNIST('data/', download=True, train=True, transform=transforms.ToTensor())
    elif dataset == 'cifar10':
        trainset = datasets.CIFAR10('data/', download=True, train=True, transform=cifar10_transform)
    elif dataset == 'emnist':
        trainset = datasets.EMNIST('data/', split='balanced', download=True, train=True, transform=transforms.ToTensor())
    elif dataset == 'svhn':
        trainset = datasets.SVHN('data/', split='train', download=True, transform=svhn_transform)
    elif dataset == 'imagenette':
        trainset = datasets.ImageFolder('D:\\Program\\MyCode\\Round_robin_SL\\Round-Robin/data/imagenette2/train', transform=imagenette_transform)
    elif dataset == 'cifar100':
        trainset = datasets.CIFAR100('data/', download=True, train=True, transform=cifar100_transform_train)
    elif dataset == 'cinic10':
        trainset = datasets.ImageFolder('D:\\Program\\MyCode\\Round_robin_SL\\Round-Robin/data/CINIC-10/train', transform=cinic10_transform)

    clients_trainsets = Divide_dataset_equally(NC, trainset)
    clients_trainloader = [[] for i in range(NC)]
    for i in range(NC):
        clients_trainloader[i] = torch.utils.data.DataLoader(clients_trainsets[i], shuffle=shuffle, batch_size=batch_size)

    return clients_trainloader


def load_clients_testsets(dataset, NC, batch_size):
    if dataset == 'mnist':
        testset = datasets.MNIST('data/', download=True, train=False, transform=transforms.ToTensor())
    elif dataset == 'f_mnist':
        testset = datasets.FashionMNIST('data/', download=True, train=False, transform=transforms.ToTensor())
    elif dataset == 'cifar10':
        testset = datasets.CIFAR10('data/', download=True, train=False, transform=cifar10_transform_test)
    elif dataset == 'emnist':
        testset = datasets.EMNIST('data/', split='balanced', download=True, train=False, transform=transforms.ToTensor())
    elif dataset == 'svhn':
        testset = datasets.SVHN('data/', split='test', download=True, transform=svhn_transform)
    elif dataset == 'imagenette':
        testset = datasets.ImageFolder('D:\\Program\\MyCode\\Round_robin_SL\\Round-Robin/data/imagenette2/val', transform=imagenette_transform)
    elif dataset == 'cifar100':
        testset = datasets.CIFAR100('data/', download=True, train=False, transform=cifar100_transform_test)
    elif dataset == 'cinic10':
        testset = datasets.ImageFolder('D:\\Program\\MyCode\\Round_robin_SL\\Round-Robin/data/CINIC-10/test', transform=cinic10_transform)

    clients_testsets = Divide_dataset_equally(NC, testset)
    clients_testloader = [[] for i in range(NC)]
    for i in range(NC):
        clients_testloader[i] = torch.utils.data.DataLoader(clients_testsets[i], shuffle=False, batch_size=batch_size)

    return clients_testloader


def Divide_dataset_equally(NC, dataset):
    divided_dataset = [[] for i in range(NC)]
    indices = list(range(len(dataset)))

    random.shuffle(indices)
    # 遍历整个数据集，将数据分配到子集中
    for index in indices:
        image, label = dataset[index]
        # 从0到9轮流分配数据
        subset_idx = index % NC
        divided_dataset[subset_idx].append((image, label))

    return divided_dataset

