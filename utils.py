import torch
import torch.nn as nn
import sys
import numpy as np
from models import *
from clients_datasets import *
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt


def data_process(data):
    data = np.array(data)

    std_multiplier = 0.5
    mean_value = np.mean(data)
    std_value = np.std(data)
    mask = (data >= mean_value - std_multiplier * std_value) & (data <= mean_value + std_multiplier * std_value)
    filtered_data = data[mask]
    outliers = data[~mask]

    filtered_mean = np.mean(filtered_data)
    data_min = np.min(filtered_data)
    data_max = np.max(filtered_data)
    data_range = data_max - data_min

    return filtered_mean, data_range


def init_weights(m):
    torch.manual_seed(7)
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            m.bias.data.zero_()
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            m.bias.data.zero_()
    if type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight, mean=1.0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def set_model_and_opt(dataset, NC):
    clients = []
    clients_opts = []

    if dataset in ['mnist', 'f_mnist']:
        server = LeNet_5().cuda()
        clients = [LeNet_5().cuda() for _ in range(NC)]

    elif dataset in  ['cifar10', 'svhn']:
        server = ResNet_9().cuda()
        clients = [ResNet_9().cuda() for _ in range(NC)]

    elif dataset == 'imagenette':
        server = VGG_16().cuda()
        clients = [VGG_16().cuda() for _ in range(NC)]

    elif dataset == 'cifar100':
        # server = ResNet_18().cuda()
        server = AlexNet(num_classes=100).cuda()
        # server = ResNet_9(num_classes=100).cuda()
        for i in range(NC):
            # clients.append(ResNet_18().cuda())
            clients.append(AlexNet(num_classes=100).cuda())
            # clients.append(ResNet_9(num_classes=100).cuda())

    server_opt = torch.optim.Adam(server.parameters(), lr=0.001, amsgrad=True, weight_decay=1e-4)
    clients_opts = [torch.optim.Adam(client.parameters(), lr=0.001, weight_decay=1e-4) for client in clients]

    return server, server_opt, clients, clients_opts


def set_graph(dataset, NC):
    if dataset == 'cora':
        server = GCN().cuda()
        server_opt = torch.optim.Adam(server.parameters(), lr=0.01, weight_decay=5e-4)
        clients = [GCN().cuda() for _ in range(NC)]
        clients_opts = [torch.optim.Adam(client.parameters(), lr=0.01, weight_decay=5e-4) for client in clients]

    
    return server, server_opt, clients, clients_opts