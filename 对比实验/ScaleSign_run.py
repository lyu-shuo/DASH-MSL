import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import numpy as np
import pickle
from tqdm.notebook import tqdm

sys.path.append('D:\\Program\\MyCode\\Round_robin_SL\\Round-Robin')
from models import *
from clients_datasets import *
from utils import *
from AttFunc import *


def scalesign_attack(benign_grad, scaling_factor=2.0):
    mal_grad = benign_grad * scaling_factor
    mask = (torch.sign(mal_grad) != torch.sign(benign_grad))
    mal_grad[mask] = benign_grad[mask]
    return mal_grad


history = None
fixed_rand = torch.randn(1204682)
last_grad = None
scaling_factor = 2.0
iters = 10

for iter in tqdm(range(iters), desc="Training", unit="iter"):
    batch_size = 600
    epochs = 30
    NC = 10
    dataset = 'mnist'
    clients_trainloader = load_clients_trainsets(dataset, NC, batch_size)
    clients_testloader = load_clients_testsets(dataset, NC, batch_size)
    server, server_opt, clients, clients_opts = set_model_and_opt(dataset, NC)
    client_level = 1
    server_level = 4
    criterion = torch.nn.CrossEntropyLoss()

    mal_client_id = iter % NC
    server.train()
    for i in range(NC):
        clients[i].train()
    server.apply(init_weights)
    clients[0].apply(init_weights)
    last_trained_params = clients[0].state_dict()

    for epoch in range(epochs):
        for idx, client in enumerate(clients):
            client.load_state_dict(last_trained_params)
            for j, data in enumerate(clients_trainloader[idx]):
                images, labels = data
                images = images.cuda()
                labels = labels.cuda()
                smashed_data = client.forward(images, client_level=client_level)
                output = server.forward(smashed_data, server_level=server_level)
                clients_opts[idx].zero_grad()
                server_opt.zero_grad()
                loss = criterion(output, labels)
                loss.backward()
                clients_opts[idx].step()
                server_opt.step()

            last_trained_params = client.state_dict()

            if idx == mal_client_id:
                current_model = [param.data.clone() for param in client.parameters()]
                current_grads = torch.cat([param.grad.view(-1) if param.grad is not None
                                           else torch.zeros_like(param.view(-1))
                                           for param in client.parameters()])

                mal_update = scalesign_attack(current_grads, scaling_factor=scaling_factor)

                param_idx = 0
                for param in client.parameters():
                    param.data += mal_update[param_idx:param_idx + param.numel()].view(param.size())
                    param_idx += param.numel()

                last_grad = torch.cat([param.grad.view(-1) if param.grad is not None
                                       else torch.zeros_like(param.view(-1))
                                       for param in client.parameters()])