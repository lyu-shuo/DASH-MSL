import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import numpy as np
import pickle


sys.path.append('D:\\Program\\MyCode\\Round_robin_SL\\Round-Robin')
from models import *
from clients_datasets import *
from tqdm.notebook import tqdm
from utils import *
from AttFunc import *
from Fisher_LeNet import *

from PoisonedFL.byzantine import poisonedfl


# 初始化攻击相关参数
history = None  # 初始化历史梯度
fixed_rand = torch.randn(1204682)  # 根据模型参数维度初始化
last_grad = None  # 初始化上一轮梯度
scaling_factor = 100000.0

for iter in tqdm(range(iters), desc="Training", unit="iter"):
    # 初始化训练
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

    # 训练
    mal_client_id = iter
    server.train()
    for i in range(NC):
        clients[i].train()
    server.apply(init_weights)
    clients[0].apply(init_weights)
    last_trained_params = clients[0].state_dict()
    for epoch in range(epochs):
        beta = beta_mean
        for idx, client in enumerate(clients):
            client.load_state_dict(last_trained_params)
            for j, data in enumerate(clients_trainloader[idx]):
                # 正常训练部分
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
            # 权重共享
            last_trained_params = client.state_dict()
            # 攻击部分
            if idx == mal_client_id:
                # 获取当前模型参数
                current_model = [param.data.clone() for param in client.parameters()]
                if history is None:
                    history = torch.cat([param.view(-1) for param in current_model])
                if last_grad is None:
                    last_grad = torch.cat([torch.zeros_like(param.view(-1)) for param in current_model])
                # 调用攻击函数
                mal_update, scaling_factor = poisonedfl(
                    v=[torch.zeros_like(param.view(-1)) for param in current_model],
                    net=client,
                    lr=0.01,
                    nfake=1,
                    history=history,
                    fixed_rand=fixed_rand,
                    init_model=current_model,
                    last_50_model=current_model,
                    last_grad=last_grad,
                    e=epoch,
                    scaling_factor=scaling_factor
                )
                # 更新恶意客户端参数
                idx = 0
                for param in client.parameters():
                    param.data += mal_update[idx:idx + param.numel()].view(param.size())
                    idx += param.numel()
                # 更新历史梯度
                last_grad = torch.cat([param.grad.view(-1) for param in client.parameters()])
    # 测试部分保持不变