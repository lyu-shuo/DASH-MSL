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



def fedghost_attack(v, net, lr, nfake, history, fixed_rand, init_model, last_grad, e, scaling_factor):
    # 1. 拼接当前恶意客户端的初始模型参数为一维向量
    current_params_flat = torch.cat([param.view(-1) for param in init_model])

    # 2. 模拟 FedGhost 的无数据伪造梯度生成 (Data-Free Generation)
    # 论文通过固定随机扰动 fixed_rand 以及历史梯度 history 来生成对抗性伪造方向
    # 这里的噪声方向通常与历史梯度方向成一定夹角，或直接对冲全局梯度
    noise_direction = fixed_rand.to(current_params_flat.device)

    # 对齐维度并利用历史梯度信息和缩放因子计算投毒方向
    # 这里的公式示意 FedGhost 结合历史梯度与伪造噪声的更新生成
    poison_direction = -lr * (last_grad + 0.1 * noise_direction)

    # 3. 计算 FedGhost 的自适应缩放因子 (Adaptive Scaling Factor)
    if e > 0:
        # 动态自适应调整缩放比例，让恶意更新刚好处于防御机制的截断边缘
        scaling_factor = scaling_factor * 0.95

        # 4. 生成最终注入的恶意模型更新 (Model Update)
    mal_update = poison_direction * scaling_factor

    return mal_update, scaling_factor

# --- 初始化 FedGhost 攻击相关特定参数 ---
history = None  # 初始化历史模型/梯度记录
fixed_rand = torch.randn(1204682)
last_grad = None  # 初始化上一轮梯度
scaling_factor = 1.0  # FedGhost 初始缩放因子
iters = 10  # 大迭代次数

for iter in tqdm(range(iters), desc="FedGhost Training", unit="iter"):
    # 初始化标准训练环境参数
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

    # 设定当前轮次的恶意客户端 ID（比如每轮切一个，或固定某几个）
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

            # --- 1. 正常的前向与反向传播（训练部分） ---
            for j, data in enumerate(clients_trainloader[idx]):
                images, labels = data
                images, labels = images.cuda(), labels.cuda()

                smashed_data = client.forward(images, client_level=client_level)
                output = server.forward(smashed_data, server_level=server_level)

                clients_opts[idx].zero_grad()
                server_opt.zero_grad()

                loss = criterion(output, labels)
                loss.backward()

                clients_opts[idx].step()
                server_opt.step()

            # 权重共享/同步
            last_trained_params = client.state_dict()

            # --- 2. FedGhost 恶意投毒攻击部分 ---
            if idx == mal_client_id:
                # 提取并克隆当前客户端的模型参数
                current_model = [param.data.clone() for param in client.parameters()]

                # 初始化历史状态记录（仅在第一轮无数据时执行）
                if history is None:
                    history = torch.cat([param.view(-1) for param in current_model])
                if last_grad is None:
                    last_grad = torch.cat([torch.zeros_like(param.view(-1)) for param in current_model])

                # 调用 FedGhost 攻击函数生成恶意更新向量
                mal_update, scaling_factor = fedghost_attack(
                    v=[torch.zeros_like(param.view(-1)) for param in current_model],
                    net=client,
                    lr=0.01,
                    nfake=1,
                    history=history,
                    fixed_rand=fixed_rand,
                    init_model=current_model,
                    last_grad=last_grad,
                    e=epoch,
                    scaling_factor=scaling_factor
                )

                # 将生成的一维 mal_update 重新注入并覆盖到当前恶意客户端的模型参数中
                param_idx = 0
                for param in client.parameters():
                    # 截取对应长度的一维切片并 reshape 回原参数形状，叠加到 data 上
                    param.data += mal_update[param_idx: param_idx + param.numel()].view(param.size())
                    param_idx += param.numel()

                # 收集本轮攻击后的梯度状态，更新历史，留给下一轮攻击使用
                last_grad = torch.cat([param.grad.view(-1) if param.grad is not None
                                       else torch.zeros_like(param.view(-1))
                                       for param in client.parameters()])

                # 更新历史全局趋势记录（FedGhost 依靠此进行 Data-Free 的方向推导）
                history = 0.9 * history + 0.1 * torch.cat([param.view(-1) for param in current_model])
