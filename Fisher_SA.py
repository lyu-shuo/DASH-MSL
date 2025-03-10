import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import numpy as np
import pickle
from models import *
from clients_datasets import *
from tqdm.notebook import tqdm
from utils import *
from AttFunc import *

def find_positions(fisher, alpha):
    fisher = np.array(fisher)

    percentile = 1-alpha
    percentile_value = np.percentile(fisher, percentile)
    mask = (fisher > percentile_value)

    positions = np.argwhere(mask)

    return positions.tolist()


def fmnist_fisher_perturbation(level, beta, bp, weight_positions, bias_positions, type=None):
    Beta = beta
    tensor_params = [param.data for param in bp]
    mp = tensor_params

    norms = [torch.norm(param) for param in tensor_params]
    units = [param/norm for param,norm in zip(tensor_params,norms)]
    signs = [torch.sign(param) for param in tensor_params]

    if level < 3:
        for index in range(len(weight_positions)):
            for position in weight_positions[index]:
                i, j, k, l = position
                if type == 'reverse':
                    mp[2*index][i][j][k][l] = -Beta * tensor_params[2*index][i][j][k][l]
                elif type == 'unit':
                    mp[2*index][i][j][k][l] = tensor_params[2*index][i][j][k][l] - Beta * units[2*index][i][j][k][l]
                elif type == 'sign':
                    mp[2*index][i][j][k][l] = tensor_params[2*index][i][j][k][l] - Beta * signs[2*index][i][j][k][l]
    else:
        for index in range(2):
            for position in weight_positions[index]:
                i, j, k, l = position
                if type == 'reverse':
                    mp[2*index][i][j][k][l] = -Beta * tensor_params[2*index][i][j][k][l]
                elif type == 'unit':
                    mp[2*index][i][j][k][l] = tensor_params[2*index][i][j][k][l] - Beta * units[2*index][i][j][k][l]
                elif type == 'sign':
                    mp[2*index][i][j][k][l] = tensor_params[2*index][i][j][k][l] - Beta * signs[2*index][i][j][k][l]
        # LeNet-5的后三层是全连接层，维度与卷积层不同，输入输出都是两维的
        for index in range(2,level):
            for position in weight_positions[index]:
                i, j = position
                if type == 'reverse':
                    mp[2*index][i][j] = -Beta * tensor_params[2*index][i][j]
                elif type == 'unit':
                    mp[2*index][i][j] = tensor_params[2*index][i][j] - Beta * units[2*index][i][j]
                elif type == 'sign':
                    mp[2*index][i][j] = tensor_params[2*index][i][j] - Beta * signs[2*index][i][j]
    for index in range(len(bias_positions)):
        for position in bias_positions[index]:
            if type == 'reverse':
                mp[2*index+1][i] = -Beta * tensor_params[2*index+1][i]
            elif type == 'unit':
                mp[2*index+1][i] = tensor_params[2*index+1][i] - Beta*units[2*index+1][i]
            elif type == 'sign':
                mp[2*index+1][i] = tensor_params[2*index+1][i] - Beta*signs[2*index+1][i]
    return mp



def fisher_simulated_annealing(level, beta, max_iters, att_type, acc0, fisher, testset):
    current_alpha = 1
    current_drop = 0

    best_alpha = 1
    best_drop = 0

    am = LeNet_5().cuda()
    am_params = torch.load('../Round_robin_SL/Round-Robin/results/f_mnist/am_params.pth')
    am.load_state_dict(am_params)

    for t in range(max_iters):
        new_alpha = current_alpha + np.random.normal(0, 0.1)
        new_alpha = np.clip(new_alpha, 0, 1)

        weight_positions = []
        bias_positions = []
        if level == 1:
            weight_positions.append(find_positions(fisher['conv_pool_1.0.weight'], new_alpha))
            bias_positions.append(find_positions(fisher['conv_pool_1.0.bias'], new_alpha))
        elif level == 2:
            weight_positions.append(find_positions(fisher['conv_pool_1.0.weight'], new_alpha))
            bias_positions.append(find_positions(fisher['conv_pool_1.0.bias'], new_alpha))
            weight_positions.append(find_positions(fisher['conv_pool_2.0.weight'], new_alpha))
            bias_positions.append(find_positions(fisher['conv_pool_2.0.bias'], new_alpha))
            # weight_positions.append(find_positions(fisher['conv1.weight'], new_alpha))
            # bias_positions.append(find_positions(fisher['conv1.bias'], new_alpha))
            # weight_positions.append(find_positions(fisher['conv2.weight'], new_alpha))
            # bias_positions.append(find_positions(fisher['conv2.bias'], new_alpha))

        new_mal_params = fmnist_fisher_perturbation(level, beta, list(am.parameters())[:2], weight_positions, bias_positions, type=att_type)
        # am_params['conv1.weight'] = new_mal_params[0]
        # am_params['conv1.bias'] = new_mal_params[1]
        am_params['conv_pool_1.0.weight'] = new_mal_params[0]
        am_params['conv_pool_1.0.bias'] = new_mal_params[1]

        new_drop = am_test(am, am_params, testset, acc0)

        if new_drop > current_drop or np.random.rand() < np.exp(-(new_drop - current_drop) / cooling_schedule(t)):
            current_alpha = new_alpha
            current_drop = new_drop

            if current_drop > best_drop:
                best_alpha = current_alpha
                best_drop = current_drop

    return best_alpha


##################################################################################################################

