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


# def fisher_perturbation(level, beta, bp, weight_positions, bias_positions, type=None):
#     Beta = beta
#     tensor_params = [param.data for param in bp]
#     mp = tensor_params
#
#     norms = [torch.norm(param) for param in tensor_params]
#     units = [param/norm for param,norm in zip(tensor_params,norms)]
#
#     signs = [torch.sign(param) for param in tensor_params]
#
#     if level < 3:
#         for index in range(len(weight_positions)):
#             for position in weight_positions[index]:
#                 i, j, k, l = position
#                 if type == 'unit':
#                     mp[2*index][i][j][k][l] = tensor_params[2*index][i][j][k][l] - Beta * units[2*index][i][j][k][l]
#                 elif type == 'sign':
#                     mp[2*index][i][j][k][l] = tensor_params[2*index][i][j][k][l] - Beta * signs[2*index][i][j][k][l]
#     else:
#         for index in range(2):
#             for position in weight_positions[index]:
#                 i, j, k, l = position
#                 if type == 'unit':
#                     mp[2*index][i][j][k][l] = tensor_params[2*index][i][j][k][l] - Beta * units[2*index][i][j][k][l]
#                 elif type == 'sign':
#                     mp[2*index][i][j][k][l] = tensor_params[2*index][i][j][k][l] - Beta * signs[2*index][i][j][k][l]
#         for index in range(2,level):
#             for position in weight_positions[index]:
#                 i, j = position
#                 if type == 'unit':
#                     mp[2*index][i][j] = tensor_params[2*index][i][j] - Beta * units[2*index][i][j]
#                 elif type == 'sign':
#                     mp[2*index][i][j] = tensor_params[2*index][i][j] - Beta * signs[2*index][i][j]
#     for index in range(len(bias_positions)):
#         for position in bias_positions[index]:
#             if type == 'unit':
#                 mp[2*index+1][i] = tensor_params[2*index+1][i] - Beta*units[2*index+1][i]
#             elif type == 'sign':
#                 mp[2*index+1][i] = tensor_params[2*index+1][i] - Beta*signs[2*index+1][i]
#
#     return mp


def fisher_perturbation(level, beta, bp, weight_positions, bias_positions, type=None):
    tensor_params = [param.data for param in bp]
    norms = [torch.norm(param) for param in tensor_params]
    units = [param / norm for param, norm in zip(tensor_params, norms)]
    signs = [torch.sign(param) for param in tensor_params]
    noises = [torch.randn_like(param) for param in tensor_params]
    gradients = [torch.autograd.grad(param.sum(), param, create_graph=True)[0] for param in bp]

    def update_params(params, positions, scale_factor, type):
        for index, pos_list in enumerate(positions):
            for pos in pos_list:
                if type == 'unit':
                    scale = units[2 * index]
                elif type == 'sign':
                    scale = signs[2 * index]
                elif type == 'random':
                    scale = noises[2 * index]
                elif type == 'grad':
                    scale = gradients[2 * index]
                params[2 * index][pos] -= beta * scale[pos]

    if level < 3:
        update_params(tensor_params, weight_positions, beta, type)
    else:
        update_params(tensor_params[:2], weight_positions[:2], beta, type)
        update_params(tensor_params[2:], weight_positions[2:], beta, type)

    def update_bias_params(params, positions, scale_factor, type):
        for index, pos_list in enumerate(positions):
            for pos in pos_list:
                if type == 'unit':
                    scale = units[2 * index + 1]
                elif type == 'sign':
                    scale = signs[2 * index + 1]
                elif type == 'random':
                    scale = noises[2 * index + 1]
                elif type == 'grad':
                    scale = gradients[2 * index + 1]
                params[2 * index + 1][pos] -= beta * scale[pos]

    update_bias_params(tensor_params, bias_positions, beta, type)

    return tensor_params





##################################################################################################################

