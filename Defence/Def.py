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
import torch
import numpy as np


def calculate_fisher(client, params_to_consider=None):
    if params_to_consider is None:
        params_to_consider = [name for name, _ in client.named_parameters()]
    fisher_matrix = {}
    for param_name, param in client.named_parameters():
        if param_name in params_to_consider:
            grad = param.grad
            if grad is not None:
                fisher_matrix[param_name] = (grad.pow(2)).cpu().detach().numpy()
    return fisher_matrix


def calculate_taylor(client, params_to_consider=None):
    if params_to_consider is None:
        params_to_consider = [name for name, _ in client.named_parameters()]
    taylor_scores = {}
    for param_name, param in client.named_parameters():
        if param_name in params_to_consider:
            if param.grad is not None:
                # |param * grad|
                taylor_scores[param_name] = torch.abs(param * param.grad).cpu().detach().numpy()
    return taylor_scores


def find_important_positions(score_matrix, percentile=0.33):
    important_positions = {}
    for param_name, scores in score_matrix.items():
        threshold = np.percentile(scores, percentile * 100)
        positions = np.where(scores >= threshold)
        important_positions[param_name] = positions
    return important_positions


def check_and_replace(params, history_params, important_positions, threshold_multiplier=2):
    for param_name, positions in important_positions.items():
        param = params[param_name]
        history_param = history_params[param_name]
        if history_param is None:
            continue  # 无历史记录，跳过检测

        # 计算权重变化
        change = torch.abs(param - history_param)

        # 使用中位数和绝对中位差 (MAD) 计算阈值
        median_change = torch.median(change)
        mad_change = torch.median(torch.abs(change - median_change))

        # 动态阈值设定
        threshold = median_change + threshold_multiplier * mad_change
        threshold = torch.clamp(threshold, min=1e-5, max=1e5)

        # 找到变化超过阈值的权重位置
        anomalies = change > threshold
        num_anomalies = anomalies.sum().item()

        if num_anomalies > 0:
            print(f"Anomalies detected in {param_name}: {num_anomalies} elements, replacing with history values.")
            param[anomalies] = history_param[anomalies]

    return params


def update_history(history_weights, client_id, params_to_update):
    for param_name, param in params_to_update.items():
        history_weights[client_id][param_name] = param.clone()


def defence(client, history_weights, client_id, mode='Fisher', threshold_multiplier=2, percentile=0.33):
    """
    基于Fisher或Taylor方法检测并修复客户端参数中的异常值。

    参数:
        client (nn.Module): 客户端模型。
        history_weights (dict): 历史权重记录字典。
        client_id (int): 客户端ID。
        mode (str): 使用的评估方法，'Fisher' 或 'Taylor'。
        threshold_multiplier (float): 阈值倍数，用于检测异常值。
        percentile (float): 选择前多少百分比的参数作为重要参数。

    返回:
        dict: 修复后的参数字典。
    """
    # print(f"Defending client {client_id} using mode: {mode}")
    if mode == 'Fisher':
        # 计算Fisher信息矩阵
        score_matrix = calculate_fisher(client)
    elif mode == 'Taylor':
        # 计算Taylor展开的参数重要性
        score_matrix = calculate_taylor(client)
    else:
        raise ValueError("Unsupported mode. Choose 'Fisher' or 'Taylor'.")

    # 选择重要权重位置
    important_positions = find_important_positions(score_matrix, percentile)

    # 获取当前参数
    current_params = {param_name: param.data.clone() for param_name, param in client.named_parameters() if
                      param_name in important_positions}

    # 获取历史参数
    history_params = history_weights[client_id]

    # 检查并替换异常权重
    checked_params = check_and_replace(current_params, history_params, important_positions, threshold_multiplier)

    return checked_params


def get_benign_params_list(benign_params):
    """
    将benign_params字典转换为参数列表，以便传递给fisher_perturbation。

    参数:
        benign_params (dict): 包含参数名称和对应张量的字典。

    返回:
        list: 包含参数张量的列表。
    """
    return list(benign_params.values())

