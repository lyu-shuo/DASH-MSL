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


def clip_gradients(gradients, clip_value):
    clipped_gradients = []
    for grad in gradients:
        norm = torch.norm(grad)
        if norm > clip_value:
            clipped_grad = grad * (clip_value / norm)
        else:
            clipped_grad = grad
        clipped_gradients.append(clipped_grad)
    return clipped_gradients


'''
bp : benign parameters
mp : malicious parameters
'''
def perturbation(beta, bp, type=None):
    Beta = beta
    tensor_params = [param.data for param in bp]

    if type == 'unit':
        norms = [torch.norm(param) for param in tensor_params]
        units = [param / norm for param, norm in zip(tensor_params, norms)]
        mp = [param - Beta * unit for param, unit in zip(tensor_params, units)]
    elif type == 'sign':
        signs = [torch.sign(param) for param in tensor_params]
        mp = [param - Beta * sign for param, sign in zip(tensor_params, signs)]
    elif type == 'random':
        noise = [torch.randn_like(param) for param in tensor_params]
        mp = [param - Beta * n for param, n in zip(tensor_params, noise)]
    elif type == 'grad':
        gradients = [torch.autograd.grad(param.sum(), param, create_graph=True)[0] for param in bp]
        clipped_gradients = clip_gradients(gradients, 0.5)
        mp = [param - Beta * grad for param, grad in zip(tensor_params,clipped_gradients)]

    return mp



def am_train(am, trainset, dataset):
    am.train()
    am_opt = torch.optim.Adam(am.parameters(), lr=0.001, amsgrad=True, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    am.apply(init_weights)

    epochs = 100
    for epoch in tqdm(range(epochs), desc="Training attack model", unit="eopch"):
    # for epoch in range(epochs):
        for j, data in enumerate(trainset):
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            output = am.forward(images)
            am_opt.zero_grad()
            loss = criterion(output, labels)
            loss.backward()
            am_opt.step()
    am_params = am.state_dict()
    torch.save(am_params, f'../Round_robin_SL/Round-Robin/params/am_{dataset}_params.pth')


def am_test(am, am_params, testset, acc0):
    am.eval()
    am.load_state_dict(am_params)
    correct = 0
    total = 0
    acc1 = 0
    for data in testset:
        images, labels = data
        images, labels = images.cuda(), labels.cuda()
        output = am.forward(images)
        _, pre = torch.max(output.data, 1)
        total += images.shape[0]
        correct += (pre == labels).sum().item()
    acc1 = 100 * correct / total
    drop = acc0 - acc1

    return drop



###########################################################################################################
###########################################      SA        ###############################################
def cooling_schedule(t):
    initial_temperature = 1.0
    return initial_temperature / np.log(t + 1)


def simulated_annealing(dataset, init_beta, max_iters, att_type, acc0, testset):
    current_beta = init_beta
    current_drop = 0
    current_mal_params = None

    best_beta = init_beta
    best_drop = 0

    if dataset == 'mnist' or dataset == 'f_mnist':
        am = LeNet_5().cuda()
    elif dataset == 'cifar10' or dataset == 'svhn':
        am = ResNet_9().cuda()
    elif dataset == 'imagenette':
        am = VGG_16().cuda()
    elif dataset == 'cifar100':
        am = AlexNet(num_classes=100).cuda()
    am_params = torch.load(f'../Round_robin_SL/Round-Robin/params/am_{dataset}_params.pth')
    am.load_state_dict(am_params)

    for t in range(max_iters):
        new_beta = current_beta + np.random.normal(0, 0.1)
        new_beta = np.clip(new_beta, 0, 2)
        # if att_type == 'sign':
        #     # new_beta = np.clip(new_beta, 0, 1)
        #     # new_beta = np.clip(new_beta, 1, 2)
        #     new_beta = np.clip(new_beta, 0, 2)
        # elif att_type == 'unit':
        #     # new_beta = np.clip(new_beta, 1, 2)
        #     # new_beta = np.clip(new_beta, 0, 1)
        #     new_beta = np.clip(new_beta, 0, 2)
        #     # if dataset == 'cifar10':
        #     #     new_beta = np.clip(new_beta, 0, 1)

        new_mal_params = perturbation(new_beta, list(am.parameters())[:2], type=att_type)
        am_params['conv1.0.weight'] = new_mal_params[0]
        am_params['conv1.0.bias'] = new_mal_params[1]
        new_drop = am_test(am, am_params, testset, acc0)

        if new_drop > current_drop or np.random.rand() < np.exp(-(new_drop - current_drop) / cooling_schedule(t)):
            current_beta = new_beta
            current_drop = new_drop
            current_mal_params = new_mal_params

            if current_drop > best_drop:
                best_beta = current_beta
                best_drop = current_drop

    return best_beta





###########################################################################################################
###########################################      PS0        ###############################################
def pso_optimization(dataset, init_beta, max_iters, att_type, acc0, testset, num_particles=30):
    # 初始化粒子位置和速度
    particle_position = np.random.uniform(0, 2, num_particles)
    particle_velocity = np.zeros(num_particles)
    pbest_position = particle_position.copy()
    pbest_value = np.zeros(num_particles)
    gbest_position = init_beta
    gbest_value = 0
    gbest_mal_params = None

    if dataset == 'mnist' or dataset == 'f_mnist':
        am = LeNet_5().cuda()
    elif dataset == 'cifar10' or dataset == 'svhn':
        am = ResNet_9().cuda()
    am_params = torch.load(f'../Round_robin_SL/Round-Robin/params/am_{dataset}_params.pth')
    am.load_state_dict(am_params)

    w = 0.5  # 惯性权重
    c1 = 1.5  # 个人学习因子
    c2 = 1.5  # 社会学习因子

    for t in range(max_iters):
        for i in range(num_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            particle_velocity[i] = (
                w * particle_velocity[i] +
                c1 * r1 * (pbest_position[i] - particle_position[i]) +
                c2 * r2 * (gbest_position - particle_position[i])
            )
            particle_position[i] = particle_position[i] + particle_velocity[i]
            particle_position[i] = np.clip(particle_position[i], 0, 2)

            new_mal_params = perturbation(new_beta, list(am.parameters())[:2], type=att_type)
            am_params['conv1.0.weight'] = new_mal_params[0]
            am_params['conv1.0.bias'] = new_mal_params[1]

            current_drop = am_test(am, am_params, testset, acc0)

            if current_drop > pbest_value[i]:
                pbest_position[i] = particle_position[i]
                pbest_value[i] = current_drop

            if current_drop > gbest_value:
                gbest_position = particle_position[i]
                gbest_value = current_drop
                gbest_mal_params = new_mal_params

    return gbest_position