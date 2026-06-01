import numpy as np
import torch

def no_byz(v, net, lr, nfake, history, fixed_rand, init_model, last_50_model, last_grad, e, scaling_factor=100000.):
    return v, scaling_factor

def compute_lambda(all_updates, model_re, n_attackers):
    distances = []
    n_benign, d = all_updates.shape
    for update in all_updates:
        distance = torch.norm(all_updates - update, dim=1)
        distances.append(distance)
    distances = torch.stack(distances)

    distances = torch.sort(distances, dim=1).values
    scores = torch.sum(distances[:, :n_benign - 1 - n_attackers], dim=1)
    min_score = torch.min(scores)
    term_1 = min_score / ((n_benign - n_attackers - 1) * torch.sqrt(torch.tensor([d]))[0])
    max_wre_dist = torch.max(torch.norm(all_updates - model_re, dim=1)) / (torch.sqrt(torch.tensor([d]))[0])
    return term_1 + max_wre_dist

def score(gradient, v, nbyz):
    num_neighbours = v.shape[0] - 2 - nbyz
    sorted_distance = torch.square(v - gradient).sum(dim=1).sort().values
    return torch.sum(sorted_distance[1:(1 + num_neighbours)]).item()


def poisonedfl(v, net, lr, nfake, history, fixed_rand, init_model, last_50_model, last_grad, e, scaling_factor=100000.):
    if fixed_rand.shape[0] == 1204682:
        k_95 = 603244
        k_99 = 603618
    elif fixed_rand.shape[0] == 139960:
        k_95 = 70288
        k_99 = 70415
    elif fixed_rand.shape[0] == 717924:
        k_95 = 359659
        k_99 = 359948
    elif fixed_rand.shape[0] == 145212:
        k_95 = 72919
        k_99 = 73049
    else:
        k_95 = int(fixed_rand.shape[0] * 0.5)
        k_99 = int(fixed_rand.shape[0] * 0.6)
    sf = scaling_factor

    if isinstance(history, torch.Tensor):
        current_model = [param.data.clone() for param in net.parameters()]
        history_norm = torch.norm(history)
        last_grad_norm = torch.norm(last_grad)

        # Ensure all tensors are on the same device
        device = fixed_rand.device
        history = history.to(device)
        last_grad = last_grad.to(device)
        history_norm = history_norm.to(device)
        last_grad_norm = last_grad_norm.to(device)

        scale = torch.norm(history - torch.unsqueeze(last_grad, dim=-1) * history_norm / (last_grad_norm + 1e-9), dim=1).to(device)
        fixed_rand = fixed_rand.to(device)

        deviation = scale * fixed_rand / (torch.norm(scale) + 1e-9)
        if e % 50 == 0:
            total_update = torch.cat([param.view(-1, 1) for param in current_model], dim=0) - \
                           torch.cat([param.view(-1, 1) for param in last_50_model], dim=0)
            total_update = torch.where(total_update == 0,
                                        torch.cat([param.view(-1, 1) for param in current_model], dim=0),
                                        total_update)
            current_sign = torch.sign(total_update)

            # Ensure current_sign and fixed_rand are on the same device
            current_sign = current_sign.to(device)
            fixed_rand = fixed_rand.to(device)

            aligned_dim_cnt = (current_sign == torch.unsqueeze(fixed_rand, dim=-1)).sum()
            if aligned_dim_cnt < k_99 and scaling_factor * 0.7 >= 0.5:
                sf = scaling_factor * 0.7
            else:
                sf = scaling_factor
            lamda_succ = sf * history_norm
        else:
            sf = scaling_factor
            lamda_succ = sf * history_norm
        mal_update = lamda_succ * deviation
        for i in range(nfake):
            v[i] = torch.unsqueeze(mal_update, dim=-1)
    return v, sf


def random_attack(v, net, lr, nfake, history, fixed_rand, init_model, last_50_model, last_grad, e, scaling_factor=100000.):
    for i in range(nfake):
        v[i] = scaling_factor * torch.randn_like(v[0])
    return v, scaling_factor

def init_attack(v, net, lr, nfake, history, fixed_rand, init_model, last_50_model, last_grad, e, scaling_factor=100000.):
    current_model = [param.data.clone() for param in net.parameters()]
    direction = torch.cat([param.view(-1, 1) for param in init_model], dim=0) - \
                torch.cat([param.view(-1, 1) for param in current_model], dim=0)
    for i in range(nfake):
        v[i] = scaling_factor * direction
    return v, scaling_factor