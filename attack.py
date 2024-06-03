import copy
import numpy as np
import torch
from collections import OrderedDict

z_values = {3: 0.69847, 5: 0.7054, 8: 0.71904, 10: 0.72575, 12: 0.73891, 28: 0.48}


def LIT_attack(myClients, client_params, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters,
               malicious_clients):
    user_grads = []
    original_params = torch.cat([param.data.clone().view(-1) for key, param in global_parameters.items()], dim=0).cpu()

    learning_rate = opti.defaults['lr']

    for client in malicious_clients:
        local_parameters = client_params[client]
        local_grads = (original_params - local_parameters) / learning_rate
        user_grads = local_grads[None, :] if len(user_grads) == 0 else torch.cat(
            (user_grads, local_grads[None, :]), 0)

    grads_mean = torch.mean(user_grads, dim=0)
    grads_stdev = torch.std(user_grads, dim=0)
    # mal_params = avg + z_values[10] * std

    params_mean = original_params - learning_rate * grads_mean

    mal_net_params = train_malicious_network(myClients, params_mean, malicious_clients, localEpoch, localBatchSize, Net,
                                             lossFun, opti)

    new_params = mal_net_params + learning_rate * grads_mean
    new_grads = (params_mean - new_params) / learning_rate

    num_std = z_values[28]
    new_user_grads = np.clip(new_grads, grads_mean - num_std * grads_stdev,
                             grads_mean + num_std * grads_stdev)

    mal_params = original_params - learning_rate * new_user_grads


    for client in malicious_clients:

        client_params[client] = copy.deepcopy(mal_params)

    return client_params


def train_malicious_network(myClients, params_mean, malicious_clients, localEpoch, localBatchSize, Net, lossFun, opti):

    global_params = OrderedDict()
    # 将聚合后的参数对应到网络中
    start_idx = 0
    for key, var in Net.state_dict().items():
        param = params_mean[start_idx:start_idx + len(var.data.view(-1))].reshape(var.data.shape)
        start_idx = start_idx + len(var.data.view(-1))
        global_params[key] = copy.deepcopy(param)

    client_params = {}
    user_params = []
    alpha = 0.2
    for client in malicious_clients:
        local_param = myClients.clients_set[client].localUpdate_backdoor(localEpoch, 64, Net, lossFun, opti,
                                                                         global_params, alpha)
        local_param_flatten = torch.cat([param.data.clone().view(-1) for key, param in local_param.items()], dim=0).cpu()
        client_params[client] = copy.deepcopy(local_param_flatten)
        user_params = local_param_flatten[None, :] if len(user_params) == 0 else torch.cat(
            (user_params, local_param_flatten[None, :]), 0)

    params_mean = torch.mean(user_params, dim=0)
    # return client_params
    return params_mean
