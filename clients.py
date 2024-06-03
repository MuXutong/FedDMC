import random
from collections import OrderedDict
import torch.nn as nn
import numpy as np
import torch
from utils import *
from torch.utils.data import DataLoader
from getData import GetDataSet, ClientDataset


class client(object):
    def __init__(self, idx, trainDataSet, dev, distribution, nclass):
        self.id = idx
        self.train_ds = trainDataSet
        self.dev = dev
        self.train_DataLoader = None
        self.local_parameters = None
        self.distribution = distribution
        self.nclass = nclass

    def localUpdate_New(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters, malicious_clients,
                        byz_type, num_in_comm):

        Net.load_state_dict(global_parameters, strict=True)
        self.train_DataLoader = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)

        if self.id not in malicious_clients:
            for epoch in range(localEpoch):
                for data, label in self.train_DataLoader:
                    data, label = data.to(self.dev), label.to(self.dev)
                    preds = Net(data)

                    loss = lossFun(preds, label.long())
                    loss.backward()
                    opti.step()
                    opti.zero_grad()
            local_param = Net.state_dict()
        else:
            for epoch in range(localEpoch):
                for data, label in self.train_DataLoader:

                    if byz_type == 'Scaling_attack':
                        for example_id in range(data.shape[0] // 2):
                            data[example_id] = Adding_Trigger(data[example_id])
                            label[example_id] = 0

                    if byz_type == 'LF_attack':
                        label = self.nclass - 1 - label
                        # label = (random.randint(1, self.nclass-1) + label) % self.nclass

                    data, label = data.to(self.dev), label.to(self.dev)

                    preds = Net(data)

                    loss = lossFun(preds, label.long())
                    loss.backward()
                    opti.step()
                    opti.zero_grad()

            local_param = Net.state_dict()
            if byz_type == 'Scaling_attack':
                clip_rate = (num_in_comm / len(malicious_clients))/2
                for key, var in local_param.items():
                    global_value = global_parameters[key].to(self.dev)
                    new_value = global_value + (var - global_value) * clip_rate
                    local_param[key].copy_(new_value)

            if byz_type == 'GS_attack':
                # print('GS_attack')
                # grad_GS = OrderedDict()
                for key, var in local_param.items():
                    noise = torch.randn(var.shape).to(self.dev)
                    # var_GS = var + noise * torch.std(var) * 2
                    a = torch.mean(var.float())
                    b = torch.std(var.float())
                    var_GS = a + noise * b
                    # grad_GS[key] = var_GS
                    local_param[key].copy_(var_GS)
                # grad_param = grad_GS

        return local_param

    def localUpdate_backdoor(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters, alpha=0.2):

        Net.load_state_dict(global_parameters, strict=True)
        self.train_DataLoader = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)

        for epoch in range(localEpoch):
            for data, label in self.train_DataLoader:

                for example_id in range(data.shape[0]):
                    data[example_id] = Adding_Trigger(data[example_id])
                    label[example_id] = 0

                data, label = data.to(self.dev), label.to(self.dev)

                preds = Net(data)

                loss = lossFun(preds, label.long())
                dist_loss_func = nn.MSELoss()

                if alpha > 0:
                    dist_loss = 0
                    for key, var in Net.state_dict().items():
                        dist_loss += dist_loss_func(var, global_parameters[key].to(self.dev))
                    # for idx, p in enumerate(Net.parameters()):
                    #     dist_loss += dist_loss_func(p, global_parameters[idx])
                    # if torch.isnan(dist_loss):
                    #     raise Exception("Got nan dist loss")

                    loss += dist_loss * alpha

                loss.backward()
                opti.step()
                opti.zero_grad()

        local_param = Net.state_dict()

        return local_param


    def localUpdate(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters):
        '''
            param: localEpoch 当前Client的迭代次数
            param: localBatchSize 当前Client的batchsize大小
            param: Net Server共享的模型
            param: LossFun 损失函数
            param: opti 优化函数
            param: global_parmeters 当前通讯中最全局参数
            return: 返回当前Client基于自己的数据训练得到的新的模型参数
        '''
        # 加载当前通信中最新全局参数
        # 传入网络模型，并加载global_parameters参数的
        Net.load_state_dict(global_parameters, strict=True)
        # 载入Client自有数据集
        # 加载本地数据
        self.train_DataLoader = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)

        # 设置迭代次数
        for epoch in range(localEpoch):
            for data, label in self.train_DataLoader:
                # 加载到GPU上

                data, label = data.to(self.dev), label.to(self.dev)
                # 模型上传入数据
                preds = Net(data)
                # 计算损失函数
                '''
                    这里应该记录一下模型得损失值 写入到一个txt文件中
                '''

                loss = lossFun(preds, label.long())

                # 反向传播
                loss.backward()
                # 计算梯度，并更新梯度
                opti.step()
                # 将梯度归零，初始化梯度
                opti.zero_grad()
        # 返回当前Client基于自己的数据训练得到的新的模型参数
        return Net.state_dict()

    def localUpdate_LF(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters, malicious_clients,
                       client):

        Net.load_state_dict(global_parameters, strict=True)

        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)

        for epoch in range(localEpoch):
            for data, label in self.train_dl:

                if client in malicious_clients:
                    label = (random.randint(0, 9) + label) % 10

                # 加载到GPU上
                data, label = data.to(self.dev), label.to(self.dev)
                # 模型上传入数据
                preds = Net(data)
                # 计算损失函数
                '''
                    这里应该记录一下模型得损失值 写入到一个txt文件中
                '''
                loss = lossFun(preds, label)
                # 反向传播
                loss.backward()
                # 计算梯度，并更新梯度
                opti.step()
                # 将梯度归零，初始化梯度
                opti.zero_grad()
        # 返回当前Client基于自己的数据训练得到的新的模型参数

        return Net.state_dict()

    def localUpdate_GS(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters, malicious_clients):

        Net.load_state_dict(global_parameters, strict=True)

        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)

        for epoch in range(localEpoch):
            for data, label in self.train_dl:
                # 加载到GPU上
                data, label = data.to(self.dev), label.to(self.dev)
                # 模型上传入数据
                preds = Net(data)
                # 计算损失函数
                '''
                    这里应该记录一下模型得损失值 写入到一个txt文件中
                '''
                loss = lossFun(preds, label)
                # 反向传播
                loss.backward()
                # 计算梯度，并更新梯度
                opti.step()
                # 将梯度归零，初始化梯度
                opti.zero_grad()
        # 返回当前Client基于自己的数据训练得到的新的模型参数

        # param_grad = []
        # for param in Net.parameters():
        #     param_grad = param.data.view(-1) if not len(param_grad) else torch.cat(
        #         (param_grad, param.data.view(-1)))
        #
        # if client in malicious_clients:
        #     noise = torch.randn(param_grad).to(self.dev)
        #     param_grad = torch.mean(param_grad) + noise * torch.std(param_grad) * 2
        #
        # return param_grad

        grad = Net.state_dict()
        if self.id in malicious_clients:
            grad_GS = OrderedDict()
            for key, var in grad.items():
                noise = torch.randn(var.shape).to(self.dev)
                # var_GS = var + noise * torch.std(var) * 2
                a = torch.mean(var)
                b = torch.std(var)
                var_GS = a + noise * b * 2
                grad_GS[key] = var_GS
            grad = grad_GS

        return grad

    def local_val(self):
        pass


class ClientsGroup(object):
    '''
        param: dataSetName 数据集的名称
        param: isIID 是否是IID
        param: numOfClients 客户端的数量
        param: dev 设备(GPU)
        param: clients_set 客户端

    '''

    def __init__(self, dataSetName, isIID, num_clients, beta=0.4, datadir="./data/", dev=torch.device("cuda")):
        self.dev = dev
        self.data_set_name = dataSetName
        self.is_iid = isIID
        self.beta = beta
        self.datadir = datadir
        self.num_of_clients = num_clients
        self.clients_set = {}
        self.test_data_loader = None
        self.clients_distributions = None

        self.partition_data()
        self.clients_distribution()

    def partition_data(self):

        dataSet = GetDataSet(self.data_set_name, self.datadir)
        nclass = np.max(dataSet.train_label) + 1
        # 加载测试数据
        test_data = dataSet.test_data
        test_label = dataSet.test_label
        transform_test = dataSet.transform_test
        transform_train = dataSet.transform_train
        # 全局测试集
        self.test_data_loader = DataLoader(ClientDataset(test_data, test_label, transform_test), batch_size=256,
                                           shuffle=False)

        '''
            判断是否IID，如果不是，按照dirichlet的beta划分数据
        '''
        if self.is_iid:

            np.random.seed(12)

            idxs = np.random.permutation(dataSet.train_data_size)
            batch_idxs = np.array_split(idxs, self.num_of_clients)
            for i in range(self.num_of_clients):
                client_train_data = dataSet.train_data[batch_idxs[i]]
                client_train_label = dataSet.train_label[batch_idxs[i]]

                distribution = [client_train_label.tolist().count(i) for i in range(nclass)]

                # 为每一个clients 设置一个名字 client10
                self.clients_set['client{}'.format(i)] = client('client{}'.format(i),

                                                                ClientDataset(client_train_data,
                                                                              client_train_label,
                                                                              transform_train),
                                                                self.dev, distribution, nclass)

        else:
            n_clients = self.num_of_clients
            train_label = dataSet.train_label

            np.random.seed(123)
            label_distribution = np.random.dirichlet([self.beta] * n_clients, nclass)

            class_idcs = [np.argwhere(train_label == y).flatten() for y in range(nclass)]

            client_idcs = [[] for _ in range(n_clients)]

            for c, fracs in zip(class_idcs, label_distribution):
                # np.split按照比例将类别为k的样本划分为了N个子集
                # for i, idcs 为遍历第i个client对应样本集合的索引
                for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
                    client_idcs[i] += [idcs]

            for i in range(self.num_of_clients):
                idcs = client_idcs[i]
                distribution = [len(c) for c in idcs]

                client_train_data = dataSet.train_data[np.concatenate(idcs)]
                client_train_label = dataSet.train_label[np.concatenate(idcs)]

                self.clients_set['client{}'.format(i)] = client('client{}'.format(i),
                                                                ClientDataset(client_train_data,
                                                                              client_train_label,
                                                                              transform_train),
                                                                self.dev, distribution, nclass)

    def clients_distribution(self):
        distributions = {}
        for i in range(self.num_of_clients):
            distributions['client{}'.format(i)] = self.clients_set['client{}'.format(i)].distribution
        self.clients_distributions = distributions


if __name__ == "__main__":
    MyClients = ClientsGroup('mnist', isIID=1, beta=1, num_clients=100)
    # print(client)
    client = MyClients.clients_set['client0']

    print(client.id)

    # for i in range(100):
    #     client = MyClients.clients_set['client{}'.format(i)]
    #     print(client.train_ds)
    # train_ids = MyClients.clients_set['client10'].train_ds[0:10]
    # i = 0
    # for x_train in train_ids[0]:
    #     print("client10 数据:" + str(i))
    #     print(x_train)
    #     i = i + 1
    # print(MyClients.clients_set['client11'].train_ds[400:500])
