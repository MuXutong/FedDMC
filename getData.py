import numpy as np
import gzip
import os
import platform
import pickle
import torchvision
from torchvision import transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
from torchvision.datasets import MNIST, EMNIST, CIFAR10

from PIL import Image

# from data.femnist import FEMNIST

DATA_nclass = {'mnist': 10, 'cifar10': 10, 'svhn': 10, 'fmnist': 10, 'celeba': 2,
               'cifar100': 100, 'tinyimagenet': 200, 'femnist': 62, 'emnist': 47,
               'xray': 2}


class ClientDataset(data.Dataset):

    def __init__(self, data, target, transform=None, target_transform=None):

        self.data = data
        self.target = target
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]


        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class GetDataSet(object):
    def __init__(self, dataSetName, datadir):
        self.name = dataSetName
        self.train_data = None  # 训练集
        self.train_label = None  # 标签
        self.train_data_size = None  # 训练数据的大小
        self.test_data = None  # 测试数据集
        self.test_label = None  # 测试的标签
        self.test_data_size = None  # 测试集数据Size
        self.transform_train = None
        self.transform_test = None

        self._index_in_train_epoch = 0

        # 如何数据集是mnist
        if self.name == 'mnist':
            self.load_mnist_data(datadir)
        elif self.name == 'femnist':
            self.load_femnist_data(datadir)
        elif self.name == 'emnist':
            self.load_emnist_data(datadir)
        elif self.name == 'cifar10':
            self.load_cifar10_data(datadir)
        else:
            raise NotImplementedError

    def load_mnist_data(self, datadir):
        train_set = MNIST(root=datadir, train=True, download=True)
        test_set = MNIST(root=datadir, train=False, download=True)

        train_data = train_set.data
        train_labels = np.array(train_set.targets)

        test_data = test_set.data  # 测试数据
        test_labels = np.array(test_set.targets)

        mean = (train_data.float().mean()) / (train_data.float().max() - train_data.float().min())
        std = (train_data.float().std()) / (train_data.float().max() - train_data.float().min())

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,))
        ])

        self.transform_train = transform
        self.transform_test = transform

        self.train_data = np.array(train_data)
        self.train_label = train_labels

        self.test_data = np.array(test_data)
        self.test_label = test_labels

        self.train_data_size = train_data.shape[0]
        self.test_data_size = test_data.shape[0]

    def load_emnist_data(self, datadir):

        train_set = EMNIST(root=datadir, split="byclass", train=True, download=True)
        test_set = EMNIST(root=datadir, split="byclass", train=False, download=True)

        train_data = train_set.data
        train_labels = np.array(train_set.targets)

        test_data = test_set.data  # 测试数据
        test_labels = np.array(test_set.targets)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1736,), (0.3317,))
        ])

        self.transform_train = transform
        self.transform_test = transform

        self.train_data = np.array(train_data)
        self.train_label = train_labels

        self.test_data = np.array(test_data)
        self.test_label = test_labels

        self.train_data_size = train_data.shape[0]
        self.test_data_size = test_data.shape[0]


    def load_cifar10_data(self, datadir, noise_level=0):

        train_set = torchvision.datasets.CIFAR10(root=datadir, train=True, download=False)
        test_set = torchvision.datasets.CIFAR10(root=datadir, train=False, download=False)

        train_data = train_set.data  # (50000, 32, 32, 3)
        train_labels = np.array(train_set.targets)  # 将标签转化为

        test_data = test_set.data  # 测试数据
        test_labels = np.array(test_set.targets)

        self.train_data = train_data
        self.train_label = train_labels

        self.test_data = test_data
        self.test_label = test_labels

        self.train_data_size = train_data.shape[0]
        self.test_data_size = test_data.shape[0]

        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(
                Variable(x.unsqueeze(0), requires_grad=False),
                (4, 4, 4, 4), mode='reflect').data.squeeze()),
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=noise_level),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])

        self.transform_train = transform_train
        self.transform_test = transform_test



if __name__ == "__main__":
    # 'test data set'
    mnistDataSet = GetDataSet('mnist', "./data/")

    print(mnistDataSet)
