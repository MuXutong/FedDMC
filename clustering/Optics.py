import numpy as np
import matplotlib.pyplot as plt
import operator
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


# 显示决策图
def plotReachability(data, eps):
    plt.figure()
    plt.plot(range(0, len(data)), data)
    plt.plot([0, len(data)], [eps, eps])
    plt.show()


# 显示分类的类别
def plotFeature(data, labels):
    clusterNum = len(set(labels))
    fig = plt.figure()
    scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown']
    ax = fig.add_subplot(111)
    for i in range(-1, clusterNum):
        colorSytle = scatterColors[i % len(scatterColors)]
        subCluster = data[np.where(labels == i)]
        ax.scatter(subCluster[:, 0], subCluster[:, 1], c=colorSytle, s=12)
    plt.show()


class OPTICS1(object):
    def __init__(self, data, eps=np.inf, minPts=15):
        self.data = data
        self.disMat = self.compute_squared_EDM(data)  # 获得距离矩阵
        self.number_sample = data.shape[0]
        self.eps = eps
        self.minPts = minPts
        self.core_distances = self.disMat[
            np.arange(0, self.number_sample), np.argsort(self.disMat)[:, minPts - 1]]  # 计算核心距离
        self.core_points_index = np.where(np.sum(np.where(self.disMat <= self.eps, 1, 0), axis=1) >= self.minPts)[0]

    # 计算距离矩阵
    def compute_squared_EDM(self, X):
        return squareform(pdist(X, metric='euclidean'))

    # 训练
    def train(self):
        # 初始化每一个点的最终可达距离(未定义)
        self.reach_dists = np.full((self.number_sample,), np.nan)
        self.orders = []  # 结果数组
        start_core_point = self.core_points_index[0]  # 从一个核心点开始
        # 标记数组
        isProcess = np.full((self.number_sample,), -1)

        # 训练
        isProcess[start_core_point] = 1
        # 选择一个核心点作为开始节点，并将其核心距离作为可达距离
        self.reach_dists[start_core_point] = self.core_distances[start_core_point]
        self.orders.append(start_core_point)  # 加入结果数组
        seeds = {}  # 种子数组，或者叫排序数组
        seeds = self.updateSeeds(seeds, start_core_point, isProcess)  # 更新排序数组
        while len(seeds) > 0:
            nextId = sorted(seeds.items(), key=operator.itemgetter(1))[0][0]  # 按可达距离排序，取第一个(最小的)
            del seeds[nextId]
            isProcess[nextId] = 1
            self.orders.append(nextId)  # 加入结果数组
            seeds = self.updateSeeds(seeds, nextId, isProcess)  # 更新种子数组和可达距离数组的可达距离

    # 更新可达距离
    def updateSeeds(self, seeds, core_PointId, isProcess):
        # 获得核心点core_PointId的核心距离
        core_dist = self.core_distances[core_PointId]
        # 计算所未访问的样本点更新可达距离
        for i in range(self.number_sample):
            if (isProcess[i] == -1):
                # 计算可达距离
                new_reach_dist = max(core_dist, self.disMat[core_PointId][i])
                if (np.isnan(self.reach_dists[i])):
                    # 可达矩阵更新
                    self.reach_dists[i] = new_reach_dist
                    seeds[i] = new_reach_dist
                elif (new_reach_dist < self.reach_dists[i]):
                    self.reach_dists[i] = new_reach_dist
                    seeds[i] = new_reach_dist
        return seeds

    # 生成label
    def predict(self):
        clusterId = 0
        self.labels = np.full((self.number_sample,), -1)
        for i in self.orders:
            if self.reach_dists[i] <= self.eps:
                self.labels[i] = clusterId
            else:
                if self.core_distances[i] <= self.eps:
                    clusterId += 1
                    self.labels[i] = clusterId


if __name__ == '__main__':
    data = np.loadtxt("r45_grad_data/n50_r45.csv", delimiter=",")
    OP = OPTICS1(data, 30, 10)
    OP.train()
    OP.predict()
    plotReachability(OP.reach_dists[OP.orders], 3)
    # plotFeature(data, OP.labels)
