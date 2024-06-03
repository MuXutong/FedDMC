from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np
# from hdbscan import condense_tree
from hdbscan._hdbscan_tree import condense_tree
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import sklearn.datasets as data

features = pd.read_csv(r'H:\By-FL\clustering\r65_grad_data\n5_r65.csv', header=None)
test_data = np.array(features)[:10]

# moons, _ = data.make_moons(n_samples=50, noise=0.05)
# blobs, _ = data.make_blobs(n_samples=50, centers=[(-0.75, 2.25), (1.0, 2.0)], cluster_std=0.25)
# test_data = np.vstack([moons, blobs])

agglomer = AgglomerativeClustering(n_clusters=2, linkage='average', compute_distances=True)

agglomer.fit(test_data)

labels = agglomer.labels_
leaves = agglomer.n_leaves_
s = np.sum(labels)

counts = np.zeros(agglomer.children_.shape[0])
n_samples = len(agglomer.labels_)

children = agglomer.children_

for i, merge in enumerate(agglomer.children_):
    current_count = 0
    for child_idx in merge:
        if child_idx < n_samples:
            current_count += 1
        else:
            current_count += counts[child_idx - n_samples]
    counts[i] = current_count

d = agglomer.distances_
child = agglomer.children_

linkage_matrix = np.column_stack([agglomer.children_, agglomer.distances_, counts]).astype(float)

dendrogram(linkage_matrix)

plt.show()

# condensed_tree = condense_tree(linkage_matrix, 2)


class Node(object):
    def __init__(self, index, lchild=None, rchild=None, distances=None, counts=None):
        self.index = index  # 创建节点值
        self.lchild = lchild  # 创建左子树
        self.rchild = rchild  # 创建右子树
        self.distances = distances  # 两个子树直接的距离
        self.counts = counts  # 包含叶子节点数量
        self.leaves = []

    def postorder_travel(self, node):
        # 如果节点为空
        if node == None:
            return []

        self.postorder_travel(node.lchild)
        self.postorder_travel(node.rchild)
        if node.counts == 1:
            self.leaves.append(node.index)
        return self.leaves


cluster_id = n_samples

queue = {}
root = None

for child in linkage_matrix:
    if child[0] < n_samples:
        lchild = Node(child[0], counts=1)
    else:
        lchild = queue[child[0]]
        del queue[child[0]]

    if child[1] < n_samples:
        rchild = Node(child[1], counts=1)
    else:
        rchild = queue[child[1]]
        del queue[child[1]]

    root = Node(cluster_id, lchild, rchild, child[2], child[3])
    queue[cluster_id] = root

    cluster_id = cluster_id + 1

original_root = root
min_cluster_size = 4
outlier_all = []

while root.rchild.counts <= min_cluster_size or root.lchild.counts <= min_cluster_size:

    if root.rchild.counts >= min_cluster_size:

        # outlier = root.lchild.preorder()
        outlier = root.lchild.postorder_travel(root.lchild)
        root = root.rchild

    elif root.lchild.counts >= min_cluster_size:
        # 如果左孩子的叶子节点数量满足，则右孩子的叶子节点标位-1，然后从树中删除。
        outlier = root.rchild.postorder_travel(root.rchild)
        root = root.lchild

    else:
        outlier = root.postorder_travel(root)
        outlier_all.extend(outlier)
        root = None
        break
    outlier_all.extend(outlier)

# root中，左右孩子分别是良性和恶意用户
benign, malicious = [], []
if root:
    if root.rchild.counts < root.lchild.counts:
        malicious = root.rchild.postorder_travel(root.rchild)
        benign = root.lchild.postorder_travel(root.lchild)
    elif root.rchild.counts > root.lchild.counts:
        benign = root.rchild.postorder_travel(root.rchild)
        malicious = root.lchild.postorder_travel(root.lchild)
    else:
        benign = root.postorder_travel(root)

print("0")
# else:

# print(child)
