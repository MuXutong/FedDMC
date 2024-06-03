from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt

features = pd.read_csv(r'H:\By-FL\clustering\r65_grad_data\n5_r65.csv', header=None)
test_data = np.array(features)[:10]

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
cluster_id = np.array(range(n_samples, n_samples + n_samples - 1, 1))
linkage_matrix = np.column_stack([agglomer.children_, agglomer.distances_, counts, cluster_id]).astype(float)

root = linkage_matrix[-1:].reshape(-1)

while 1:
    left_tree_id = root[0]
    right_tree_id = root[1]

    if left_tree_id < n_samples:
        left_count = 1
    else:
        left_counts = linkage_matrix[left_tree_id - n_samples]


    if right_tree_id < n_samples:
        right_count = 1
    else:
        right_count = linkage_matrix[right_tree_id - n_samples][3]

    if left_count > 3 and right_count > 3:
        break
    elif left_count <=3:
        root = linkage_matrix[right_tree_id - n_samples]


dendrogram(linkage_matrix)

plt.show()
