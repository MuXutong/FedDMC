import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets as data
from hdbscan import HDBSCAN
# matplotlib inline


sns.set_context('poster')
sns.set_style('white')
sns.set_color_codes()
plot_kwds = {'alpha': 0.5, 's': 80, 'linewidths': 0}

# moons, _ = data.make_moons(n_samples=50, noise=0.05)
# blobs, _ = data.make_blobs(n_samples=50, centers=[(-0.75, 2.25), (1.0, 2.0)], cluster_std=0.25)
# test_data = np.vstack([moons, blobs])
# fig = plt.figure(figsize=(12, 9))
# plt.scatter(test_data.T[0], test_data.T[1], color='b', **plot_kwds)
# plt.show()

features = pd.read_csv(r'H:\By-FL\clustering\r65_grad_data\n5_r65.csv', header=None)
test_data = np.array(features)[:10]

clusterer = HDBSCAN(min_cluster_size=3, gen_min_span_tree=True, cluster_selection_method='leaf')
clusterer.fit(test_data)
result = clusterer.labels_

# clusterer.minimum_spanning_tree_.plot(edge_cmap='viridis',
#                                       edge_alpha=0.6,
#                                       node_size=80,
#                                       edge_linewidth=2)
condensed_tree = clusterer.condensed_tree_



b = clusterer._condensed_tree
c = clusterer.cluster_selection_method
d = clusterer.allow_single_cluster


plot_data = condensed_tree.get_plot_data()
plt.show()