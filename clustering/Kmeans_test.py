from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

features = pd.read_csv(r'H:\By-FL\PCA_verify\Reducing_grad\n5_r5.csv', header=None)
features = np.array(features)

km = KMeans(n_clusters=2)
km.fit(features)

result = km.labels_

clients0 = []
clients1 = []

for id, value in enumerate(result):
    if value == 0:
        clients0.append('client{}'.format(id))
    else:
        clients1.append('client{}'.format(id))

if len(clients0) >= len(clients1):
    benign_client = clients0
    malicious_clients = clients1
else:
    benign_client = clients1
    malicious_clients = clients0


print()
