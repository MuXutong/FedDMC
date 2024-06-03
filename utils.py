import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
import math


def Adding_Trigger(data):
    if data.shape[0] == 3:
        for i in range(3):
            data[i][1][28] = 1
            data[i][1][29] = 1
            data[i][1][30] = 1
            data[i][2][29] = 1
            data[i][3][28] = 1
            data[i][4][29] = 1
            data[i][5][28] = 1
            data[i][5][29] = 1
            data[i][5][30] = 1

    if data.shape[0] == 1:
        data[0][1][24] = 1
        data[0][1][25] = 1
        data[0][1][26] = 1
        data[0][2][24] = 1
        data[0][3][25] = 1
        data[0][4][26] = 1
        data[0][5][24] = 1
        data[0][5][25] = 1
        data[0][5][26] = 1
    return data


def model_dist_norm(model, target_params):
    squared_sum = 0
    for name, layer in model.named_parameters():
        squared_sum += torch.sum(torch.pow(layer.data - target_params[name].data, 2))
    return math.sqrt(squared_sum)


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


def Building_tree(linkage_matrix, n_samples):
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
    return root


def Removing_outliers(root, min_cluster_size=3):
    outlier_all = []
    n_clients = root.counts

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
        if len(outlier_all) > (n_clients // 2):
            break

    # root中，左右孩子分别是良性和恶意用户
    benign, malicious = [], []
    if root:
        if root.lchild.counts > (n_clients // 2) or root.rchild.counts > (n_clients // 2):
            if root.rchild.counts < root.lchild.counts:
                malicious = root.rchild.postorder_travel(root.rchild)
                benign = root.lchild.postorder_travel(root.lchild)
            elif root.rchild.counts > root.lchild.counts:
                benign = root.rchild.postorder_travel(root.rchild)
                malicious = root.lchild.postorder_travel(root.lchild)
        else:
            benign = root.postorder_travel(root)
    return benign, malicious, outlier_all


def get_linkage_matrix(agglomer):
    counts = np.zeros(agglomer.children_.shape[0])
    n_samples = len(agglomer.labels_)

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
    linkage_matrix = np.column_stack([agglomer.children_, d, counts]).astype(float)

    return linkage_matrix


def lbfgs(S_k_list, Y_k_list, v):
    curr_S_k = torch.cat(S_k_list, dim=1)
    curr_Y_k = torch.cat(Y_k_list, dim=1)
    S_k_time_Y_k = torch.mm(curr_S_k.T, curr_Y_k)
    S_k_time_S_k = torch.mm(curr_S_k.T, curr_S_k)
    R_k = torch.triu(S_k_time_Y_k)
    L_k = S_k_time_Y_k - R_k
    sigma_k = torch.mm(Y_k_list[-1].T, S_k_list[-1]) / (torch.mm(S_k_list[-1].T, S_k_list[-1]))
    D_k_diag = torch.diag(S_k_time_Y_k)
    upper_mat = torch.cat([sigma_k * S_k_time_S_k, L_k], dim=1)
    lower_mat = torch.cat([L_k.T, -torch.diag(D_k_diag)], dim=1)
    mat = torch.cat([upper_mat, lower_mat], dim=0)
    mat_inv = torch.inverse(mat)

    approx_prod = sigma_k * v
    p_mat = torch.cat([torch.mm(curr_S_k.T, sigma_k * v), torch.mm(curr_Y_k.T, v)], dim=0)
    approx_prod -= torch.mm(torch.mm(torch.cat([sigma_k * curr_S_k, curr_Y_k], dim=1), mat_inv), p_mat)

    return approx_prod


def Gap_Statistics(score, B, K, num_in_comm):
    nrefs = torch.tensor(B)
    ks = range(1, K)
    gaps = torch.zeros(len(ks))
    gapDiff = torch.zeros(len(ks) - 1)
    sdk = torch.zeros(len(ks))
    # print(score)
    min = torch.min(score)
    max = torch.max(score)
    score = (score - min) / (max - min)
    for i, k in enumerate(ks):
        estimator = KMeans(n_clusters=k)
        score1 = score.reshape(-1, 1)
        estimator.fit(score1)
        label_pred = torch.tensor(estimator.labels_)
        center = torch.tensor(estimator.cluster_centers_)
        # mm = [torch.pow(score[m] - center[label_pred[m]], 2) for m in range(len(score))]
        # mmm = torch.tensor([torch.pow(score[m] - center[label_pred[m]], 2) for m in range(len(score))])
        Wk = torch.sum(torch.tensor([torch.pow(score[m] - center[label_pred[m]], 2) for m in range(len(score))]))
        WkRef = torch.zeros(B)
        for j in range(nrefs):
            rand = torch.tensor(np.random.uniform(0, 1, len(score)))
            estimator = KMeans(n_clusters=k)
            estimator.fit(rand.reshape(-1, 1))
            label_pred = torch.tensor(estimator.labels_)
            center = torch.tensor(estimator.cluster_centers_)
            WkRef[j] = torch.sum(
                torch.tensor([torch.pow(rand[m] - center[label_pred[m]], 2) for m in range(len(rand))]))
        gaps[i] = torch.log(torch.mean(WkRef)) - torch.log(Wk)
        sdk[i] = torch.sqrt((1.0 + nrefs) / nrefs) * torch.std(torch.log(WkRef))

        if i > 0:
            gapDiff[i - 1] = gaps[i - 1] - gaps[i] + sdk[i]
    # print(gapDiff)
    select_k = 0
    for i in range(len(gapDiff)):
        if gapDiff[i] >= 0:
            select_k = i + 1
            break
    if select_k == 1:
        print('No attack detected!')
        return 0
    else:
        print('Attack Detected!')
        return 1


def grad_simple_mean(old_client_grad_list, client_grad_list, b=0, hvp=None):
    if hvp is not None:
        pred_grad = []
        distance = []
        for i in range(len(old_client_grad_list)):
            pred_grad.append(old_client_grad_list[i] + hvp)
            # distance.append((1 - nd.dot(pred_grad[i].T, client_grad_list[i]) / (
            # nd.norm(pred_grad[i]) * nd.norm(client_grad_list[i]))).asnumpy().item())

        pred = torch.zeros(5)
        pred[:b] = 1
        distance = torch.norm((torch.cat(old_client_grad_list, dim=1) - torch.cat(client_grad_list, dim=1)), dim=0)
        auc1 = roc_auc_score(pred, distance)
        distance = torch.norm((torch.cat(pred_grad, dim=1) - torch.cat(client_grad_list, dim=1)), dim=0)
        auc2 = roc_auc_score(pred, distance)
        print("Detection AUC: %0.4f; Detection AUC: %0.4f" % (auc1, auc2))

        # distance = nd.norm((nd.concat(*old_gradients, dim=1) - nd.concat(*client_grad_list, dim=1)), axis=0).asnumpy()
        # distance = nd.norm(nd.concat(*client_grad_list, dim=1), axis=0).asnumpy()
        # normalize distance
        distance = distance / torch.sum(distance)
    else:
        distance = None

    mean_nd = torch.mean(torch.cat(client_grad_list, dim=1), dim=-1, keepdim=True)

    return mean_nd, distance


def computer_defense_acc(detect_malicious_client, malicious_clients, clients_in_comm):
    d_m_c = []
    m_c = []

    for client in clients_in_comm:
        d_m_c.append(1) if client in detect_malicious_client else d_m_c.append(0)
        m_c.append(1) if client in malicious_clients else m_c.append(0)

    count = 0
    for key, value in enumerate(d_m_c):
        if m_c[key] == value:
            count += 1

    defense_acc = count / len(clients_in_comm)

    count1 = 0
    for client in malicious_clients:
        if client in detect_malicious_client:
            count1 += 1

    if len(malicious_clients) != 0:
        malicious_precision = count1 / len(malicious_clients)
    else:
        malicious_precision = 1

    count2 = 0
    for client in detect_malicious_client:
        if client in malicious_clients:
            count2 += 1

    if len(detect_malicious_client) != 0:
        malicious_recall = count1 / len(detect_malicious_client)
    else:
        malicious_recall = 0

    return defense_acc, malicious_precision, malicious_recall


def param_to_list(local_parameters):
    parameters = []
    for key, var in local_parameters.items():
        if key == 'fc1.weight' or key == 'fc1.bias':
            temp = var.clone().cpu().tolist()
            if len(np.array(temp).shape) == 2:
                temp = sum(temp, [])
            parameters.append(temp)
    parameters = sum(parameters, [])
    return parameters


def cosine_clients(param_matrix):
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    param_tf = torch.FloatTensor(param_matrix).to(dev)
    cos_mat_torch = list(
        map(lambda x: list(map(lambda y: F.cosine_similarity(x, y, dim=0).item(), param_tf)), param_tf))
    # cos_mat = param_tf.mm(param_tf.t())
    return cos_mat_torch


def euclidean_clients(param_matrix):
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    param_tf = torch.FloatTensor(param_matrix).to(dev)
    output = torch.cdist(param_tf, param_tf, p=2)

    return output.tolist()


def TSNE_clients(param_matrix, dimension):
    tsne = TSNE(n_components=dimension, init='pca', random_state=0)
    param_tsne = tsne.fit_transform(param_matrix)

    return param_tsne


# def param_todict(local_parameters):
#     parameters = {}
#     for key, var in local_parameters.items():
#         parameters[key] = var.clone().cpu().tolist()
#     return parameters

def test_ASR(net, parameters, testDataLoader):
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    criterion = F.cross_entropy
    with torch.no_grad():
        net.load_state_dict(parameters, strict=True)

        sum_ASR = 0
        count = 0
        backdoor_label = 0

        # 载入测试集
        for data, label in testDataLoader:
            data, label = data.to(dev), label.to(dev)
            # backdoor_label = torch.zeros(label.shape).to(dev)
            for example_id in range(data.shape[0]):
                data[example_id] = Adding_Trigger(data[example_id])
                # backdoor_label[example_id] = 0

            preds = net(data)

            preds = torch.argmax(preds, dim=1)

            for i, v in enumerate(preds):
                if v != label[i] and v == 0:
                    count += 1
            # sum_accu += (preds == label).float().mean()
            sum_ASR += data.shape[0]

            # sum_ASR += (preds == backdoor_label).float().mean()

        asr = count / sum_ASR

    return asr


def test_accuracy(net, parameters, testDataLoader):
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    loss_collector = []
    criterion = F.cross_entropy
    with torch.no_grad():
        net.load_state_dict(parameters, strict=True)
        sum_accu = 0
        num = 0
        loss_collector.clear()
        # 载入测试集
        for data, label in testDataLoader:
            data, label = data.to(dev), label.to(dev)
            preds = net(data)
            loss = criterion(preds, label.long())
            # loss = 1
            loss_collector.append(loss.item())
            preds = torch.argmax(preds, dim=1)
            sum_accu += (preds == label).float().mean()
            num += 1

        accuracy = sum_accu / num
        avg_loss = sum(loss_collector) / len(loss_collector)
    return avg_loss, accuracy


def launch_tensor_board(log_path, port, host):
    """Function for initiating TensorBoard.

    Args:
        log_path: Path where the log is stored.
        port: Port number used for launching TensorBoard.
        host: Address used for launching TensorBoard.
    """
    os.system(f"tensorboard.exe --logdir={log_path} --port={port} --host={host}")
    return True


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass


if __name__ == '__main__':
    s = torch.randn(5, 1)
    Gap_Statistics(s, 2, 3, 5)

    # det_malicious_clients = ['client12', 'client14', 'client64']
    #
    # malicious_clients = ['client12', 'client14', 'client64', 'client40', 'client89', 'client58', 'client60', 'client35',
    #                      'client18', 'client52', 'client69', 'client46', 'client15', 'client48', 'client71', 'client62',
    #                      'client42', 'client70', 'client31', 'client83']
    #
    # acc = computer_defense_acc(det_malicious_clients, malicious_clients)
    # print(acc)
