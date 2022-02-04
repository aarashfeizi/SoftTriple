import copy

import torch
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def evaluation(X, Y, Kset):
    num = X.shape[0]
    classN = np.max(Y) + 1
    kmax = np.max(Kset)
    recallK = np.zeros(len(Kset))
    # compute NMI
    kmeans = KMeans(n_clusters=classN).fit(X)
    nmi = normalized_mutual_info_score(Y, kmeans.labels_, average_method='arithmetic')
    # compute Recall@K
    sim = X.dot(X.T)
    minval = np.min(sim) - 1.
    sim -= np.diag(np.diag(sim))
    sim += np.diag(np.ones(num) * minval)
    indices = np.argsort(-sim, axis=1)[:, : kmax]
    YNN = Y[indices]
    for i in range(0, len(Kset)):
        pos = 0.
        for j in range(0, num):
            if Y[j] in YNN[j, :Kset[i]]:
                pos += 1.
        recallK[i] = pos / num
    auc = calc_auroc(X, Y)
    return auc, nmi, recallK


def make_batch_bce_labels(labels):
    """
    :param labels: e.g. tensor of size (N,1)
    :return: binary matrix of labels of size (N, N)
    """

    l_ = labels.repeat(len(labels)).reshape(-1, len(labels))
    l__ = labels.repeat_interleave(len(labels)).reshape(-1, len(labels))

    final_bce_labels = (l_ == l__).type(torch.float32)

    # final_bce_labels.fill_diagonal_(0)

    return final_bce_labels
def get_samples(l, k):
    if len(l) < k:
        to_ret = np.random.choice(l, k, replace=True)
    else:
        to_ret = np.random.choice(l, k, replace=False)

    return to_ret
def get_xs_ys(bce_labels, k=1):
    """

    :param bce_labels: tensor of (N, N) with 0s and 1s
    :param k: number of pos and neg samples per anch
    :return:

    """
    xs = []
    ys = []
    bce_labels_copy = copy.deepcopy(bce_labels)
    bce_labels_copy.fill_diagonal_(-1)
    for i, row in enumerate(bce_labels_copy):
        neg_idx = torch.where(row == 0)[0]
        pos_idx = torch.where(row == 1)[0]

        ys.extend(get_samples(neg_idx, k))
        ys.extend(get_samples(pos_idx, k))
        xs.extend(get_samples([i], 2 * k))

    return xs, ys

def calc_auroc(embeddings, labels):
    from sklearn.metrics import roc_auc_score

    if type(labels) != torch.Tensor:
        labels = torch.tensor(labels)

    bce_labels = make_batch_bce_labels(labels)
    print('Calculating cosine sims')
    similarities = cosine_similarity(embeddings)
    print('Done calculating cosine sims')

    xs, ys = get_xs_ys(bce_labels)

    true_labels = bce_labels[xs, ys]
    predicted_labels = similarities[xs, ys]

    return roc_auc_score(true_labels, predicted_labels)

