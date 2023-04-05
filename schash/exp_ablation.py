import random
from torch_geometric.datasets import TUDataset
from utils import embbeding
import time
import torch
import numpy as np
from scipy.spatial.distance import pdist,squareform
import sklearn
from sklearn import svm
from sklearn.model_selection import cross_val_score
dataset=['MUTAG','PROTEINS']
t=1
per_dim=[25,50,10]
Max_dim=[0,1,2]
def test_dim():
    Max_dim=[0,1,2]
    for ds_name in dataset:
        for max_dim in Max_dim:
            datasets = TUDataset(root=f'/tmp/{ds_name}', name=f'{ds_name}')
            start_time = time.time()
            classes = []
            m=25
            t=1
            result_square = np.zeros((t, len(datasets),m * (sum(per_dim[:max_dim + 1]))))
            for g in range(len(datasets)):
                np.random.seed(222)
                data = datasets[g]
                F = embbeding(g, ds_name, data.edge_index, data.x, max_dim, m, t, per_dim,
                                          m * (sum(per_dim[:max_dim + 1])), result_square)
                classes.append(int(data.y))
            gram_matrix = np.zeros((len(datasets), len(datasets)))
            # result_square=result_square.tolist()
            for r in range(t):
                gram_matrix = gram_matrix + (1 - squareform(pdist(result_square[r], 'hamming')))
            predictor = svm.SVC(C=1, kernel='precomputed')
            scores = cross_val_score(predictor, gram_matrix, classes, cv=10, scoring='accuracy')
            acc_mean = np.mean(scores)
            acc_std = np.std(scores)
            print(ds_name, m)
            print("Score: {} ± {}\n".format(acc_mean, acc_std))
            end_time = time.time()
            cost_time=end_time-start_time
            print("cost_time: {}\n".format(cost_time))
test_dim()

