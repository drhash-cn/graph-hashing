from torch_geometric.datasets import TUDataset
from utils import embbeding
import time
import torch
import numpy as np
from scipy.spatial.distance import pdist,squareform
from svmutil import *
import sklearn
from sklearn import svm
import random
from sklearn.model_selection import cross_val_score
np.set_printoptions(threshold=np.inf)
random.seed(22)
dataset=['SF-295']
ds_name='SF-295'
datasets = TUDataset(root=f'/tmp/{ds_name}', name=f'{ds_name}')
#dim=[0,1,2]
ds_size=[100,1000,2000,4000,6000,8000,10000,20000,30000,40271]
ds_size=[10000,20000,30000,40000]
per_dim=[25,50,10]
B = np.arange(len(datasets)).tolist()
random.seed(222)
Time=list()
Acc=list()
classes = []


# for d in dim:
#     #graph_id=np.random.randint(0,len(datasets),i)
#     start_time=time.time()
#     for data in datasets:
#         result_square = np.zeros((1, len(data.x), 20 * (sum(per_dim[:d+1]))))
#         #data=datasets[graph_id[g]]
#         result_square=embbeding(1,ds_name,data.edge_index,data.x,d,20,1,per_dim[:d+1],20*(sum(per_dim[:d+1])),result_square)
#         classes.append(int(data.y))
#     print("Cost time  ",time.time()-start_time)
#     gram_matrix=1-squareform(pdist(result_square[0], 'hamming'))
#     predictor = svm.SVC(C=2, kernel='precomputed')
#     scores = np.mean(cross_val_score(predictor, gram_matrix, classes, scoring='accuracy'))
#     print("Score: {}".format(scores))
#     classes.clear()
# for data in datasets:
#     classes.append(int(data.y))
# print(classes)
for ds_name in dataset:
    for i in ds_size:
        graph_id =random.sample(B, i)
        if len(set(graph_id)) == len(graph_id):
            print(False)
        else:
            print(True)
        start_time=time.time()
        for g in range(i):
            result_square = np.zeros((1, i, 20 * (sum(per_dim[:2]))))
            data=datasets[graph_id[g]]
            flag=embbeding(g,ds_name,data.edge_index,data.x,1,20,1,per_dim,20*(sum(per_dim[:2])),result_square)
            classes.append(int(data.y))


        gram_matrix=1-squareform(pdist(result_square[0], 'hamming'))
        predictor = svm.SVC(C=1, kernel='precomputed')
        scores = np.mean(cross_val_score(predictor, gram_matrix, classes, scoring='accuracy'))
        # print("Score: {}".format(scores))
        # print("Cost time ", time.time() - start_time)
        Acc.append(scores*100)
        Time.append(time.time() - start_time)
        classes.clear()
    print(ds_name,Time,Acc)