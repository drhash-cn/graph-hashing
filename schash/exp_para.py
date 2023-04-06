from torch_geometric.datasets import TUDataset
from utils import embbeding
import time
import torch
import numpy as np
from scipy.spatial.distance import pdist,squareform
import sklearn
from sklearn import svm
import random
from sklearn.model_selection import cross_val_score
np.set_printoptions(threshold=np.inf)
dataset=['PROTEINS']
iterations=[1,2,3,4,5]
per_dim=[25,50,10]
Time=list()
Acc=list()
def run():
    for ds_name in dataset:
        datasets = TUDataset(root=f'/tmp/DATA', name=f'{ds_name}')
        M=[15,20,25,30]
        T=[1,2,3,4,5]
        max_dim=2
        for j in range(len(T)):
            Acc = list()
            Time = list()
            for i in range(len(M)):
                start_time = time.time()
                classes=[]
                result_square = np.zeros((T[j], len(datasets), M[i]*(sum(per_dim[:max_dim+1]))))
                for g in range(len(datasets)):
                    np.random.seed(222)
                    data=datasets[g]
                    f=embbeding(g,ds_name,data.edge_index,data.x, max_dim,M[i],T[j],per_dim,M[i]*(sum(per_dim[:max_dim+1])),result_square)
                    classes.append(int(data.y))
                gram_matrix=np.zeros((len(datasets),len(datasets)))
                #result_square=result_square.tolist()
                for r in range(T[j]):
                    gram_matrix=gram_matrix+(1-squareform(pdist(result_square[r], 'hamming')))
                predictor = svm.SVC(C=1, kernel='precomputed')
                scores = cross_val_score(predictor, gram_matrix, classes, cv=10,scoring='accuracy')
                acc_mean=np.mean(scores)
                acc_std=np.std(scores,ddof=1)
                print("Score: {} ± {}".format(acc_mean,acc_std))
                end_time = time.time()
                print("Cost Time",(end_time-start_time))
                Acc.append(np.mean(scores))
                Time.append(end_time-start_time)

            print(Time)
            print(Acc)

run()