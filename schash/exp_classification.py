from torch_geometric.datasets import TUDataset
from utils import embbeding
import time
import numpy as np
from scipy.spatial.distance import pdist,squareform
import sklearn
from sklearn import svm
from sklearn.model_selection import cross_val_score
np.set_printoptions(threshold=np.inf)
dataset=['MUTAG','BZR','PROTEINS','AIDS']
per_dim= [25,50,10]
for ds_name in dataset:
    datasets = TUDataset(root=f'/tmp/DATA', name=f'{ds_name}')
    M=25
    if ds_name=='PROTEINS':
        T=3
        M=20
    elif ds_name=='AIDS':
        T=4
    else:
        T=1
    if ds_name == 'PROTEINS':
        max_dim=2
    else:
        max_dim=1

    start_time = time.time()
    classes=[]
    result_square = np.zeros((T, len(datasets), M*(sum(per_dim[:max_dim+1]))))
    for g in range(len(datasets)):
        data=datasets[g]
        flag = embbeding(g, ds_name, data.edge_index, data.x, max_dim, M, T, per_dim,
                                  M* (sum(per_dim[:max_dim + 1])), result_square)
        classes.append(int(data.y))
    result_square=result_square[:,:len(classes),:]
    gram_matrix=np.zeros((len(classes),len(classes)))
    for r in range(T):
        gram_matrix=gram_matrix+(1-squareform(pdist(result_square[r], 'hamming')))
    predictor = svm.SVC(C=1,kernel='precomputed')
    scores = cross_val_score(predictor, gram_matrix, classes, cv=10,scoring='accuracy')
    acc_mean=np.mean(scores)
    acc_std=np.std(scores)
    print("Score: {} ± {}".format(acc_mean,acc_std))
    end_time = time.time()
    print("Cost Time",(end_time-start_time))
    msg = (
        f'========== Result ============\n'
        f'Dataset:        {dataset}\n'
        f'Accuracy:       {scores}\n'
        f'Cost_time:      {end_time-start_time}\n'
        '-------------------------------\n\n')
    # file = open(f'./results/{ds_name}/{ds_name}_M{M[i]}_T{str(T[j])}.txt', 'w')
    # file.write(msg)





