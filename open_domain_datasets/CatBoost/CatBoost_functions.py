import numpy as np
import pandas as pd
import networkx as nx
from sklearn import preprocessing
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import PCA
from tqdm import tqdm

def get_features(train_embs,
                 num_clusters):    
    '''
        getting features for each cluster
    '''
    array = train_embs
    null_array = [0 for i in range(len(array[0]))]

    # id of the cluster
    
    for i in range(num_clusters):
        array[i].append(i)
        
    null_array.append(num_clusters)
    
    return (array, null_array)


def get_data(features, null_features,
             clusters, dataset, top_k, 
             num_clusters, embs):    
    '''
        getting data and targets
    '''
    target = []
    embs_dim = len(embs[0])

    index = 0
    data = []
    num = 0

    for obj in tqdm(dataset):
        num += 1    
        
        pred_clusters = [(num_clusters, -1) for _ in range(top_k)]
        pred_embs = np.zeros((top_k, embs_dim)).tolist()

        for j in range(len(obj)):        
            cur_cluster = (clusters["cluster"][index], 0)

            data_obj = []
            target.append(cur_cluster[0])

            for k in range(top_k):
                cluster_k = pred_clusters[k]

                if cluster_k[1] == 0:
                    data_obj += features[cluster_k[0]]
                else:
                    data_obj += null_features

                data_obj += pred_embs[k]

            data.append(data_obj)
            cur_embs = embs[index].tolist()

            index += 1

            for k in range(1, top_k):
                pred_clusters[k - 1] = pred_clusters[k]
                pred_embs[k - 1] = pred_embs[k]
                
            pred_clusters[top_k - 1] = cur_cluster
            pred_embs[top_k - 1] = cur_embs
            
    data = np.array(data)
    data_target = pd.DataFrame({"target" : target})

    return (data, data_target)

