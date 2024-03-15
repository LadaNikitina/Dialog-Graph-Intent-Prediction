from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
from catboost import CatBoostClassifier
from torch import nn
from torch_geometric.data import Data
from torch_scatter import scatter_add
from torch_geometric.nn import MessagePassing
from torch.nn import Linear
from torch_geometric.nn import GCNConv

import pandas as pd
import numpy as np
import networkx as nx
import os
import torch
import math
import random

def get_data(dataset, top_k, num_clusters, train_user, train_system, user_embs, sys_embs):
    data_user = []
    target_user = []
    data_sys = []
    target_sys = []
    zero_cluster = 2 * num_clusters
    
    edges = []

    for k in range(top_k - 1):
        edges.append([k, k + 1])
    edge_index = torch.tensor(edges, dtype = torch.long)    
    
    ind_user = 0
    ind_system = 0
    embs_dim = user_embs.shape[1]
    null_cluster_emb = np.zeros(embs_dim)
    
    for obj in dataset:
        utterance_clusters = [zero_cluster for i in range(top_k)]
        utterance_embs = [null_cluster_emb for i in range(top_k)]
   
        for j in range(len(obj["utterance"])):
            if obj['speaker'][j] == 0:
                utterance_clusters.append(train_user["cluster"][ind_user])
                utterance_embs.append(user_embs[ind_user])
                ind_user += 1
            else:
                utterance_clusters.append(train_system["cluster"][ind_system] + num_clusters)
                utterance_embs.append(sys_embs[ind_system])
                ind_system += 1
        
        for j in range(top_k, len(utterance_clusters)):
            history = []
            uttr_history = []
            
            for k in range(j - top_k, j):
                history.append(utterance_clusters[k])
                uttr_history.append(utterance_embs[k])
                
            if utterance_clusters[j] < num_clusters:
                data_user.append((history, np.array(uttr_history)))
                target_user.append(utterance_clusters[j] % num_clusters)
            else:
                data_sys.append((history, np.array(uttr_history)))
                target_sys.append(utterance_clusters[j] % num_clusters)
                      
            
    return data_user, np.array(target_user), data_sys, np.array(target_sys)