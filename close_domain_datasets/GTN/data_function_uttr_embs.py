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
# import tensorflow as tf
import random

def get_data(dataset, top_k, num_clusters, train_user, train_system, user_embs, sys_embs):
    data_user = []
    target_user = []
    data_sys = []
    target_sys = []
    data_user_embs = []
    data_sys_embs = []
    
    embs_dim = len(user_embs[0])
    zero_cluster = 2 * num_clusters
    
    edges = []

    for k in range(top_k - 1):
        edges.append([k, k + 1])
    edge_index = torch.tensor(edges, dtype = torch.long)    
    
    ind_user = 0
    ind_system = 0
    
    for obj in dataset:
        utterance_clusters = [zero_cluster for i in range(top_k)]
        utterance_embs = list(np.zeros((top_k, embs_dim)))
    
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
            history_embs = []
            
            for k in range(j - top_k, j):
                history.append(utterance_clusters[k])
                history_embs.append(utterance_embs[k])
                    
            if utterance_clusters[j] < num_clusters:
                data_user.append(history)
                data_user_embs.append(history_embs)
                target_user.append(utterance_clusters[j] % num_clusters)
            else:
                data_sys.append(history)
                data_sys_embs.append(history_embs)
                target_sys.append(utterance_clusters[j] % num_clusters)
                      
            
    return np.array(data_user), np.array(target_user), \
           np.array(data_user_embs), np.array(data_sys), \
           np.array(target_sys), np.array(data_sys_embs)