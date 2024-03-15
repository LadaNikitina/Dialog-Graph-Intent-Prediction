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

def get_data(dataset, top_k, num_clusters, clusters_info, embs):
    data = []
    target = []
    zero_cluster = num_clusters
    
    edges = []

    for k in range(top_k - 1):
        edges.append([k, k + 1])
    edge_index = torch.tensor(edges, dtype = torch.long)    
    
    index = 0
    embs_dim = embs.shape[1]
    null_cluster_emb = np.zeros(embs_dim)
    
    for obj in dataset:
        utterance_clusters = [zero_cluster for i in range(top_k)]
        utterance_embs = [null_cluster_emb for i in range(top_k)]
   
        for j in range(len(obj)):
            utterance_clusters.append(clusters_info["cluster"][index])
            utterance_embs.append(embs[index])
            index += 1
        
        for j in range(top_k, len(utterance_clusters)):
            history = []
            uttr_history = []
            
            for k in range(j - top_k, j):
                history.append(utterance_clusters[k])
                uttr_history.append(utterance_embs[k])
                
            data.append((history, np.array(uttr_history)))
            target.append(utterance_clusters[j])
         
    return data, target