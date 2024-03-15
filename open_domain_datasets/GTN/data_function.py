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

def get_data(dataset, top_k, num_clusters, train, embs):
    data = []
    target = []
    data_embs = []
    
    embs_dim = len(embs[0])
    zero_cluster = num_clusters
    
    edges = []

    for k in range(top_k - 1):
        edges.append([k, k + 1])
    edge_index = torch.tensor(edges, dtype = torch.long)    
    
    index = 0
    
    for obj in dataset:
        utterance_clusters = [zero_cluster for i in range(top_k)]
        utterance_embs = list(np.zeros((top_k, embs_dim)))
    
        for j in range(len(obj)):
            utterance_clusters.append(train["cluster"][index])
            utterance_embs.append(embs[index])
            index += 1
        
        for j in range(top_k, len(utterance_clusters)):
            history = []
            history_embs = []
            
            for k in range(j - top_k, j):
                history.append(utterance_clusters[k])
                history_embs.append(utterance_embs[k])
                    
  
            data.append(history)
            data_embs.append(history_embs)
            target.append(utterance_clusters[j])
                      
            
    return np.array(data), np.array(target), \
           np.array(data_embs)