from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
from torch import nn
from torch_geometric.data import Data
from torch_scatter import scatter_add
from torch_geometric.nn import MessagePassing
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from dgl.dataloading import GraphDataLoader
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import networkx as nx
import sys
import os
import torch
import math
# import tensorflow as tf
import random
import dgl
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn
import torch.nn as nn

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
print(torch.cuda.device_count())

path = input("Dataset path: ")
sys.path.insert(1, '/cephfs/home/ledneva/lrec_coling/common_utils/')

from dgac_two_speakers import Clusters, get_accuracy_k, get_all_accuracy_k

def create_graph(num_nodes, train_user_data, train_system_data, 
              test_user_data, test_system_data, 
              train_dataset, test_dataset):
    w_user_sys = np.zeros((num_nodes + 1, num_nodes))
    w_sys_user = np.zeros((num_nodes + 1, num_nodes))

    ind_user = 0
    ind_system = 0

    for obj in train_dataset:
        pred_cluster = num_nodes

        for j in range(len(obj["utterance"])):
            if obj['speaker'][j] == 0:
                cur_cluster = train_user_data["cluster"][ind_user]

                ind_user += 1

                w_sys_user[pred_cluster][cur_cluster] += 1
                pred_cluster = cur_cluster
            else:
                cur_cluster = train_system_data["cluster"][ind_system]

                ind_system += 1

                w_user_sys[pred_cluster][cur_cluster] += 1
                pred_cluster = cur_cluster

    for i in range(num_nodes + 1):
        sum_i_user_sys = sum(w_user_sys[i])
        sum_i_sys_user = sum(w_sys_user[i])

        if sum_i_user_sys != 0:
            w_user_sys[i] /= sum_i_user_sys

        if sum_i_sys_user != 0:    
            w_sys_user[i] /= sum_i_sys_user 

    sys_test = []
    user_test = []

    ind_user = 0
    ind_system = 0

    for obj in test_dataset:
        pred_cluster = num_nodes

        for j in range(len(obj["utterance"])):
            if obj['speaker'][j] == 0:
                cur_cluster = test_user_data["cluster"][ind_user]
                user_test.append(w_sys_user[pred_cluster])
                ind_user += 1
                pred_cluster = cur_cluster
            else:
                cur_cluster = test_system_data["cluster"][ind_system]
                sys_test.append(w_user_sys[pred_cluster])
                ind_system += 1
                pred_cluster = cur_cluster
                
#     get_accuracy_k(k, clusters.test_user_df, user_test, clusters.test_dataset, 0)
#     get_accuracy_k(k, clusters.test_user_df, user_test, clusters.test_dataset, 1)
#     get_all_accuracy_k(k, clusters.test_user_df, clusters.test_system_df, user_test, sys_test, clusters.test_dataset)

num_clusters_list = [(200, 30), (400, 60), (800, 120)]
for (first_num_clusters, second_num_clusters) in num_clusters_list:
    num_iters = 3
    
    for i in range(num_iters):
        clusters = Clusters(path, "en", first_num_clusters, second_num_clusters)
        clusters.form_clusters()

        create_graph(
            second_num_clusters, 
            clusters.train_user_df, 
            clusters.train_system_df, 
            clusters.test_user_df, 
            clusters.test_system_df,
            clusters.train_dataset,
            clusters.test_dataset
        )
        del clusters