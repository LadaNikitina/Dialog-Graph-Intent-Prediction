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
import torch
import math
import random
import dgl
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn
import torch.nn as nn
import time

import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,2,3"
print(torch.cuda.device_count())

path = input("Dataset path: ")
sys.path.insert(1, '/cephfs/home/ledneva/lrec_coling/common_utils/')

from dgac_one_speaker import Clusters, get_accuracy_k

num_clusters_list = [(200, 30), (400, 60), (800, 120)]

for (first_num_clusters, second_num_clusters) in num_clusters_list:
    def create_graph(num_nodes, train_data, test_data, train_dataset, test_dataset):
        probs = np.zeros((num_nodes + 1, num_nodes))

        index = 0

        for obj in train_dataset:
            pred_cluster = num_nodes

            for j in range(len(obj)):
                cur_cluster = train_data["cluster"][index]
                index += 1

                probs[pred_cluster][cur_cluster] += 1
                pred_cluster = cur_cluster

        for i in range(num_nodes + 1):
            sum_i_probs = sum(probs[i])

            if sum_i_probs != 0:
                probs[i] /= sum_i_probs

        test = []

        index = 0

        for obj in test_dataset:
            pred_cluster = num_nodes

            for j in range(len(obj)):
                cur_cluster = test_data["cluster"][index]
                test.append(probs[pred_cluster])
                index += 1
                pred_cluster = cur_cluster

#         get_accuracy_k(k, test_data, test, test_dataset)

    num_iters = 3

    for i in range(num_iters):
        start_time = time.time()
        clusters = Clusters(path, "en", first_num_clusters, second_num_clusters)
        clusters.form_clusters()

        create_graph(second_num_clusters, 
                     clusters.cluster_train_df, 
                     clusters.cluster_test_df, 
                     clusters.train_dataset,
                     clusters.test_dataset)