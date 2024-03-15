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
import random
import dgl
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn
import torch.nn as nn

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
print(torch.cuda.device_count())

sys.path.insert(1, '/cephfs/home/ledneva/lrec_coling/common_utils/')

from dgac_two_speakers import Clusters, get_accuracy_k, get_all_accuracy_k

import numpy as np
import faiss

path = input("Dataset path: ")


class FaissKNeighbors:
    def __init__(self, train_clusters):
        self.index = None
        self.train_clusters = train_clusters

    def fit(self, X):
        index = faiss.IndexFlatIP(X.shape[1])
        ngpu = 1
        resources = [faiss.StandardGpuResources() for i in range(ngpu)]
        self.index_gpu = faiss.index_cpu_to_gpu_multiple_py(resources, index)
        self.index_gpu.add(X.astype(np.float32))

    def predict(self, X, k):
        distances, indices = self.index_gpu.search(X.astype(np.float32), k = k)
        return self.train_clusters[indices], distances

num_clusters_list = [(200, 30), (400, 60), (800, 120)]
for (first_num_clusters, second_num_clusters) in num_clusters_list:
    num_iterations = 3

    for iteration in range(num_iterations):
        clusters = Clusters(path, "en", first_num_clusters, second_num_clusters)
        clusters.form_clusters()

        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
        model = model.to('cuda')

        train_user_utterances = clusters.train_user_df['utterance'].tolist()
        train_system_utterances = clusters.train_system_df['utterance'].tolist()
        train_user_clusters = clusters.train_user_df['cluster'].to_numpy()
        train_system_clusters = clusters.train_system_df['cluster'].to_numpy()

        from tqdm import tqdm

        train_user_embeddings = clusters.train_user_embs
        train_system_embeddings = clusters.train_system_embs
            
        faiss_user_index = FaissKNeighbors(train_user_clusters)
        faiss_user_index.fit(train_user_embeddings)
        
        faiss_system_index = FaissKNeighbors(train_system_clusters)
        faiss_system_index.fit(train_system_embeddings)    
        
        user_metric = {1 : [], 3 : [], 5 : [], 10 : []}
        system_metric = {1 : [], 3 : [], 5 : [], 10 : []}
        num = 0
        all_num = 0
        ind_user = 0
        ind_system = 0

        for obj in tqdm(clusters.test_dataset):
            user_utterance_metric = {1 : [], 3 : [], 5 : [], 10 : []}
            system_utterance_metric = {1 : [], 3 : [], 5 : [], 10 : []}

            for j in range(len(obj["utterance"])):
                all_num += 1
                utterance_history = " "

                if j > 0:
                    utterance_history = obj["utterance"][j - 1]


                context_encoding = model.encode(utterance_history)

                if obj['speaker'][j] == 0:
                    cur_cluster = clusters.test_user_df["cluster"][ind_user]
                    answer_clusters, distances = faiss_user_index.predict(np.array([context_encoding]), 1000)

                    result_clusters = []

                    for cluster in answer_clusters[0]:
                        if cluster not in result_clusters:
                            result_clusters.append(cluster)
                        if len(result_clusters) == 10:
                            break

                    for k in [1, 3, 5, 10]:
                        if cur_cluster in result_clusters[:k]:
                            user_utterance_metric[k].append(1) 
                        else:
                            user_utterance_metric[k].append(0) 
                    ind_user += 1
                else:
                    cur_cluster = clusters.test_system_df["cluster"][ind_system]
                    answer_clusters, distances = faiss_system_index.predict(np.array([context_encoding]), 1000)

                    result_clusters = []

                    for cluster in answer_clusters[0]:
                        if cluster not in result_clusters:
                            result_clusters.append(cluster)
                        if len(result_clusters) == 10:
                            break

                    for k in [1, 3, 5, 10]:
                        if cur_cluster in result_clusters[:k]:
                            system_utterance_metric[k].append(1) 
                        else:
                            system_utterance_metric[k].append(0) 
                    ind_system += 1

            for k in [1, 3, 5, 10]:
                if len(user_utterance_metric[k]) > 0:
                    user_metric[k].append(np.mean(user_utterance_metric[k])) 
                if len(system_utterance_metric[k]) > 0:                    
                    system_metric[k].append(np.mean(system_utterance_metric[k])) 