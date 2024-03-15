import os
import torch
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
print(torch.cuda.device_count())

import tensorflow as tf
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
from faiss.loader import *

import pandas as pd
import numpy as np
import networkx as nx
import sys
import math
import random
import dgl
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn
import torch.nn as nn

from conversational_sentence_encoder.vectorizers import SentenceEncoder

path = input("Dataset path: ")
sys.path.insert(1, '/cephfs/home/ledneva/lrec_coling/common_utils/')

from dgac_two_speakers import Clusters, get_accuracy_k, get_all_accuracy_k

import numpy as np
import faiss

def index_cpu_to_gpu_multiple_py(resources, index, co=None, gpus=None):
    if gpus is None:
        gpus = range(len(resources))
    vres = GpuResourcesVector()
    vdev = Int32Vector()
    for i, res in zip(gpus, resources):
        vdev.push_back(i)
        vres.push_back(res)
    if isinstance(index, IndexBinary):
        return index_binary_cpu_to_gpu_multiple(vres, vdev, index, co)
    else:
        return index_cpu_to_gpu_multiple(vres, vdev, index, co)


def index_cpu_to_all_gpus(index, co=None, ngpu=-1):
    index_gpu = index_cpu_to_gpus_list(index, co=co, gpus=None, ngpu=ngpu)
    return index_gpu


def index_cpu_to_gpus_list(index, co=None, gpus=None, ngpu=-1):
    if gpus is None and ngpu == -1:  # All blank
        gpus = range(get_num_gpus())
    elif gpus is None and ngpu != -1:  # Get number of GPU's only
        gpus = range(ngpu)
    res = [StandardGpuResources() for _ in gpus]
    index_gpu = index_cpu_to_gpu_multiple_py(res, index, co, gpus)
    return index_gpu


class FaissKNeighbors:
    def __init__(self, train_clusters):
        self.index = None
        self.train_clusters = train_clusters

    def fit(self, X):
        index = faiss.IndexFlatIP(X.shape[1])
        ngpu = 3
        self.index_gpu = index_cpu_to_gpus_list(index, gpus=[1,2,3])
        self.index_gpu.add(X.astype(np.float32))

    def predict(self, X, k):
        distances, indices = self.index_gpu.search(X.astype(np.float32), k = k)
        return self.train_clusters[indices], distances

num_clusters_list = [(200, 30), (400, 60), (800, 120)]
for (first_num_clusters, second_num_clusters) in num_clusters_list:
    num_iterations = 3

    sentence_encoder = SentenceEncoder(multiple_contexts=True)

    for iteration in range(num_iterations):
        clusters = Clusters(path, "en", first_num_clusters, second_num_clusters)
        clusters.form_clusters()

        from sentence_transformers import SentenceTransformer

        train_user_utterances = clusters.train_user_df['utterance'].tolist()
        train_system_utterances = clusters.train_system_df['utterance'].tolist()
        train_user_clusters = clusters.train_user_df['cluster'].to_numpy()
        train_system_clusters = clusters.train_system_df['cluster'].to_numpy()

        batches_train_user_utterances = np.array_split(train_user_utterances, 200)
        batches_train_system_utterances = np.array_split(train_system_utterances, 200)

        from tqdm import tqdm
        top_k = 10

        train_user_embeddings = np.concatenate([sentence_encoder.encode_responses(train_user_utterances)
                                                for train_user_utterances in tqdm(batches_train_user_utterances)])


        train_system_embeddings = np.concatenate([sentence_encoder.encode_responses(train_system_utterances)
                                                  for train_system_utterances in tqdm(batches_train_system_utterances)])
        
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
                utterances_histories = [""]

                if j > 0:
                    for k in range(max(0, j - top_k), j):
                        utterances_histories.append(obj["utterance"][k])

                context_encoding = (sentence_encoder.encode_multicontext(utterances_histories))[0]

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

#         for k in [1, 3, 5, 10]:
#             file.write(f"Acc@{k}: {np.mean(user_metric[k])}\n")