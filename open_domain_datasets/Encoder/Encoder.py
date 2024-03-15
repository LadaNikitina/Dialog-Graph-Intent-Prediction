import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
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

print(torch.cuda.device_count())

path = input("Dataset path: ")
sys.path.insert(1, '/cephfs/home/ledneva/lrec_coling/common_utils/')

from dgac_one_speaker import Clusters, get_accuracy_k

num_clusters_list = [(200, 30), (400, 60), (800, 120)]
for (first_num_clusters, second_num_clusters) in num_clusters_list:
    num_iterations = 3

    for iteration in range(num_iterations):
        clusters = Clusters(path, "en", first_num_clusters, second_num_clusters)
        clusters.form_clusters()

        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
        model = model.to('cuda')

        train_utterances = clusters.cluster_train_df['sentences'].tolist()
        train_clusters = clusters.cluster_train_df['cluster'].tolist()

        from tqdm import tqdm

        train_embeddings = torch.from_numpy(clusters.train_embs)
        test_embeddings = torch.from_numpy(clusters.test_embs)
        empty_emb = model.encode("")

        metric = {1 : [], 3 : [], 5 : [], 10 : []}
        num = 0
        all_num = 0
        index = 0

        for obj in tqdm(clusters.test_dataset):
            utterance_metric = {1 : [], 3 : [], 5 : [], 10 : []}

            for j in range(len(obj)):
                all_num += 1
                utterance_history = " "

                if j > 0:
                    context_encoding = clusters.cluster_test_df["cluster"][index - 1]
                else:
                    context_encoding = empty_emb

                cur_cluster = clusters.cluster_test_df["cluster"][index]
                probs = context_encoding.dot(train_embeddings.T)

                scores = list(zip(probs, train_clusters))
                sorted_scores = list(map(lambda x: x[1], sorted(scores, reverse = True)))

                result_clusters = []

                for cluster in sorted_scores:
                    if cluster not in result_clusters:
                        result_clusters.append(cluster)
                    if len(result_clusters) == 10:
                        break

                for k in [1, 3, 5, 10]:
                    if cur_cluster in result_clusters[:k]:
                        utterance_metric[k].append(1) 
                    else:
                        utterance_metric[k].append(0) 
                index += 1

            for k in [1, 3, 5, 10]:
                metric[k].append(np.mean(utterance_metric[k])) 

#         file.write("ACCURACY METRIC\n")

#         for k in [1, 3, 5, 10]:
#             file.write(f"Acc@{k}: {np.mean(metric[k])}\n")
#     file.close()