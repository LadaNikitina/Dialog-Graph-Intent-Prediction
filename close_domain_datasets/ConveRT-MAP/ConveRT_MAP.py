import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4,5"

import logging
import math
import random
import sys
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import dgl
import dgl.nn.pytorch as dglnn
from dgl.dataloading import GraphDataLoader

import networkx as nx

import tensorflow as tf
import tensorflow_hub as tfhub
import tensorflow_text

from datasets import load_dataset

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, MessagePassing
from torch_scatter import scatter_add

from torch.nn.modules.normalization import LayerNorm

print(torch.cuda.device_count())

# The following setting allows the TF1 model to run in TF2
tf.compat.v1.disable_eager_execution()

# setting the logging verbosity level to errors-only
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

path = input("Dataset path: ")

import sys
sys.path.insert(1, '/cephfs/home/ledneva/lrec_coling/common_utils/')

from dgac_two_speakers import Clusters, get_accuracy_k, get_all_accuracy_k, get_encoder_metrics
from ConveRT import SentenceEncoder, Encoder_MAP

num_iterations = 3

num_clusters_list = [(200, 30), (400, 60), (800, 120)]
for (first_num_clusters, second_num_clusters) in num_clusters_list:
    for iteration in range(num_iterations):
        clusters = Clusters(path, "en", first_num_clusters, second_num_clusters)
        clusters.form_clusters()

        sentence_encoder = SentenceEncoder(multiple_contexts=True)
        
        train_user_utterances = clusters.train_user_df['utterance'].tolist()
        train_system_utterances = clusters.train_system_df['utterance'].tolist()
        train_user_clusters = clusters.train_user_df['cluster'].tolist()
        train_system_clusters = clusters.train_system_df['cluster'].tolist()

        train_user_utterances = clusters.train_user_df['utterance'].tolist()
        train_system_utterances = clusters.train_system_df['utterance'].tolist()
        valid_user_utterances = clusters.valid_user_df['utterance'].tolist()
        valid_system_utterances = clusters.valid_system_df['utterance'].tolist()

        batches_train_user_utterances = np.array_split(train_user_utterances, 2500)
        batches_train_system_utterances = np.array_split(train_system_utterances, 2500)
        batches_valid_user_utterances = np.array_split(valid_user_utterances, 100)
        batches_valid_system_utterances = np.array_split(valid_system_utterances, 100)

        from tqdm import tqdm

        train_user_embeddings = np.concatenate([
            sentence_encoder.encode(train_user_utterances)
            for train_user_utterances in tqdm(batches_train_user_utterances)
        ])

        train_system_embeddings = np.concatenate([
            sentence_encoder.encode(train_system_utterances) 
            for train_system_utterances in tqdm(batches_train_system_utterances)
        ])
        valid_user_embeddings = np.concatenate([
            sentence_encoder.encode(valid_user_utterances)
            for valid_user_utterances in batches_valid_user_utterances
        ])

        valid_system_embeddings = np.concatenate([
            sentence_encoder.encode(valid_system_utterances)
            for valid_system_utterances in batches_valid_system_utterances
        ])
        
        top_k = 5
        
        def get_data(dataset, user_embs, system_embs):
            ''' create pairs context-response '''
            data = []

            ind_user = 0
            ind_system = 0
            for obj in tqdm(dataset):
                for j in range(len(obj["utterance"])):
                    utterances_histories = [""]

                    if j > 0:
                        for k in range(max(0, j - top_k), j):
                            utterances_histories.append(obj[k])

                    utterance_emb = sentence_encoder.encode_multicontext(utterances_histories)

                    if obj['speaker'][j] == 0:
                        data.append((utterance_emb, user_embs[ind_user]))
                        ind_user += 1
                    else:
                        data.append((utterance_emb, system_embs[ind_system]))
                        ind_system += 1

            data_loader = DataLoader(data, batch_size=32, shuffle=True)

            return data_loader

        train_loader = get_data(clusters.train_dataset, train_user_embeddings, train_system_embeddings)
        valid_loader = get_data(clusters.valid_dataset, valid_user_embeddings, valid_system_embeddings)
        
        import random
        num_negative_samples = 5

        def generate_negative_samples(context_emb, response_emb, num_samples):
            batch_size = context_emb.shape[0]
            neg_context_samples = []
            neg_response_samples = []
            pos_context_samples = []
            pos_response_samples = []


            for i in range(batch_size):
                indexes = list(range(batch_size))
                indexes.remove(i)

                random_responses = response_emb[random.sample(indexes, num_samples)]
                neg_context_samples.extend([context_emb[i]] * num_samples)
                neg_response_samples.extend(random_responses)

                pos_context_samples.append(context_emb[i])
                pos_response_samples.append(response_emb[i])

            neg_context_samples = torch.stack(neg_context_samples)
            neg_response_samples = torch.stack(neg_response_samples)
            pos_context_samples = torch.stack(pos_context_samples)
            pos_response_samples = torch.stack(pos_response_samples)

            return neg_context_samples, neg_response_samples, \
                   pos_context_samples, pos_response_samples

        def get_negative_data(pos_loader):
            batches = []
            for batch in tqdm(pos_loader):
                context_emb = batch[0]
                response_emb = batch[1]

                samples = generate_negative_samples(context_emb, response_emb, num_negative_samples)
                batches.append(samples)
            return batches

        train_samples = get_negative_data(train_loader)
        valid_samples = get_negative_data(valid_loader)
        
        from torch import nn
        from torch import linalg as LA

        class FeedForward2(nn.Module):
            """Fully-Connected 2-layer Linear Model"""

            def __init__(self, feed_forward2_hidden, num_embed_hidden):
                super().__init__()
                self.linear_1 = nn.Linear(feed_forward2_hidden, feed_forward2_hidden)
                self.linear_2 = nn.Linear(feed_forward2_hidden, feed_forward2_hidden)
                self.norm1 = nn.LayerNorm(feed_forward2_hidden)
                self.norm2 = nn.LayerNorm(feed_forward2_hidden)
                self.final = nn.Linear(feed_forward2_hidden, num_embed_hidden)
                self.orthogonal_initialization()

            def orthogonal_initialization(self):
                for l in [self.linear_1, self.linear_2]:
                    torch.nn.init.xavier_uniform_(l.weight)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = x + F.gelu(self.linear_1(self.norm1(x)))
                x = x + F.gelu(self.linear_2(self.norm2(x)))

                return F.normalize(self.final(x), dim=1, p=2)

        model = Encoder_MAP(768, 1024)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        
        max_epochs = 20
        best_valid_loss = 100.0
        for epoch in range(max_epochs):
            model.train()
            train_losses = []

            for batch in tqdm(train_samples):
                optimizer.zero_grad()

                neg_context_emb = batch[0]
                neg_response_emb = batch[1]
                pos_context_emb = batch[2]
                pos_response_emb = batch[3]

                pos_cos_sim = model(pos_context_emb, pos_response_emb)
                neg_cos_sim = model(neg_context_emb, neg_response_emb)

                loss = model.calculate_loss(pos_cos_sim, neg_cos_sim)
                train_losses.append(loss.item())
                loss.backward()
                optimizer.step()

            avg_train_loss = np.mean(train_losses)

            model.eval()
            with torch.no_grad():
                val_losses = []
                for batch in tqdm(valid_samples):
                    neg_context_emb = batch[0]
                    neg_response_emb = batch[1]
                    pos_context_emb = batch[2]
                    pos_response_emb = batch[3]

                    pos_cos_sim = model(pos_context_emb, pos_response_emb)
                    neg_cos_sim = model(neg_context_emb, neg_response_emb)

                    val_loss = model.calculate_loss(pos_cos_sim, neg_cos_sim)
                    val_losses.append(val_loss.item())

                avg_val_loss = np.mean(val_losses)
                if avg_val_loss > best_valid_loss:
                    break

                best_valid_loss = avg_val_loss
                print(f"Epoch {epoch} | Train Loss: {avg_train_loss} | Validation Loss: {avg_val_loss}")      
        
        with torch.no_grad():
            convert_train_user_embeddings = model.encode_reply(torch.from_numpy(train_user_embeddings)).cpu().numpy()
            convert_train_system_embeddings = model.encode_reply(torch.from_numpy(train_system_embeddings)).cpu().numpy()
            
#             user_metric, system_metric = get_encoder_metrics(
#                 clusters, 
#                 sentence_encoder, 
#                 model,
#                 convert_train_user_embeddings,
#                 convert_train_system_embeddings,
                  top_k
#             )

#         for k in [1, 3, 5, 10]:
#             file.write(f"Acc@{k}: {np.mean(user_metric[k])}\n")