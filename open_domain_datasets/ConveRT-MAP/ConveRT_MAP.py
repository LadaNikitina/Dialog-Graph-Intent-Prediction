from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
from torch import nn
from torch.nn import Linear

import pandas as pd
import numpy as np
import networkx as nx
import sys
import os
import torch
import math
import tensorflow as tf
import random
import dgl
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn
import torch.nn as nn

from conversational_sentence_encoder.vectorizers import SentenceEncoder
print(torch.cuda.device_count())

# The following setting allows the TF1 model to run in TF2
tf.compat.v1.disable_eager_execution()

# setting the logging verbosity level to errors-only
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="6,7"
print(torch.cuda.device_count())

path = input("Dataset path: ")

import sys
sys.path.insert(1, '/cephfs/home/ledneva/lrec_coling/common_utils/')

from dgac_one_speaker import Clusters, get_accuracy_k
from ConveRT import SentenceEncoder, Encoder_MAP

num_iterations = 3

num_clusters_list = [(200, 30), (400, 60), (800, 120)]
for (first_num_clusters, second_num_clusters) in num_clusters_list:
    for iteration in range(num_iterations):
        clusters = Clusters(path, "en", first_num_clusters, second_num_clusters)

        clusters.form_clusters()

        train_utterances = clusters.cluster_train_df['sentences'].tolist()
        train_clusters = clusters.cluster_train_df['cluster'].tolist()

        valid_utterances = clusters.cluster_valid_df['sentences'].tolist()
        valid_clusters = clusters.cluster_valid_df['cluster'].tolist()

        batches_train_utterances = np.array_split(train_utterances, 2500)
        batches_valid_utterances = np.array_split(valid_utterances, 2500)


        # create pairs context-response
        from tqdm import tqdm 
        from torch.utils.data import DataLoader

        top_k = 5

        sentence_encoder = SentenceEncoder(multiple_contexts=True)

        train_embeddings = np.concatenate([
            sentence_encoder.encode_responses(train_utterances)
            for train_utterances in tqdm(batches_train_utterances)
        ])

        valid_embeddings = np.concatenate([
            sentence_encoder.encode_responses(valid_utterances)
            for valid_utterances in tqdm(batches_valid_utterances)
        ])


        def get_data(dataset, embs):
            ''' create pairs context-response '''
            num_negative = 5
            data = []

            index = 0
            for obj in tqdm(dataset):
                for j in range(len(obj)):
                    utterances_histories = [""]

                    if j > 0:
                        for k in range(max(0, j - top_k), j):
                            utterances_histories.append(obj[k])

                    utterance_emb = sentence_encoder.encode_multicontext(utterances_histories)

                    data.append((utterance_emb, embs[index]))
                    index += 1

            data_loader = DataLoader(data, batch_size=32, shuffle=True)

            return data_loader


        train_loader = get_data(clusters.train_dataset, train_embeddings)
        valid_loader = get_data(clusters.valid_dataset, valid_embeddings)

        import random
        def generate_negative_samples(context_emb, response_emb, num_samples):
            batch_size = context_emb.shape[0]
            negative_context_samples = []
            negative_response_samples = []


            for i in range(batch_size):
                indexes = list(range(batch_size))
                indexes.remove(i)
                random_responses = response_emb[random.sample(indexes, num_samples)]
                negative_context_samples.extend([context_emb[i]] * num_samples)
                negative_response_samples.extend(random_responses)

            negative_context_samples = torch.stack(negative_context_samples)
            negative_response_samples = torch.stack(negative_response_samples)
            return negative_context_samples, negative_response_samples

        def get_negative_data(pos_loader):
            neg_batches = []
            for batch in tqdm(train_loader):
                context_emb = batch[0]
                response_emb = batch[1]

                num_negative_samples = 5
                neg_batches.append(generate_negative_samples(context_emb, response_emb, num_negative_samples))
            return neg_batches

        train_negative_samples = get_negative_data(train_loader)
        valid_negative_samples = get_negative_data(valid_loader)

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

        import logging
        from collections import OrderedDict

        import numpy as np
        import torch

        from torch.nn.modules.normalization import LayerNorm

        class ConveRT_MAP(nn.Module):
            def __init__(self, feed_forward2_hidden, num_embed_hidden):
                super().__init__()
                self.ff2_context = FeedForward2(feed_forward2_hidden, num_embed_hidden)
                self.ff2_reply = FeedForward2(feed_forward2_hidden, num_embed_hidden)


            def forward(self, context_emb, response_emb):
                context = self.ff2_context(context_emb.reshape(context_emb.shape[0], context_emb.shape[2]))
                response = self.ff2_reply(response_emb)
                cosine_sim = F.cosine_similarity(context, response, dim=1)
                return cosine_sim

            def calculate_loss(self, cos_sim_pos, cos_sim_neg):
        #         pos_loss = torch.mean(pos_cos_sim)
        #         neg_loss = torch.mean(neg_cos_sim)
                total_loss = torch.mean(torch.pow(cos_sim_neg, 2)) + torch.mean(torch.pow(torch.clamp(1.0 - cos_sim_pos, min=0.0), 2))
        #         total_loss = torch.max(torch.tensor(0.0), 1.0 - pos_loss + neg_loss)
                return total_loss

            def encode_context(self, x):
                return self.ff2_context(x)

            def encode_reply(self, x):
                return self.ff2_reply(x)

        model = ConveRT_MAP(512, 1024)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)


        max_epochs = 20
        best_valid_loss = 100.0
        for epoch in range(max_epochs):
            model.train()
            train_losses = []

            for i, batch in tqdm(enumerate(train_loader)):
                optimizer.zero_grad()

                context_emb = batch[0]
                response_emb = batch[1]
                pos_cos_sim = model(context_emb, response_emb)

        #         # Generate negative samples
                negative_context_samples, negative_response_samples = train_negative_samples[i][0], train_negative_samples[i][1]
                neg_cos_sim = model(negative_context_samples, negative_response_samples)

                loss = model.calculate_loss(pos_cos_sim, neg_cos_sim)
                train_losses.append(loss.item())
                loss.backward()
                optimizer.step()

            avg_train_loss = np.mean(train_losses)

            model.eval()
            with torch.no_grad():
                val_losses = []
                for i, batch in tqdm(enumerate(valid_loader)):
                    context_emb = batch[0]
                    response_emb = batch[1]
                    pos_cos_sim = model(context_emb, response_emb)

                    # Generate negative samples
                    negative_context_samples, negative_response_samples = valid_negative_samples[i][0], valid_negative_samples[i][1]
                    neg_cos_sim = model(negative_context_samples, negative_response_samples)

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

        with torch.no_grad():
            convert_train_embeddings = model.encode_reply(torch.from_numpy(train_embeddings)).cpu().numpy()

        with torch.no_grad():
            metric = {1 : [], 3 : [], 5 : [], 10 : []}
            num = 0
            all_num = 0
            index = 0

            for obj in tqdm(clusters.test_dataset):
                utterance_metric = {1 : [], 3 : [], 5 : [], 10 : []}

                for j in range(len(obj)):
                    all_num += 1
                    utterances_histories = [""]

                    if j > 0:
                        for k in range(max(0, j - top_k), j):
                            utterances_histories.append(obj[k])

                    convert_encoding = sentence_encoder.encode_multicontext(utterances_histories)
                    context_encoding = model.encode_context(torch.from_numpy((convert_encoding)))[0].cpu().numpy()

                    cur_cluster = clusters.cluster_test_df["cluster"][index]
                    probs = context_encoding.dot(convert_train_embeddings.T)

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

#             file.write("ACCURACY METRIC\n")

#             for k in [1, 3, 5, 10]:
#                 file.write(f"Acc@{k}: {np.mean(metric[k])}\n")