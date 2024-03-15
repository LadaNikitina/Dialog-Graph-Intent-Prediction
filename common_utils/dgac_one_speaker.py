from collections import Counter
from datasets import load_dataset
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer

import json
import torch
import faiss
import itertools
import numpy as np
import pandas as pd
import random
import torch

def get_pd_utterances_speaker(data):
    '''
        parsing data
    '''
    utterances = []

    for obj in data:
        utterances += obj['utterance']

    speakers = []

    for obj in data:
        speakers += obj['speaker']    
    
    df = pd.DataFrame()
    
    df['utterance'] = utterances
    df['speaker'] = speakers
    
    return df

class Clusters:
    ''' 
        the class that forms clusters
    '''
    def __init__(self, data_path, language, first_num_clusters, second_num_clusters):
        self.data_path = data_path
        self.first_num_clusters = first_num_clusters
        self.second_num_clusters = second_num_clusters
        self.language = language
        
        if self.first_num_clusters == -1:
            self.first_num_clusters = self.second_num_clusters
            self.second_num_clusters = -1
            
    def data_loading(self):
        '''
            data loading
        '''
        with open(self.data_path) as file:
            dataset = json.load(file)
                
       # train-validation splitting 
        random.shuffle(dataset)
        validation_split = int(len(dataset) * 0.6)
        test_split = int(len(dataset) * 0.8)
        
        # you can replace it with your own train/test/valid
        self.test_dataset = dataset[test_split : ]
        self.valid_dataset = dataset[validation_split : test_split]
        self.train_dataset = dataset[ : validation_split]
                
        # get utterances from data
        self.train_df = get_pd_utterances_speaker(self.train_dataset)
        self.valid_df = get_pd_utterances_speaker(self.valid_dataset)
        self.test_df = get_pd_utterances_speaker(self.test_dataset)
        
        self.df = pd.concat([self.train_df, self.valid_df, self.test_df], ignore_index=True)
        
        self.train_index = self.train_df.index
        self.valid_index = self.valid_df.index + len(self.train_df)
        self.test_index = self.test_df.index + len(self.train_df) + len(self.valid_df)
    
    def get_first_clusters(self, embs, n_clusters):
        '''
            first-stage clustering
        '''
        kmeans = faiss.Kmeans(embs.shape[1], n_clusters, verbose = True, max_points_per_centroid = 5000)
        kmeans.train(embs)

        _, labels = kmeans.index.search(embs, 1)

        return labels.squeeze()

    def get_embeddings(self):
        '''
            calculating embeddings
        '''
        if self.language == "en":
            self.encoder_model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
        else:
            raise ValueError('Wrong language!')
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder_model = self.encoder_model.to(device)
        embeddings = self.encoder_model.encode(self.df["utterance"]).astype(np.float32, copy=False)
        
        np.save(f"{self.language}_embeddings", embeddings)
        
        self.train_embs = embeddings[self.train_index]
        self.valid_embs = embeddings[self.valid_index]
        self.test_embs = embeddings[self.test_index]

        self.embs_dim = embeddings.shape[1]
    
    def first_stage(self):
        '''
            creating first-stage clusters
        '''
        self.train_df_first_stage = self.train_df.copy()

        self.train_df_first_stage['cluster'] = self.get_first_clusters(self.train_embs, self.first_num_clusters)

        # counting center of mass of the cluster
        self.mean_emb = np.zeros((self.first_num_clusters, self.embs_dim))

        for i in range(self.first_num_clusters):
            index_cluster = self.train_df_first_stage[self.train_df_first_stage['cluster'] == i].index
            self.mean_emb[i] = np.mean(self.train_embs[index_cluster], axis = 0)

        # counting word2vec embeddings of the cluster
        index = 0
        array_for_word2vec = []

        for obj in self.train_dataset:
            utterance_clusters = []

            for j in range(len(obj)):
                utterance_clusters.append(str(self.train_df_first_stage["cluster"][index]))
                index += 1

            array_for_word2vec.append(utterance_clusters)       

        model_first_stage = Word2Vec(sentences = array_for_word2vec, sg = 0, min_count = 1, workers = 4, window = 10, epochs = 20)

        # counting final embeddings of the clusters
        self.cluster_embs_first_stage = []

        for i in range(self.first_num_clusters):
            self.cluster_embs_first_stage.append(list(model_first_stage.wv[str(i)]) + list(self.mean_emb[i]))
    
    def get_validation_vs_test_clusters(self, num_clusters):
        '''
            cluster searching for validation and test
        '''
        self.cluster_valid_df = self.valid_df.copy()

        # searching the nearest cluster for each validation utterance
        valid_clusters = []

        for i in range(len(self.cluster_valid_df)):
            distances = []
            emb = np.array(self.valid_embs[i])

            for j in range(num_clusters):
                distances.append((np.sqrt(np.sum(np.square(emb - self.mean_emb[j]))), j))

            distances = sorted(distances)
            valid_clusters.append(distances[0][1])

        self.cluster_valid_df['cluster'] = valid_clusters 
        
        self.cluster_test_df = self.test_df.copy()

        # searching the nearest cluster for each test utterance
        test_clusters = []

        for i in range(len(self.cluster_test_df)):
            distances = []
            emb = np.array(self.test_embs[i])

            for j in range(num_clusters):
                distances.append((np.sqrt(np.sum(np.square(emb - self.mean_emb[j]))), j))

            distances = sorted(distances)
            test_clusters.append(distances[0][1])

        self.cluster_test_df['cluster'] = test_clusters 
        
    
    def utterance_cluster_search(self, utterance, speaker):
        distances = []
        utterance_embedding = self.encoder_model.encode(utterance)
        
        for j in range(self.second_num_clusters):
            distances.append((np.sqrt(np.sum(np.square(utterance_embedding - self.mean_emb[j]))), j))

        distances = sorted(distances)
        return distances[0][1], utterance_embedding
            
    def one_stage_clustering(self):
        '''
            one stage clustering
        '''
        self.get_validation_vs_test_clusters(self.first_num_clusters)
        self.cluster_embs = self.cluster_embs_first_stage
        self.cluster_train_df = self.train_df_first_stage
          
    def second_stage(self):
        '''
            creating second-stage clusters
        '''
        # creating user second-stage clusters
        self.train_df_sec_stage = self.train_df.copy()

        kmeans = faiss.Kmeans(np.array(self.cluster_embs_first_stage).shape[1], self.second_num_clusters)
        kmeans.train(np.array(self.cluster_embs_first_stage).astype(np.float32, copy=False))
    
        _, first_stage_clusters = kmeans.index.search(np.array(self.cluster_embs_first_stage).astype(np.float32, copy=False), 1)
        first_stage_clusters = first_stage_clusters.squeeze()
        
        new_clusters = []

        for i in range(len(self.train_df_first_stage)):
            cur_cluster = self.train_df_first_stage['cluster'][i]
            new_clusters.append(first_stage_clusters[cur_cluster])

        self.train_df_sec_stage['cluster'] = new_clusters

        # counting center of mass of the cluster        
        self.mean_emb = np.zeros((self.second_num_clusters, self.embs_dim))

        for i in range(self.second_num_clusters):
            index_cluster = self.train_df_sec_stage[self.train_df_sec_stage['cluster'] == i].index
            self.mean_emb[i] = np.mean(self.train_embs[index_cluster], axis = 0)

        # counting word2vec embeddings of the cluster
        index = 0
        array_for_word2vec = []

        for obj in self.train_dataset:
            utterance_clusters = []

            for j in range(len(obj)):
                utterance_clusters.append(str(self.train_df_sec_stage["cluster"][index]))
                index += 1

            array_for_word2vec.append(utterance_clusters)       

        model_sec_stage = Word2Vec(sentences = array_for_word2vec, sg = 0, min_count = 1, workers = 4, window = 10, epochs = 20)

        # counting final embeddings of the clusters
        self.cluster_embs_sec_stage = []

        for i in range(self.second_num_clusters):
            self.cluster_embs_sec_stage.append(list(model_sec_stage.wv[str(i)]) + list(self.mean_emb[i]))
    
    def two_stage_clustering(self):
        '''
            two_stage_clustering
        '''

        self.get_validation_vs_test_clusters(self.second_num_clusters)
        self.cluster_embs = self.cluster_embs_sec_stage
        self.cluster_train_df = self.train_df_sec_stage
        
    def form_clusters(self):
        '''
            formation of clusters
        '''
        print("The data is loading...")
        self.data_loading()
        print("The embeddings are loading...")
        self.get_embeddings()
        print("The first stage of clustering has begun...")
        self.first_stage()
        
        if self.second_num_clusters == -1:
            print("The searching clusters for validation and test has begun...")
            self.one_stage_clustering()
        else:
            print("The second stage of clustering has begun...")
            self.second_stage()
            print("The searching clusters for validation and test has begun...")
            self.two_stage_clustering()
            
def get_accuracy_k(k, test_df, probabilities, data):
    '''
        metric function
    '''
    index = 0
    metric = []
    
    for obj in data:
        utterence_metric = []

        for i in range(len(obj)):
            cur_cluster = test_df["cluster"][index]
                
            top = []
                    
            for j in range(len(probabilities[index][:])):
                top.append((probabilities[index][j], j))
                        
            top.sort(reverse=True)
            top = top[:k]

            if (probabilities[index][cur_cluster], cur_cluster) in top:
                utterence_metric.append(1)
            else:
                utterence_metric.append(0)
            index += 1
                
        metric.append(np.array(utterence_metric).mean()) 
    return np.array(metric).mean()
