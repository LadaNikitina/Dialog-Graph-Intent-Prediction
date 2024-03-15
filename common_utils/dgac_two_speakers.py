from collections import Counter
from datasets import load_dataset
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

import torch
import numpy as np
import json
import itertools
import pandas as pd
import faiss
import random
import os

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
        train_df = get_pd_utterances_speaker(self.train_dataset)
        valid_df = get_pd_utterances_speaker(self.valid_dataset)
        test_df = get_pd_utterances_speaker(self.test_dataset)
        
        self.df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

        # split data on user/system train/valid/test
        self.user_train_df = train_df[train_df["speaker"] == 0].reset_index(drop=True)
        self.user_valid_df = valid_df[valid_df["speaker"] == 0].reset_index(drop=True)
        self.user_test_df = test_df[test_df["speaker"] == 0].reset_index(drop=True)
        
        self.system_train_df = train_df[train_df["speaker"] == 1].reset_index(drop=True)
        self.system_valid_df = valid_df[valid_df["speaker"] == 1].reset_index(drop=True)
        self.system_test_df = test_df[test_df["speaker"] == 1].reset_index(drop=True)
        
        # get user/system train/valid/test indexes for getting embeddings
        self.user_train_index = train_df[train_df["speaker"] == 0].index
        self.user_valid_index = valid_df[valid_df["speaker"] == 0].index + len(train_df)
        self.user_test_index = test_df[test_df["speaker"] == 0].index + len(train_df) + len(valid_df)
        
        self.system_train_index = train_df[train_df["speaker"] == 1].index
        self.system_valid_index = valid_df[valid_df["speaker"] == 1].index + len(train_df)
        self.system_test_index = test_df[test_df["speaker"] == 1].index + len(train_df) + len(valid_df)        

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
        embeddings = self.encoder_model.encode(self.df["utterance"])
        
        np.save(f"{self.language}_embeddings", embeddings)
        
        self.embs_dim = embeddings.shape[1]
        # train user/system embeddings
        self.train_user_embs = embeddings[self.user_train_index]
        self.train_system_embs = embeddings[self.system_train_index]

        # validation user/system embeddings
        self.valid_user_embs = embeddings[self.user_valid_index]
        self.valid_system_embs = embeddings[self.system_valid_index]
       
        # test user/system embeddings
        self.test_user_embs = embeddings[self.user_test_index]
        self.test_system_embs = embeddings[self.system_test_index]
        
    
    def get_first_clusters(self, embs, n_clusters):
        '''
            first-stage clustering
        '''
        kmeans = faiss.Kmeans(embs.shape[1], n_clusters, verbose = True, max_points_per_centroid = 5000)
        kmeans.train(embs.astype(np.float32, copy=False))

        _, labels = kmeans.index.search(embs.astype(np.float32, copy=False), 1)
        return labels.squeeze()
    
    def first_stage(self):
        '''
            creating first-stage clusters
        '''
        self.train_user_df_first_stage = self.user_train_df.copy()
        self.train_system_df_first_stage = self.system_train_df.copy()

        self.train_user_df_first_stage['cluster'] = self.get_first_clusters(self.train_user_embs, self.first_num_clusters)
        self.train_system_df_first_stage['cluster'] = self.get_first_clusters(self.train_system_embs, self.first_num_clusters)

        # counting center of mass of the cluster
        self.user_mean_emb = np.zeros((self.first_num_clusters, self.embs_dim))
        self.system_mean_emb = np.zeros((self.first_num_clusters, self.embs_dim))

        for i in range(self.first_num_clusters):
            index_cluster = self.train_user_df_first_stage[self.train_user_df_first_stage['cluster'] == i].index
            self.user_mean_emb[i] = np.mean(self.train_user_embs[index_cluster], axis = 0)

            index_cluster = self.train_system_df_first_stage[self.train_system_df_first_stage['cluster'] == i].index
            self.system_mean_emb[i] = np.mean(self.train_system_embs[index_cluster], axis = 0)

        # counting word2vec embeddings of the cluster
        ind_user = 0
        ind_system = 0
        array_for_word2vec = []

        for obj in self.train_dataset:
            utterance_clusters = []

            for j in range(len(obj["utterance"])):
                if obj['speaker'][j] == 0:
                    utterance_clusters.append(str(self.train_user_df_first_stage["cluster"][ind_user]) + "-user")
                    ind_user += 1
                else:
                    utterance_clusters.append(str(self.train_system_df_first_stage["cluster"][ind_system]) + "-system")
                    ind_system += 1

            array_for_word2vec.append(utterance_clusters)       

        model_first_stage = Word2Vec(sentences = array_for_word2vec, sg = 0, min_count = 1, workers = 4, window = 10, epochs = 20)

        # counting final embeddings of the clusters
        self.user_cluster_embs_first_stage = []
        self.system_cluster_embs_first_stage = []

        for i in range(self.first_num_clusters):
            self.user_cluster_embs_first_stage.append(list(model_first_stage.wv[str(i)  + "-user"]))
            self.system_cluster_embs_first_stage.append(list(model_first_stage.wv[str(i)  + "-system"]))
    
    def get_validation_vs_test_clusters(self, num_clusters):
        '''
            cluster searching for validation
        '''

        self.valid_user_df = self.user_valid_df.copy()
        self.valid_system_df = self.system_valid_df.copy()

        # searching the nearest cluster for each validation user utterance
        valid_user_clusters = []

        for i in range(len(self.valid_user_df)):
            distances = []
            emb = np.array(self.valid_user_embs[i])

            for j in range(num_clusters):
                distances.append((np.sqrt(np.sum(np.square(emb - self.user_mean_emb[j]))), j))

            distances = sorted(distances)
            valid_user_clusters.append(distances[0][1])

        self.valid_user_df['cluster'] = valid_user_clusters      

        # searching the nearest cluster for each validation system utterance
        valid_system_clusters = []

        for i in range(len(self.valid_system_df)):
            distances = []
            vec = np.array(self.valid_system_embs[i])

            for j in range(num_clusters):
                distances.append((np.sqrt(np.sum(np.square(vec - self.system_mean_emb[j]))), j))

            distances = sorted(distances)
            valid_system_clusters.append(distances[0][1])

        self.valid_system_df['cluster'] = valid_system_clusters
        
        self.test_user_df = self.user_test_df.copy()
        self.test_system_df = self.system_test_df.copy()

        # searching the nearest cluster for each test user utterance
        test_user_clusters = []

        for i in range(len(self.test_user_df)):
            distances = []
            emb = np.array(self.test_user_embs[i])

            for j in range(num_clusters):
                distances.append((np.sqrt(np.sum(np.square(emb - self.user_mean_emb[j]))), j))

            distances = sorted(distances)
            test_user_clusters.append(distances[0][1])

        self.test_user_df['cluster'] = test_user_clusters      

        # searching the nearest cluster for each test system utterance
        test_system_clusters = []

        for i in range(len(self.test_system_df)):
            distances = []
            vec = np.array(self.test_system_embs[i])

            for j in range(num_clusters):
                distances.append((np.sqrt(np.sum(np.square(vec - self.system_mean_emb[j]))), j))

            distances = sorted(distances)
            test_system_clusters.append(distances[0][1])

        self.test_system_df['cluster'] = test_system_clusters
        
        
    def utterance_cluster_search(self, utterance, speaker):
        distances = []
        utterance_embedding = self.encoder_model.encode(utterance)
        
        if speaker == 0:
            for j in range(self.second_num_clusters):
                distances.append((np.sqrt(np.sum(np.square(utterance_embedding - self.user_mean_emb[j]))), j))
        elif speaker == 1:
            for j in range(self.second_num_clusters):
                distances.append((np.sqrt(np.sum(np.square(utterance_embedding - self.system_mean_emb[j]))), j))

        distances = sorted(distances)
        return distances[0][1], utterance_embedding
        
            
    def one_stage_clustering(self):
        '''
            one stage clustering
        '''
 
        self.get_validation_vs_test_clusters(self.first_num_clusters)
        self.user_cluster_embs = self.user_cluster_embs_first_stage
        self.system_cluster_embs = self.system_cluster_embs_first_stage
        self.train_user_df = self.train_user_df_first_stage
        self.train_system_df = self.train_system_df_first_stage
          
    def second_stage(self):
        '''
            creating second-stage clusters
        '''
        # creating user second-stage clusters
        self.train_user_df_sec_stage = self.user_train_df.copy()

        kmeans = faiss.Kmeans(len(self.user_cluster_embs_first_stage[0]), self.second_num_clusters, min_points_per_centroid=10)
        kmeans.train(np.array(self.user_cluster_embs_first_stage).astype(np.float32, copy=False))
    
        _, user_new_clusters = kmeans.index.search(np.array(self.user_cluster_embs_first_stage).astype(np.float32, copy=False), 1)
        user_new_clusters = user_new_clusters.squeeze()

        new_user_clusters = []

        for i in range(len(self.train_user_df_first_stage)):
            cur_cluster = self.train_user_df_first_stage['cluster'][i]
            new_user_clusters.append(user_new_clusters[cur_cluster])

        self.train_user_df_sec_stage['cluster'] = new_user_clusters
        
        # creating system second-stage clusters
        self.train_system_df_sec_stage = self.system_train_df.copy()

        kmeans = faiss.Kmeans(len(self.system_cluster_embs_first_stage[0]), self.second_num_clusters, min_points_per_centroid=10)
        kmeans.train(np.array(self.system_cluster_embs_first_stage).astype(np.float32, copy=False))
    
        _, system_new_clusters = kmeans.index.search(np.array(self.system_cluster_embs_first_stage).astype(np.float32, copy=False), 1)
        system_new_clusters = system_new_clusters.squeeze()

        new_sys_clusters = []

        for i in range(len(self.train_system_df_first_stage)):
            cur_cluster = self.train_system_df_first_stage['cluster'][i]
            new_sys_clusters.append(system_new_clusters[cur_cluster])

        self.train_system_df_sec_stage['cluster'] = new_sys_clusters

        # counting center of mass of the cluster        
        self.user_mean_emb = np.zeros((self.second_num_clusters, self.embs_dim))
        self.system_mean_emb = np.zeros((self.second_num_clusters, self.embs_dim))

        for i in range(self.second_num_clusters):
            index_cluster = self.train_user_df_sec_stage[self.train_user_df_sec_stage['cluster'] == i].index
            self.user_mean_emb[i] = np.mean(self.train_user_embs[index_cluster], axis = 0)

            index_cluster = self.train_system_df_sec_stage[self.train_system_df_sec_stage['cluster'] == i].index
            self.system_mean_emb[i] = np.mean(self.train_system_embs[index_cluster], axis = 0)

        # counting word2vec embeddings of the cluster
        ind_user = 0
        ind_system = 0
        array_for_word2vec = []

        for obj in self.train_dataset:
            utterance_clusters = []

            for j in range(len(obj["utterance"])):
                if obj['speaker'][j] == 0:
                    utterance_clusters.append(str(self.train_user_df_sec_stage["cluster"][ind_user]) + "-user")
                    ind_user += 1
                else:
                    utterance_clusters.append(str(self.train_system_df_sec_stage["cluster"][ind_system]) + "-system")
                    ind_system += 1

            array_for_word2vec.append(utterance_clusters)       

        model_sec_stage = Word2Vec(sentences = array_for_word2vec, sg = 0, min_count = 1, workers = 4, window = 10, epochs = 20)

        # counting final embeddings of the clusters
        self.user_cluster_embs_sec_stage = []
        self.system_cluster_embs_sec_stage = []

        for i in range(self.second_num_clusters):
            self.user_cluster_embs_sec_stage.append(list(model_sec_stage.wv[str(i)  + "-user"]) + list(self.user_mean_emb[i]))
            self.system_cluster_embs_sec_stage.append(list(model_sec_stage.wv[str(i)  + "-user"]) + list(self.system_mean_emb[i]))
    
    def two_stage_clustering(self):
        '''
            two_stage_clustering
        '''

        self.get_validation_vs_test_clusters(self.second_num_clusters)
        self.user_cluster_embs = self.user_cluster_embs_sec_stage
        self.system_cluster_embs = self.system_cluster_embs_sec_stage
        self.train_user_df = self.train_user_df_sec_stage
        self.train_system_df = self.train_system_df_sec_stage
        
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
            self.one_stage_clustering()
        else:
            print("The second stage of clustering has begun...")
            self.second_stage()
            print("The searching clusters for validation and test has begun...")
            self.two_stage_clustering()
            
            
            
def get_accuracy_k(k, test_df, probabilities, data, flag):
    '''
        metric function, flag: user - speaker 0, system - speaker 1
    '''
    index = 0
    metric = []
    
    for obj in data:
        utterence_metric = []

        for i in range(len(obj["utterance"])):
            if obj['speaker'][i] == flag:
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




def get_all_accuracy_k(k, test_user_data, test_system_data, probs_sys_user, probs_user_sys, data):
    '''
        metric function for both speakers
    '''
    ind_user = 0
    ind_system = 0
    metric = []
    
    for obj in data:
        utterence_metric = []
        pred_cluster = -1

        for i in range(len(obj["utterance"])):
            if obj['speaker'][i] == 0:
                cur_cluster = test_user_data["cluster"][ind_user]
                
                top = []
                    
                for j in range(len(probs_sys_user[ind_user][:])):
                    top.append((probs_sys_user[ind_user][j], j))
                        
                top.sort(reverse=True)
                top = top[:k]

                if (probs_sys_user[ind_user][cur_cluster], cur_cluster) in top:
                    utterence_metric.append(1)
                else:
                    utterence_metric.append(0)
                pred_cluster = cur_cluster   
                ind_user += 1
            else:
                cur_cluster = test_system_data["cluster"][ind_system]
                
                top = []
                    
                for kk in range(len(probs_user_sys[ind_system][:])):
                    top.append((probs_user_sys[ind_system][kk], kk))
                        
                top.sort(reverse=True)
                top = top[:k]

                if (probs_user_sys[ind_system][cur_cluster],cur_cluster) in top:
                    utterence_metric.append(1)
                else:
                    utterence_metric.append(0)
                pred_cluster = cur_cluster  
                ind_system += 1
         
                
        metric.append(np.array(utterence_metric).mean()) 
    return np.array(metric).mean()

def get_encoder_metrics(clusters, sentence_encoder, model, train_user_embs, train_system_embs, top_k):
    train_user_clusters = clusters.train_user_df['cluster'].tolist()
    train_system_clusters = clusters.train_system_df['cluster'].tolist()
    
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

            convert_encoding = sentence_encoder.encode_multicontext(utterances_histories)
            context_encoding = model.encode_context(torch.from_numpy((convert_encoding)))[0].cpu().numpy()

            if obj['speaker'][j] == 0:
                cur_cluster = clusters.test_user_df["cluster"][ind_user]
                probs = context_encoding.dot(train_user_embs.T)

                scores = list(zip(probs, train_user_clusters))
                sorted_scores = list(map(lambda x: x[1], sorted(scores, reverse = True)))
                result_clusters = []

                for cluster in sorted_scores:
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

                probs = context_encoding.dot(train_system_embs.T).tolist()
                scores = list(zip(probs, train_system_clusters))
                sorted_scores = list(map(lambda x: x[1], sorted(scores, reverse = True)))

                result_clusters = []

                for cluster in sorted_scores:
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
            user_metric[k].append(np.mean(user_utterence_metric[k])) 
            system_metric[k].append(np.mean(system_utterence_metric[k])) 
            
    return user_metric, system_metric