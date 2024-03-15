import os


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="5,6,7,8,9"

import torch

print(torch.cuda.device_count())

from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
from catboost import CatBoostClassifier

import pandas as pd
import numpy as np
import networkx as nx
import torch
import sys

from CatBoost_functions import get_features, get_data

sys.path.insert(1, '/cephfs/home/ledneva/lrec_coling/common_utils/')

from dgac_two_speakers import Clusters, get_accuracy_k, get_all_accuracy_k

path = input("Dataset path: ")

num_iterations = 3

num_clusters_list = [(200, 30), (400, 60), (800, 120)]
for (first_num_clusters, second_num_clusters) in num_clusters_list:
    for iteration in range(num_iterations):
        print(f"Iteration number {iteration}")

        clusters = Clusters(path, "en", first_num_clusters, second_num_clusters)

        clusters.form_clusters()

        near_number = 3
        top_k = 5
        num_coords = len(clusters.user_cluster_embs[0])

        user_features, system_features, null_features = get_features(
            clusters.train_user_df, 
            clusters.train_system_df, 
            clusters.train_dataset, 
            np.array(clusters.user_cluster_embs), 
            np.array(clusters.system_cluster_embs), 
            near_number, 
            num_coords,
            second_num_clusters
        )

        user_train_X, user_train_y, system_train_X, system_train_y = get_data(
            user_features.tolist(), 
            system_features.tolist(), 
            null_features,
            clusters.train_user_df, 
            clusters.train_system_df, 
            clusters.train_dataset, 
            top_k, 
            second_num_clusters,
            clusters.train_user_embs,
            clusters.train_system_embs
        )

        user_test_X, user_test_y, system_test_X, system_test_y = get_data(
            user_features.tolist(), 
            system_features.tolist(), 
            null_features,
            clusters.test_user_df, 
            clusters.test_system_df, 
            clusters.test_dataset, 
            top_k, 
            second_num_clusters,
            clusters.test_user_embs,
            clusters.test_system_embs
        )

        user_valid_X, user_valid_y, system_valid_X, system_valid_y = get_data(
            user_features.tolist(), 
            system_features.tolist(), 
            null_features,
            clusters.valid_user_df, 
            clusters.valid_system_df, 
            clusters.valid_dataset, 
            top_k, 
            second_num_clusters,
            clusters.valid_user_embs,
            clusters.valid_system_embs
        )

        user_classif = CatBoostClassifier(
            iterations = 500, learning_rate = 0.1, random_seed = 43, loss_function = 'MultiClass', task_type = 'GPU'
        )
        user_classif.fit(user_train_X, user_train_y, eval_set = [(user_valid_X, user_valid_y)], verbose = 10)

        system_classif = CatBoostClassifier(
            iterations = 500, learning_rate = 0.1, random_seed = 43, loss_function = 'MultiClass', task_type = 'GPU'
        )
        system_classif.fit(system_train_X, system_train_y, eval_set = [(system_test_X, system_test_y)], verbose = 10)

        test_user_pred = user_classif.predict_proba(user_test_X)
        test_sys_pred = system_classif.predict_proba(system_test_X)

        test_user_true = user_test_y['target'].tolist()
        test_sys_true = system_test_y['target'].tolist()

        torch.cuda.empty_cache()
        
#         get_accuracy_k(k, clusters.test_user_df, test_user_pred, clusters.test_dataset, 0)
#         get_accuracy_k(k, clusters.test_system_df, test_sys_pred, clusters.test_dataset, 1)
#         get_all_accuracy_k(k, clusters.test_user_df, clusters.test_system_df, test_user_pred, test_sys_pred, clusters.test_dataset)