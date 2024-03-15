import os


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7,8,9"

import torch

print(torch.cuda.device_count())

from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
from catboost import CatBoostClassifier
from catboost import Pool

import pandas as pd
import numpy as np
import networkx as nx
import sys

from CatBoost_functions import get_features, get_data

path = input("Dataset path: ")
sys.path.insert(1, '/cephfs/home/ledneva/lrec_coling/common_utils/')

from dgac_one_speaker import Clusters, get_accuracy_k

num_iterations = 3

num_clusters_list = [(200, 30), (400, 60), (800, 120)]
for (first_num_clusters, second_num_clusters) in num_clusters_list:
    for iteration in range(num_iterations):
        clusters = Clusters(path, "en", first_num_clusters, second_num_clusters)
        clusters.form_clusters()

        top_k = 5
        features, null_features = get_features(clusters.cluster_embs, second_num_clusters)

        train_X, train_y = get_data(features, null_features,
                                              clusters.cluster_train_df, 
                                              clusters.train_dataset, top_k, 
                                              second_num_clusters,
                                              np.array(clusters.train_embs).astype(np.float64, copy=False))
        train_pool = Pool(data=train_X, label=train_y)

        test_X, test_y = get_data(features, null_features,
                                  clusters.cluster_test_df, 
                                  clusters.test_dataset, top_k, 
                                  second_num_clusters,
                                  np.array(clusters.test_embs).astype(np.float64, copy=False))
        test_pool = Pool(test_X)
        

#         valid_X, valid_y = get_data(features, null_features,
#                                    clusters.cluster_valid_df, 
#                                    clusters.valid_dataset, top_k, 
#                                    second_num_clusters,
#                                    np.array(clusters.valid_embs).astype(np.float64, copy=False))

        classif = CatBoostClassifier(iterations = 300, learning_rate = 0.1, random_seed = 43, loss_function = 'MultiClass', task_type = 'GPU')
        classif.fit(train_pool, verbose = 10)

        test_pred = classif.predict_proba(test_pool)
        test_true = test_y['target'].tolist()


#         get_accuracy_k(k, clusters.cluster_test_df, test_pred, clusters.test_dataset)