from torch_geometric.utils import add_self_loops
from utils import init_seed, _norm
from tqdm import tqdm
import numpy as np
import torch
import os

def preprocessing(x, y, batch_size, top_k, embs, uttr_embs, num_clusters, shuffle):
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="4"
    null_cluster = 2 * num_clusters
    
    data_len = len(x)
    indexes = np.arange(data_len)
    uttr_embs_dim = len(uttr_embs[0][0])


    if shuffle == 1:
        np.random.shuffle(indexes)
        
    # create batches
    # all batches must be the same size
    # solution to the problem: add graphs with the same structures, 
    # but zero clusters (clusters without replicas)
    x_ = np.concatenate((x[indexes],  \
                         np.full((batch_size - data_len % batch_size, top_k), \
                                  null_cluster + 1)), axis = 0)
    y_ = np.concatenate((y[indexes], \
                         np.full((batch_size - data_len % batch_size), -1)), \
                         axis = 0)
    uttr_embs_ = np.concatenate((uttr_embs[indexes], \
                                 np.zeros((batch_size - data_len % batch_size, \
                                           
                                          top_k, uttr_embs_dim))), axis = 0)
    
    batches_x = np.reshape(x_, (len(x_) // batch_size, batch_size, top_k))
    batches_y = np.reshape(y_, (len(y_) // batch_size, batch_size))
    batches_embs = np.reshape(uttr_embs_, (len(uttr_embs_) // batch_size, 
                                           batch_size * top_k, uttr_embs_dim))
    
    num_batches = len(batches_x)
    matrices = []
    all_node_features = []
    all_labels = []
    num_edge_types = 3
    
    for ind_batch in tqdm(range(num_batches)):
#         print(ind_batch, '/', num_batches)
        batch_x = batches_x[ind_batch]
        batch_embs = batches_embs[ind_batch]
        labels = batches_y[ind_batch]
        
        len_batch = len(batch_x)
        get_features = lambda i: embs[i]
#         node_features = get_features(np.reshape(batch_x, (len_batch * top_k)))
        node_features = batch_embs
        edges = np.zeros((num_edge_types, len_batch * top_k, len_batch * top_k))
           
        # build adjacency matrix for each edge type
        for ind_graph in range(len_batch):
            graph = batch_x[ind_graph]
            
            for i in range(top_k - 1):
                j = i + ind_graph * top_k
                if graph[i + 1] == null_cluster:
                    edges[2][j][j + 1] = 1
                elif graph[i + 1] > num_clusters:
                    edges[0][j][j + 1] = 1
                else:
                    edges[1][j][j + 1] = 1
        
        num_nodes = edges[0].shape[0]
        adjacency_matrix = []
        
        # https://github.com/seongjunyun/Graph_Transformer_Networks/blob/master/main.py
        for i, edge in enumerate(edges):
            # for fastGTN and sparse matrices multiplication
            edge_tmp = torch.from_numpy(np.vstack((edge.nonzero()[1], \
                                                   edge.nonzero()[0]))).type(torch.cuda.LongTensor)
            value_tmp = torch.ones(edge_tmp.shape[1]).type(torch.cuda.FloatTensor)
            
            # normalize each adjacency matrix
            edge_tmp, value_tmp = add_self_loops(edge_tmp, edge_attr = value_tmp,
                                                 fill_value = 1e-20, num_nodes=num_nodes)
            deg_inv_sqrt, deg_row, deg_col = _norm(edge_tmp.detach(), num_nodes, value_tmp.detach())
            value_tmp = deg_inv_sqrt[deg_row] * value_tmp
            adjacency_matrix.append((edge_tmp, value_tmp))

        edge_tmp = torch.stack((torch.arange(0, num_nodes), \
                                torch.arange(0, num_nodes))).type(torch.cuda.LongTensor)
        value_tmp = torch.ones(num_nodes).type(torch.cuda.FloatTensor)
        adjacency_matrix.append((edge_tmp, value_tmp))
                    
        # edges[0] - from user/null in sys
        # edges[1] - from sys/null in user
        # edges[2] - to null
        matrices.append(adjacency_matrix)
        all_node_features.append(torch.from_numpy(node_features).type(torch.cuda.FloatTensor))
        all_labels.append(labels)

    return matrices, all_node_features, all_labels