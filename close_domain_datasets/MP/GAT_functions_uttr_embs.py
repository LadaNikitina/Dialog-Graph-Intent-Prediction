import torch
import numpy as np
import dgl
from torch.utils.data import DataLoader

def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)

def get_data_dgl_no_cycles(x, y, train_or_test, top_k, batch_size):
    data = []
    y = torch.tensor(y)

    for i in range(len(x)):
        graph = x[i][0]
        embs = x[i][1]
        
        top_k_nums = list(range(top_k))
        g = dgl.graph((top_k_nums[:-1], top_k_nums[1:]), num_nodes = top_k)
        g.ndata['attr'] = torch.tensor(graph)
        g.ndata['emb'] = torch.tensor(embs)
        g = dgl.add_self_loop(g)
        data.append((g, y[i]))
    
    if train_or_test == 1:
        data = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=False,
                         collate_fn=collate)
    else:
        data = DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False,
                         collate_fn=collate)
    return data