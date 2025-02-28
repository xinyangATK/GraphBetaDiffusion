import torch
import numpy as np
import torch.nn.functional as F


def order_graph(self, X, E, node_mask, node_idx):
    reordered_X, reordered_E, reordered_node_mask = [], [], []
    for g_id in range(node_idx.size(0)):
        node_order = node_idx[g_id]
        reordered_X.append(X[g_id][node_order])
        reordered_E.append(E[g_id][node_order, :][:, node_order])
        reordered_node_mask.append(node_mask[g_id][node_order])
    X = torch.stack(reordered_X, dim=0)
    E = torch.stack(reordered_E, dim=0)
    node_mask = torch.stack(reordered_node_mask, dim=0)

    return X, E, node_mask