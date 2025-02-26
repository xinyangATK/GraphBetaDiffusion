import torch
import numpy as np

import os
import time
import pickle
import math
import torch
from torch_geometric.utils import to_networkx
from GDSS_utils.utils.graph_utils import adjs_to_graphs, init_flags, quantize, quantize_mol
from GDSS_utils.evaluation.stats import eval_graph_list, nspdk_stats
from GDSS_utils.utils.mol_utils import gen_mol, mols_to_smiles, load_smiles, canonicalize_smiles, mols_to_nx
from GDSS_utils.evaluation.molsets import get_all_metrics

class EVAL_METRICS:
    def __init__(self, dataset, num_mols=10000):
        self.dataset = dataset
        self.num_mols = num_mols
        self.scores = {}
        # self.train_graphs = self.loader_to_nx(datamodule.train_dataloader())
        # self.val_graphs = self.loader_to_nx(datamodule.val_dataloader())
        # self.test_graphs = self.loader_to_nx(datamodule.test_dataloader())

    def loader_to_nx(self, loader):
        networkx_graphs = []
        for i, batch in enumerate(loader):
            data_list = batch.to_data_list()
            for j, data in enumerate(data_list):
                networkx_graphs.append(to_networkx(data, node_attrs=None, edge_attrs=None, to_undirected=True,
                                                   remove_self_loops=True))
        return networkx_graphs

    def process_graph(self, X, E):
        '''
            X: [bs, n, num_atom_class]
            E: [bs, n, n, num_edge_class], {'no': 0, 'S': 1, 'D': 2, 'T': 3, 'mask': -1}

            Convert (X, E) into (X_new, E_new)
            X_new: [bs, n, num_atom_class + 1], with virtual node
            E_new: [bs, n, n, num_edge_class + 1], with virtual edge
        '''
        if self.dataset == 'QM9':
            # 32, 9, 4 -> 32, 9, 5
            X = X - 1
            X[X ==- 2] = 3  # set last place as virtual node
            X = X + 1
            new_X = torch.nn.functional.one_hot(X, num_classes=5)
        else:
            assert self.dataset == 'ZINC250K'
            # 32, 38, 9 -> 32, 38, 10
            X = X - 1
            X[X ==- 2] = 8  # set last place as virtual node
            X = X + 1
            new_X = torch.nn.functional.one_hot(X, num_classes=10)

        E[E == -1] = 0  # 'mask' ===> 'no edge' : 0
        new_E = E - 1
        new_E[new_E == -1] = 3  # 0, 1, 2, 3 (no, S, D, T) -> 3, 0, 1, 2
        new_E = torch.nn.functional.one_hot(new_E, num_classes=4)  # (S, D, T, no)

        return new_X, new_E

    def gen_mol_from_graph(self, X, E):
        gen_mols, num_mols_wo_correction = gen_mol(X, E, self.dataset)

        return gen_mols, num_mols_wo_correction

    def smiles_form_mol(self, gen_mols):
        gen_smiles = mols_to_smiles(gen_mols)
        gen_smiles = [smi for smi in gen_smiles if len(smi)]

        return gen_smiles


    def eval_generic_graph(self):

        pass

    def eval_mol_graph_scores(self, gen_smiles, device='cpu', n_jobs=1, test_smiles=None, train_smiles=None):
        scores = get_all_metrics(gen=gen_smiles, k=len(gen_smiles), n_jobs=n_jobs, device=device, test=test_smiles, train=train_smiles)
        self.scores.update(scores)

        return scores

    def eval_mol_graph_nspdk(self, test_graphs, gen_mol):
        nspdk_score = nspdk_stats(test_graphs, mols_to_nx(gen_mol))
        nspdk_score = round(nspdk_score, 4)
        self.scores.update({'NSPDK/Test': nspdk_score})

        return nspdk_score

    def print_results(self, log2file=None):
        metrics_list = ['FCD/Test', 'Scaf/Test', 'Frag/Test', 'SNN/Test', f'unique@{self.num_mols}', 'Novelty', 'valid']
        if 'NSPDK/Test' in self.scores.keys():
            metrics_list.append('NSPDK/Test')
        for metric in metrics_list:
            if log2file is not None:
                log2file.info(f'{metric}: {self.scores[metric]}\n')
            print(f'{metric}: {self.scores[metric]}')
        print('='*100)

