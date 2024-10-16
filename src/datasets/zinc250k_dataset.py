import json

from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT

import os
import os.path as osp
import pathlib
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch_geometric.data import Data, InMemoryDataset, download_url
import pandas as pd

from src import utils
from src.analysis.rdkit_functions import mol2smiles, build_molecule_with_partial_charges, compute_molecular_metrics
from src.datasets.abstract_dataset import AbstractDatasetInfos, MolecularDataModule
# from src.GDSS_utils.utils import *
import scipy.sparse as sp


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]




class ZINC250KDataset(InMemoryDataset):

    def __init__(self, stage, root, transform=None, pre_transform=None, pre_filter=None):
        self.stage = stage
        self.data_name = 'ZINC250K'
        if self.stage == 'train':
            self.file_idx = 0
        elif self.stage == 'val':
            self.file_idx = 1
        else:
            self.file_idx = 2

        self.max_node_num = 38
        self.max_feat_num = 40
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])

    @property
    def raw_file_names(self):
        return ['train_zink250k.csv', 'val_zink250k.csv', 'testzink250k.csv']

    @property
    def split_file_name(self):
        return ['train_zink250k.csv', 'val_zink250k.csv', 'test_zink250k.csv']

    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        splitting."""
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        return ['train.pt', 'test.pt', 'test.pt']

    def download(self):
        import rdkit  # noqa
        if not (osp.exists(osp.join(self.raw_dir, f'train_vaild_mols.pt')) and osp.exists(osp.join(self.raw_dir, f'test_vaild_mols.pt'))):
            self.pre_process()

    def load_mol(self, filepath):
        print(f'Loading file {filepath}')
        if not os.path.exists(filepath):
            raise  ValueError(f'Invaild filepath {filepath} for dataset')
        load_data = np.load(filepath)
        res = []
        i = 0
        while True:
            key = f'arr_{i}'
            if key in load_data.keys():
                res.append(load_data[key])
                i += 1
            else:
                break
        return list(map(lambda x, a: (x, a), res[0], res[1]))


    def pre_process(self):
        mols = self.load_mol(os.path.join(self.raw_dir, f'{self.data_name.lower()}_kekulized.npz'))
        with open(os.path.join(self.raw_dir, f'valid_idx_{self.data_name.lower()}.json')) as f:
            test_idx = json.load(f)

        train_idx = [i for i in range(len(mols)) if i not in test_idx]
        print(f'Number of training mols: {len(train_idx)} | Number of test mols: {len(test_idx)}')

        train_mols = [mols[i] for i in train_idx]
        test_mols = [mols[i] for i in test_idx]

        torch.save(train_mols, osp.join(self.raw_dir, f'train_vaild_mols.pt'))
        torch.save(test_mols, osp.join(self.raw_dir, f'test_vaild_mols.pt'))

    def process(self):
        RDLogger.DisableLog('rdApp.*')
        mols_path = osp.join(self.raw_dir, f'{self.stage}_vaild_mols.pt')
        mols = torch.load(mols_path)


        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2}

        data_list = []

        zinc250k_atomic_num_list = [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]

        for i, mol in enumerate(tqdm(mols)):
            x, adj = mol
            N = (x != 0).sum()
            x_ = np.zeros((N, 9), dtype=np.float32)
            indices = x[:N]
            for i in range(N):
                ind = zinc250k_atomic_num_list.index(x[i])
                x_[i, ind] = 1.
            x = torch.tensor(x_).to(torch.float32)

            adj_ = adj.transpose(1, 2, 0)
            adj_ = np.concatenate([1 - np.sum(adj_[..., :3], axis=-1, keepdims=True), adj_[..., :3]], axis=-1)

            # re-order graph
            node_order = np.argsort(indices)
            x = x[node_order]
            adj_[:N, :][:, :N] = adj_[node_order, :][:, node_order]

            adj_ = np.argmax(adj_, -1)
            coo_adj = sp.coo_matrix(adj_)
            edge_index = torch.tensor([coo_adj.row, coo_adj.col], dtype=torch.long)
            edge_type = coo_adj.data
            edge_attr = F.one_hot(torch.tensor(edge_type), num_classes=len(bonds) + 1).to(torch.float)

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_attr = edge_attr[perm]

            y = torch.zeros(size=(1, 0), dtype=torch.float)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=i)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])



class ZINC250KDataModule(MolecularDataModule):
    def __init__(self, cfg):
        self.remove_h = False
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        datasets = {'train': ZINC250KDataset(stage='train', root=root_path),
                    'test': ZINC250KDataset(stage='test', root=root_path),
                    'val': ZINC250KDataset(stage='test', root=root_path)}
        super().__init__(cfg, datasets)




class ZINC250Kinfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg, recompute_statistics=False):
        self.remove_h = cfg.dataset.remove_h
        self.need_to_strip = False        # to indicate whether we need to ignore one output from the model

        self.name = 'zinc250k'
        # 6: C, 7: N, 8: O, 9: F, 15: P, 16: S, 17: Cl, 35: Br, 53: I
        zinc250k_atomic_num_list = [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]

        self.atom_encoder = {'C': 0, 'N': 1, 'O': 2, 'F': 3, 'P': 4, 'S': 5, 'Cl': 6, 'Br': 7, 'I': 8}
        self.atom_decoder = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
        self.num_atom_types = 9
        self.num_edge_types = 4
        # self.valencies = [4, 3, 2, 1]
        # self.atom_weights = {0: 12, 1: 14, 2: 16, 3: 19}
        self.max_n_nodes = 38
        self.max_weight = 150
        self.n_nodes = torch.tensor([0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                                    1.2026e-05, 2.0044e-05, 6.4140e-05, 2.8863e-04, 7.7770e-04, 2.8943e-03,
                                    4.7463e-03, 7.1396e-03, 1.1281e-02, 1.7282e-02, 2.5255e-02, 3.4684e-02,
                                    4.6574e-02, 5.8435e-02, 7.0858e-02, 8.1967e-02, 7.4651e-02, 8.4444e-02,
                                    9.2790e-02, 9.1520e-02, 7.7361e-02, 6.3150e-02, 4.0276e-02, 3.1220e-02,
                                    2.4537e-02, 1.9126e-02, 1.4933e-02, 1.0463e-02, 6.9832e-03, 4.1090e-03,
                                    1.6075e-03, 5.4519e-04, 8.0175e-06])
        self.node_types = torch.tensor([7.3678e-01, 1.2211e-01, 9.9746e-02, 1.3745e-02, 2.4428e-05, 1.7806e-02,
                                        7.4231e-03, 2.2057e-03, 1.5522e-04])
        self.edge_types = torch.tensor([9.0658e-01, 6.9411e-02, 2.3771e-02, 2.3480e-04])

        self.edge_types_new = torch.tensor([0.0666, 0.0228, 0.0002])
        self.degree_counts = datamodule.degree_counts() 
        self.atom_dist = datamodule.atom_distribution()
        
        super().complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
        self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
        self.valency_distribution[0: 7] = torch.tensor([0.0000e+00, 1.1364e-01, 3.0431e-01, 3.5063e-01, 2.2655e-01, 2.2697e-05,
                                                        4.8356e-03])

        if recompute_statistics:
            np.set_printoptions(suppress=True, precision=5)
            self.n_nodes = datamodule.node_counts()
            print("Distribution of number of nodes", self.n_nodes)
            np.savetxt('n_counts.txt', self.n_nodes.numpy())
            self.node_types = datamodule.node_types()                                     # There are no node types
            print("Distribution of node types", self.node_types)
            np.savetxt('atom_types.txt', self.node_types.numpy())

            self.edge_types = datamodule.edge_counts()
            print("Distribution of edge types", self.edge_types)
            np.savetxt('edge_types.txt', self.edge_types.numpy())

            valencies = datamodule.valency_count(self.max_n_nodes)
            print("Distribution of the valencies", valencies)
            np.savetxt('valencies.txt', valencies.numpy())
            self.valency_distribution = valencies
            assert False

def get_train_smiles(cfg, train_dataloader, dataset_infos, train=True, evaluate_dataset=False):
    if evaluate_dataset:
        assert dataset_infos is not None, "If wanting to evaluate dataset, need to pass dataset_infos"
    datadir = cfg.dataset.datadir
    atom_decoder = dataset_infos.atom_decoder
    root_dir = pathlib.Path(os.path.realpath(__file__)).parents[2]
    if train:
        smiles_file_name = 'train_smiles_no_h.npy'
    else:
        smiles_file_name = 'test_smiles_no_h.npy'
    smiles_path = os.path.join(root_dir, datadir, smiles_file_name)
    if os.path.exists(smiles_path):
        print("Dataset smiles were found.")
        train_smiles = np.load(smiles_path)
    else:
        print("Computing dataset smiles...")
        train_smiles = compute_zinc250k_smiles(atom_decoder, train_dataloader)
        np.save(smiles_path, np.array(train_smiles))

    if evaluate_dataset:
        train_dataloader = train_dataloader
        all_molecules = []
        for i, data in enumerate(train_dataloader):
            dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
            dense_data = dense_data.mask(node_mask, collapse=True)
            X, E = dense_data.X, dense_data.E

            for k in range(X.size(0)):
                n = int(torch.sum((X != -1)[k, :]))
                atom_types = X[k, :n].cpu()
                edge_types = E[k, :n, :n].cpu()
                all_molecules.append([atom_types, edge_types])

        print("Evaluating the dataset -- number of molecules to evaluate", len(all_molecules))
        metrics = compute_molecular_metrics(molecule_list=all_molecules, train_smiles=train_smiles,
                                            dataset_info=dataset_infos)
        print(metrics[0])

    return train_smiles

def compute_zinc250k_smiles(atom_decoder, train_dataloader, remove_h=True):
    '''

    :param dataset_name: qm9 or qm9_second_half
    :return:
    '''
    print(f"\tConverting QM9 dataset to SMILES for remove_h={remove_h}...")

    mols_smiles = []
    len_train = len(train_dataloader)
    invalid = 0
    disconnected = 0
    for i, data in enumerate(train_dataloader):
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask, collapse=True)
        X, E = dense_data.X, dense_data.E

        x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1
        X = torch.argmax(X, dim=-1)
        E = torch.argmax(E, dim=-1)

        X[node_mask == 0] = - 1
        E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1

        n_nodes = [int(torch.sum((X != -1)[j, :])) for j in range(X.size(0))]

        molecule_list = []
        for k in range(X.size(0)):
            n = n_nodes[k]
            atom_types = X[k, :n].cpu()
            edge_types = E[k, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        for l, molecule in enumerate(molecule_list):
            mol = build_molecule_with_partial_charges(molecule[0], molecule[1], atom_decoder)
            smile = mol2smiles(mol)
            if smile is not None:
                mols_smiles.append(smile)
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                if len(mol_frags) > 1:
                    print("Disconnected molecule", mol, mol_frags)
                    disconnected += 1
            else:
                print("Invalid molecule obtained.")
                invalid += 1

        if i % 1000 == 0:
            print("\tConverting ZINC250K dataset to SMILES {0:.2%}".format(float(i) / len_train))
    print("Number of invalid molecules", invalid)
    print("Number of disconnected molecules", disconnected)
    return mols_smiles

if __name__ == "__main__":
    ds = [ZINC250KDataset(s, os.path.join(os.path.abspath(__file__), "../../../data/zinc250k"),
                       ) for s in ["train", "val", "test"]]