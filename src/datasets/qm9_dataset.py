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




class QM9Dataset(InMemoryDataset):

    def __init__(self, stage, root, transform=None, pre_transform=None, pre_filter=None):
        self.stage = stage
        self.data_name = 'QM9'
        if self.stage == 'train':
            self.file_idx = 0
        elif self.stage == 'val':
            self.file_idx = 1
        else:
            self.file_idx = 2

        self.max_node_num = 9
        self.max_feat_num = 15

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])

    @property
    def raw_file_names(self):
        return ['train_qm9.csv', 'val_qm9.csv', 'test_qm9.csv']

    @property
    def split_file_name(self):
        return ['train_qm9.csv', 'val_qm9.csv', 'test_qm9.csv']

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

        test_idx = test_idx['valid_idxs']
        test_idx = [int(i) for i in test_idx]

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

        for i, mol in enumerate(tqdm(mols)):
            x, adj = mol
            indices = np.where(x >= 6, x - 6, 4)
            indices = indices[indices != 4]
            N = len(indices)
            x_ = np.zeros((N, 4))
            x_[np.arange(N), indices] = 1
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



class QM9DataModule(MolecularDataModule):
    def __init__(self, cfg):
        self.remove_h = False
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        datasets = {'train': QM9Dataset(stage='train', root=root_path),
                    'test': QM9Dataset(stage='test', root=root_path),
                    'val': QM9Dataset(stage='test', root=root_path)}
        super().__init__(cfg, datasets)




class QM9infos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg, recompute_statistics=False):
        self.remove_h = cfg.dataset.remove_h
        self.need_to_strip = False        # to indicate whether we need to ignore one output from the model

        self.name = 'qm9'

        self.atom_encoder = {'C': 0, 'N': 1, 'O': 2, 'F': 3}
        self.atom_decoder = ['C', 'N', 'O', 'F']
        self.num_atom_types = 4
        self.num_edge_types = 4
        self.valencies = [4, 3, 2, 1]
        self.atom_weights = {0: 12, 1: 14, 2: 16, 3: 19}
        self.max_n_nodes = 9
        self.max_weight = 150
        self.n_nodes = torch.tensor([0.0000e+00, 2.2407e-05, 3.7345e-05, 6.7222e-05, 2.3154e-04, 9.7098e-04,
                                    4.6159e-03, 2.3879e-02, 1.3667e-01, 8.3351e-01])
        self.node_types = torch.tensor([0.7187, 0.1189, 0.1596, 0.0028])
        self.edge_types = torch.tensor([0.7268, 0.2338, 0.0314, 0.0080])
        self.edge_types_new = torch.tensor([0.2074, 0.0278, 0.0071])
        self.degree_counts = datamodule.degree_counts()
        self.atom_dist = datamodule.atom_distribution()
        # self.node_sparsity = datamodule.node_sparsity()
        # self.edge_sparsity = datamodule.edge_sparsity()


        super().complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
        self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
        self.valency_distribution[0: 6] = torch.tensor([2.6071e-06, 0.163, 0.352, 0.320, 0.16313, 0.00073])

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
        train_smiles = compute_qm9_smiles(atom_decoder, train_dataloader)
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

def compute_qm9_smiles(atom_decoder, train_dataloader, remove_h=True):
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
            print("\tConverting QM9 dataset to SMILES {0:.2%}".format(float(i) / len_train))
    print("Number of invalid molecules", invalid)
    print("Number of disconnected molecules", disconnected)
    return mols_smiles

if __name__ == "__main__":
    ds = [QM9Dataset(s, os.path.join(os.path.abspath(__file__), "../../../data/qm9"),
                       ) for s in ["train", "val", "test"]]