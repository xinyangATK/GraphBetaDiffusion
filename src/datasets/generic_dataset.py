import os
import pathlib
import pickle
import torch
from torch.utils.data import random_split
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset, download_url

from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
from src.GDSS_utils.utils.graph_utils import graphs_to_tensor, init_features

class GenericGraphDataset(InMemoryDataset):
    def __init__(self, dataset_name, split, root, transform=None, pre_transform=None, pre_filter=None, data_cfg=None):
        self.data_cfg = data_cfg
        raw_data_dir = "https://github.com/harryjo97/GDSS/raw/master/data"
        self.path_mappings = {
            'comm20': "community_small.pkl",
            'ego': "ego_small.pkl",
            'sbm': "sbm.pkl",
            'planar': "planar.pkl"
        }
        self.url_mappings = {
            'omm20': raw_data_dir + "/community_small.pkl",
            'ego': raw_data_dir + "/ego_small.pkl",
            'sbm': raw_data_dir + "/sbm.pkl",
            'planar': raw_data_dir + "/planar.pkl"
        }
        if dataset_name not in self.url_mappings:
            raise ValueError(f"Undefined dataset '{dataset_name}'.")

        self.dataset_name = dataset_name
        if dataset_name == 'comm20':
            self.max_node_num = 20
            self.max_deg_num = 9  # 15, 8, 9, 10, 10
        elif dataset_name == 'ego':
            self.max_node_num = 18
            self.max_deg_num = 17
        elif dataset_name == 'planar':
            self.max_node_num = 64
            self.max_deg_num = 13
        elif dataset_name == 'sbm':
            self.max_node_num = 192
            self.max_deg_num = 24
        else:
            raise ValueError(f"Undefined dataset '{dataset_name}'.")

        self.split = split

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    @property
    def processed_file_names(self):
            return [self.split + '.pt']

    def graph2tensor(self, graph_list):
        adjs_tensor = graphs_to_tensor(graph_list, self.max_node_num)
        x_tensor, feat_dim = init_features(self.data_cfg.feat, adjs_tensor, self.max_deg_num)

        return {'x': x_tensor, 'feat_dim': feat_dim, 'adj': adjs_tensor}

    def download(self):
        if self.dataset_name in ['comm20', 'ego']:
            """ download data from https://github.com/harryjo97/GDSS/tree/master/data """
            # raw_path = download_url(self.url_mappings[self.dataset_name], self.raw_dir)
            if not (os.path.exists(self.raw_paths[0]) and os.path.exists(self.raw_paths[2])):
                raw_path = os.path.join(self.raw_dir, self.path_mappings[self.dataset_name])
                with open(raw_path, 'rb') as f:
                    graph_list = pickle.load(f)

                test_size = int(0.2 * len(graph_list))
                train_graph_list, test_graph_list = graph_list[test_size:], graph_list[:test_size]

                train_graph_tensor = self.graph2tensor(train_graph_list)
                test_graph_tensor = self.graph2tensor(test_graph_list)

                torch.save(train_graph_tensor, self.raw_paths[0])
                torch.save(test_graph_tensor, self.raw_paths[1])
                torch.save(test_graph_tensor, self.raw_paths[2])
        elif self.dataset_name in ['planar', 'sbm']:
            if not (os.path.exists(self.raw_paths[0]) and os.path.exists(self.raw_paths[2])):
                raw_path = os.path.join(self.raw_dir, self.path_mappings[self.dataset_name])
                with open(raw_path, 'rb') as f:
                    train_graph_list, val_graph_list, test_graph_list = pickle.load(f)

                train_graph_tensor = self.graph2tensor(train_graph_list)
                test_graph_tensor = self.graph2tensor(test_graph_list)

                torch.save(train_graph_tensor, self.raw_paths[0])
                torch.save(test_graph_tensor, self.raw_paths[1])
                torch.save(test_graph_tensor, self.raw_paths[2])

        else:
            raise ValueError(f"Undefined dataset '{self.dataset_name}'.")


    def process(self):
        file_idx = {'train': 0, 'val': 1, 'test': 2}
        raw_dataset = torch.load(self.raw_paths[file_idx[self.split]])
        raw_x, raw_adj = raw_dataset['x'], raw_dataset['adj']
        raw_feat_dim = raw_dataset['feat_dim']

        data_list = []
        for (x, adj) in zip(raw_x, raw_adj):
            n = x[..., 1:raw_feat_dim[0]].sum().to(torch.long)

            # GBD change: reorder via node degree
            node_order = torch.arange(self.max_node_num)
            node_order[:n] = torch.argsort(adj[:n, :n].sum(-1), dim=-1, descending=True)
            x = x[node_order]
            adj = adj[node_order, :][:, node_order]
            # X = torch.ones(n, 1, dtype=torch.float32)

            y = torch.zeros([1, 0]).float()
            edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
            edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
            edge_attr[:, 1] = 1
            n_nodes = n * torch.ones(1, dtype=torch.long)
            data = torch_geometric.data.Data(x=x[:n], edge_index=edge_index, edge_attr=edge_attr,
                                             y=y, n_nodes=n_nodes)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])



class GenericGraphDataModule(AbstractDataModule):
    def __init__(self, cfg, n_graphs=200):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)


        datasets = {'train': GenericGraphDataset(dataset_name=self.cfg.dataset.name,
                                                 split='train', root=root_path, data_cfg=cfg.dataset),
                    'val': GenericGraphDataset(dataset_name=self.cfg.dataset.name,
                                        split='test', root=root_path, data_cfg=cfg.dataset),
                    'test': GenericGraphDataset(dataset_name=self.cfg.dataset.name,
                                        split='test', root=root_path, data_cfg=cfg.dataset)}
        # print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')

        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]


class GenericDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = 'nx_graphs'
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = torch.tensor([1])               # There are no node types
        self.edge_types = self.datamodule.edge_counts()
        self.degree_counts = self.datamodule.degree_counts()
        self.node_sparsity = self.datamodule.node_sparsity()
        self.edge_sparsity = self.datamodule.edge_sparsity()
        self.max_deg_num = self.datamodule.train_dataset.max_deg_num
        super().complete_infos(self.n_nodes, self.node_types)
