from src.diffusion.distributions import DistributionNodes
import src.utils as utils
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch_geometric.data.lightning import LightningDataset


class AbstractDataModule(LightningDataset):
    def __init__(self, cfg, datasets):
        super().__init__(train_dataset=datasets['train'], val_dataset=datasets['val'], test_dataset=datasets['test'],
                         batch_size=cfg.train.batch_size if 'debug' not in cfg.general.name else 2,
                         num_workers=cfg.train.num_workers,
                         pin_memory=getattr(cfg.dataset, "pin_memory", False))
        self.cfg = cfg
        self.input_dims = None
        self.output_dims = None

    def __getitem__(self, idx):
        return self.train_dataset[idx]

    def node_counts(self, max_nodes_possible=300):
        all_counts = torch.zeros(max_nodes_possible)
        for loader in [self.train_dataloader(), self.val_dataloader()]:
            for data in loader:
                unique, counts = torch.unique(data.batch, return_counts=True)
                for count in counts:
                    all_counts[count] += 1
        max_index = max(all_counts.nonzero())
        all_counts = all_counts[:max_index + 1]
        all_counts = all_counts / all_counts.sum()
        return all_counts

    def node_types(self):
        num_classes = None
        for data in self.train_dataloader():
            num_classes = data.x.shape[1]
            break

        counts = torch.zeros(num_classes)

        for i, data in enumerate(self.train_dataloader()):
            counts += data.x.sum(dim=0)

        counts = counts / counts.sum()
        return counts

    def edge_counts(self):
        num_classes = None
        for data in self.train_dataloader():
            num_classes = data.edge_attr.shape[1]
            break

        d = torch.zeros(num_classes, dtype=torch.float)

        for i, data in enumerate(self.train_dataloader()):
            unique, counts = torch.unique(data.batch, return_counts=True)

            all_pairs = 0
            for count in counts:
                all_pairs += count * (count - 1)

            num_edges = data.edge_index.shape[1]
            num_non_edges = all_pairs - num_edges

            edge_types = data.edge_attr.sum(dim=0)
            assert num_non_edges >= 0
            d[0] += num_non_edges
            d[1:] += edge_types[1:]

        d = d / d.sum()
        return d

    def degree_counts(self):

        deg = []
        if isinstance(self.cfg.model.max_feat_num, int):
            max_feat_dim = self.cfg.model.max_feat_num
        else:
            max_feat_dim = self.cfg.model.max_feat_num[0]
        for loader in [self.train_dataloader(), self.val_dataloader()]:
            for i, data in enumerate(loader):
                dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch, max_num_nodes=loader.dataset.max_node_num)
                dense_data = dense_data.mask(node_mask)
                X, E = dense_data.X[..., :max_feat_dim], dense_data.E[..., 1:]
                deg += E.sum(dim=(-1, -2)).to(torch.long)

        deg = torch.stack(deg, dim=0)

        return deg

    def atom_distribution(self):
        atom_dist = []
        for loader in [self.train_dataloader(), self.val_dataloader()]:
            for i, data in enumerate(loader):
                dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch, max_num_nodes=loader.dataset.max_node_num)
                dense_data = dense_data.mask(node_mask)
                X, E = dense_data.X, dense_data.E[..., 1:]
                atom_dist += (X.sum(dim=-2)).to(torch.long)

        atom_dist = torch.stack(atom_dist, dim=0)

        return atom_dist
    
    def edge_sparsity(self):
        num_classes = None
        for data in self.train_dataloader():
            num_classes = data.edge_attr.shape[1] - 1
            break

        types = torch.zeros(num_classes, dtype=torch.float)
        nodes = torch.zeros(num_classes, dtype=torch.float)
        for loader in [self.train_dataloader(), self.val_dataloader()]:
            for i, data in enumerate(loader):
                dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch, max_num_nodes=loader.dataset.max_node_num)
                dense_data = dense_data.mask(node_mask)
                X, E = dense_data.X, dense_data.E[..., 1:]
                for g_id in range(X.size(0)):
                    for dim_idx in range(num_classes):
                        nodes[dim_idx] += (node_mask[g_id].sum(-1)**2)
                        types[dim_idx] += E[g_id, ..., dim_idx].sum((-1, -2))
        
        edge_sparsity = types / nodes
        return edge_sparsity
    
    def node_sparsity(self):
        num_classes = None
        for data in self.train_dataloader():
            num_classes = data.x.shape[1] - 3
            break

        types = torch.zeros(num_classes, dtype=torch.float)
        nodes = torch.zeros(num_classes, dtype=torch.float)
        for loader in [self.train_dataloader(), self.val_dataloader()]:
            for i, data in enumerate(loader):
                dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch, max_num_nodes=loader.dataset.max_node_num)
                dense_data.X = dense_data.X[..., 1:-2]
                dense_data = dense_data.mask(node_mask)
                X, E = dense_data.X, dense_data.E[..., 1:]
                for g_id in range(X.size(0)):
                    for dim_idx in range(num_classes):
                        nodes[dim_idx] += node_mask[g_id].sum(-1)
                        types[dim_idx] += X[g_id, ..., dim_idx].sum(-1)
        
        node_sparsity = types / nodes
        return node_sparsity



class MolecularDataModule(AbstractDataModule):
    def valency_count(self, max_n_nodes):
        valencies = torch.zeros(3 * max_n_nodes - 2)   # Max valency possible if everything is connected

        # No bond, single bond, double bond, triple bond, aromatic bond
        multiplier = torch.tensor([0, 1, 2, 3])

        for data in self.train_dataloader():
            n = data.x.shape[0]

            for atom in range(n):
                edges = data.edge_attr[data.edge_index[0] == atom]
                edges_total = edges.sum(dim=0)
                valency = (edges_total * multiplier).sum()
                valencies[valency.long().item()] += 1
        valencies = valencies / valencies.sum()
        return valencies


class AbstractDatasetInfos:
    def complete_infos(self, n_nodes, node_types):
        self.input_dims = None
        self.output_dims = None
        self.num_classes = len(node_types)
        self.max_n_nodes = len(n_nodes) - 1
        self.nodes_dist = DistributionNodes(n_nodes)

    def compute_input_output_dims(self, datamodule, extra_features, domain_features, cfg):
        example_batch = next(iter(datamodule.train_dataloader()))
        ex_dense, node_mask = utils.to_dense(example_batch.x, example_batch.edge_index, example_batch.edge_attr,
                                             example_batch.batch)
        # if cfg.dataset.name in ['sbm', 'planar']:
        if cfg.model.noise_feat_type == 'deg':
            ex_dense.X = ex_dense.X[..., 1:-2]
        elif cfg.model.noise_feat_type == 'eig':
            ex_dense.X = ex_dense.X[..., -2:]
        elif cfg.model.noise_feat_type == 'all':
            ex_dense.X = ex_dense.X[..., 1:]
        else:
            ex_dense.X = torch.ones_like(ex_dense.X)

        dense_data = ex_dense.mask(node_mask)
        example_batch['x'], example_batch['edge_attr'] = dense_data.X, dense_data.E[..., 1:]  # remove 'no edge' and 'Aromatic bond'

        example_data = {'X_t': ex_dense.X, 'E_t': ex_dense.E[..., 1:], 'y_t': example_batch['y'], 'node_mask': node_mask}

        order = 1
        if cfg.model.high_order:
            order = 2
        self.input_dims = {'X': example_batch['x'].size(-1),
                           'E': example_batch['edge_attr'].size(-1),
                           'y': example_batch['y'].size(1) + 1}      # + 1 due to time conditioning

        ex_extra_feat = extra_features(example_data)

        self.input_dims['X'] += ex_extra_feat.X.size(-1)
        self.input_dims['E'] += ex_extra_feat.E.size(-1)
        self.input_dims['y'] += ex_extra_feat.y.size(-1)

        ex_extra_molecular_feat = domain_features(example_data)
        self.input_dims['X'] += ex_extra_molecular_feat.X.size(-1)
        self.input_dims['E'] += ex_extra_molecular_feat.E.size(-1)
        self.input_dims['y'] += ex_extra_molecular_feat.y.size(-1)

        self.input_dims['E'] = self.input_dims['E'] * order

        self.output_dims = {'X': example_batch['x'].size(-1),
                            'E': example_batch['edge_attr'].size(-1),
                            'y': 0}


