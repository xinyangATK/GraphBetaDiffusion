import math
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import time
import wandb
import os
from contextlib import contextmanager
from tqdm import tqdm
from models.transformer_model import GraphTransformer
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
from src.diffusion import diffusion_utils
from metrics.train_metrics import TrainLossBeta, TrainLoss, TrainLossDiscrete
from metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchKL, NLL
from src import utils
import pickle
from random import sample
from GDSS_utils.utils.mol_utils import *
from GDSS_utils.utils.graph_utils import *
from GDSS_utils.utils.loader import *
from GDSS_utils.evaluation.stats import *
import logging
from model_init import load_ema, load_ema_from_ckpt

EPS = torch.finfo(torch.float32).eps
MIN = torch.finfo(torch.float32).tiny
clamp_min = 0.405465
clamp_max = 4.595119


class myloggger(object):
    def __init__(self, file_path):
        self.file_path = file_path

    def info(self, messages):
        with open(self.file_path, 'a') as f:
            f.writelines(messages)


class BetaDiffusion(pl.LightningModule):
    def __init__(self, cfg, dataset_infos, visualization_tools, extra_features,
                 domain_features, eval_metrics=None):
        super().__init__()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist
        self.degree_counts = dataset_infos.degree_counts

        self.cfg = cfg
        self.name = cfg.general.name
        self.model_dtype = torch.float32
        self.T = cfg.model.diffusion_steps

        if self.cfg.model.noise_feat_type == 'deg':
            self.max_feat_num = cfg.model.max_feat_num[0] - 1  # remove 'degree = 0' 
        elif self.cfg.model.noise_feat_type == 'eig':
            self.max_feat_num = sum(cfg.model.max_feat_num[1:])
        elif self.cfg.model.noise_feat_type == 'all':
            self.max_feat_num = sum(cfg.model.max_feat_num) - 1  # remove 'degree = 0'  
        else:
            self.max_feat_num = 1

        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        self.node_dist = nodes_dist

        self.dataset_info = dataset_infos

        self.train_loss = TrainLossBeta(self.cfg.model.lambda_train)

        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features

        self.model = GraphTransformer(n_layers=cfg.model.n_layers,
                                      input_dims=input_dims,
                                      hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                      hidden_dims=cfg.model.hidden_dims,
                                      output_dims=output_dims,
                                      act_fn_in=nn.ReLU(),
                                      act_fn_out=nn.ReLU(),
                                      high_order=cfg.model.high_order)

        if self.cfg.train.ema_decay > 0:
            self.use_ema = True
        else:
            self.use_ema = False

        if self.use_ema:
            self.model_ema = load_ema(self.model, decay=cfg.train.ema_decay)

        self.save_hyperparameters(ignore=['train_metrics'])
        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.best_val_nll = 1e8
        self.val_counter = 0

        # Hyper-parameters for BetaDIffusion
        self.sigmoid_start = cfg.model.sigmoid_start
        self.sigmoid_end = cfg.model.sigmoid_end
        self.sigmoid_power = cfg.model.sigmoid_power
        self.beta_max = 20
        self.beta_min = 0.1
        self.Scale = {'node': cfg.model.scale_shift.node[0], 'edge':cfg.model.scale_shift.edge[0]}
        self.Shift = {'node': cfg.model.scale_shift.node[1], 'edge':cfg.model.scale_shift.edge[1]}
        self.eta = {'node': torch.tensor(cfg.model.eta.node, dtype=torch.float32),
                    'edge':torch.tensor(cfg.model.eta.edge, dtype=torch.float32)}
        self.input_space = cfg.model.input_space
        self.pre_condition = cfg.model.pre_condition
        self.eval_metrics = eval_metrics

        self.noise_feat_type = self.cfg.model.noise_feat_type

        self.prob_X = torch.tensor([0.0000, 0.0393, 0.2376, 0.2376, 0.2192, 0.1394, 0.0910, 0.0340, 0.0020])
        # self.prob_X = torch.tensor([0.0000, 0.1098, 0.2259, 0.2310, 0.1961, 0.1332, 0.0901, 0.0133, 0.0006])  # 0.6
        # self.prob_X = torch.tensor([0.0604, 0.1604, 0.3026, 0.2006, 0.1162, 0.0961, 0.0532, 0.0104])  # 0.5
        self.prob_E = torch.tensor([0.2914])  # 0.2683 0.23, 0.20

        self.log2file = myloggger(os.path.join(os.getcwd(), f'res.txt'))

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    def calculate_coefs(self, t):
        t_norm = t
        s_norm = t_norm * 0.95

        logit_alpha_t = self.sigmoid_start + (self.sigmoid_end - self.sigmoid_start) * (
                t_norm ** self.sigmoid_power)
        logit_alpha_s = self.sigmoid_start + (self.sigmoid_end - self.sigmoid_start) * (
                s_norm ** self.sigmoid_power)

        alpha_t, alpha_s = torch.sigmoid(logit_alpha_t), torch.sigmoid(logit_alpha_s)
        delta = (logit_alpha_s.to(torch.float64).sigmoid() - logit_alpha_t.to(torch.float64).sigmoid()).to(
            torch.float32)

        coefs = {'alpha_t': alpha_t, 'alpha_s': alpha_s, 'delta': delta}

        return coefs

    def mask4train(self, X, E, pred_X, pred_E, node_mask):
        x_mask = node_mask.unsqueeze(-1)
        e_mask1 = x_mask.unsqueeze(2)
        e_mask2 = x_mask.unsqueeze(1)
        e_mask = e_mask1 * e_mask2
        diag_mask = torch.eye(X.size(1)).unsqueeze(0).expand(X.size(0), -1, -1)

        # MASK
        X = X * x_mask
        E = E * e_mask
        E[diag_mask.bool()] = 0.
        pred_X = pred_X * x_mask
        pred_E = pred_E * e_mask
        pred_E[diag_mask.bool()] = 0.

        return X, E, pred_X, pred_E

    def get_beta_stats(self, bounds, eta, alpha_t, probs=None):
        """
        Calculate the mean and std of z_t ~ Beta(eta * alpha_t * z_0, eta * (1 - alpha_t * z_0))
        z_0 follows a categorical distribution, defined by the values of the probability of taking each value
        """

        mean_z0 = (bounds * probs).sum()
        var_z0 = (bounds ** 2. * probs).sum() - mean_z0 ** 2.

        mean_zt = alpha_t * mean_z0
        var_zt = (mean_z0 * (1 - mean_z0) - (alpha_t ** 2. * var_z0)) / (eta + 1.) + (alpha_t ** 2. * var_z0)

        return mean_zt, torch.sqrt(var_zt)

    def get_logit_beta_stats(self, bounds, eta, alpha_t, probs=None):
        """
        Calculate the mean and std of logit(z_t), where z_t ~ Beta(eta * alpha_t * z_0, eta * (1 - alpha_t * z_0))
        z_0 follows a categorical distribution, defined by the values of the probability of taking each value
        """
        # bs, 1, 1
        a0 = eta * alpha_t * bounds[0]
        b0 = eta - a0
        a1 = eta * alpha_t * bounds[1]
        b1 = eta - a1

        # all are (bs, 1, 1)
        mean_dg_a = probs[0] * torch.digamma(a0) + probs[1] * torch.digamma(a1)
        mean_dg_b = probs[0] * torch.digamma(b0) + probs[1] * torch.digamma(b1)
        mean_tg_a = probs[0] * torch.polygamma(1, a0) + probs[1] * torch.polygamma(1, a1)
        mean_tg_b = probs[0] * torch.polygamma(1, b0) + probs[1] * torch.polygamma(1, b1)
        mean_dgsq_a = probs[0] * torch.digamma(a0) ** 2. + probs[1] * torch.digamma(a1) ** 2.
        mean_dgsq_b = probs[0] * torch.digamma(b0) ** 2. + probs[1] * torch.digamma(b1) ** 2.

        var_dg_a = F.relu(mean_dgsq_a - mean_dg_a ** 2.)  # (bs, 1, 1)
        var_dg_b = F.relu(mean_dgsq_b - mean_dg_b ** 2.)  # (bs, 1, 1)

        mean_zt = mean_dg_a - mean_dg_b
        var_zt = mean_tg_a + mean_tg_b + var_dg_a + var_dg_b

        return mean_zt, torch.sqrt(var_zt)

    def get_logit_beta_stats_con(self, bounds, eta, alpha_t):
        """
        Calculate the mean and std of logit(z_t), where z_t ~ Beta(eta * alpha_t * z_0, eta * (1 - alpha_t * z_0))
        z_0 follows a categorical distribution, defined by the values of the probability of taking each value
        """
        xmin = bounds[0]
        xmax = bounds[1]
        bs = eta.size(0)
        alpha_t = alpha_t.expand(-1, eta.size(1), -1)
        alpha_t = alpha_t.contiguous().view(-1, 1)
        eta = eta.contiguous().view(-1, 1)

        E1 = 1.0 / (eta * alpha_t * xmin) * (
                (eta * alpha_t * xmax).lgamma() - (eta * alpha_t * xmin).lgamma())
        E2 = 1.0 / (eta * alpha_t * xmin) * (
                (eta - eta * alpha_t * xmin).lgamma() - (eta - eta * alpha_t * xmax).lgamma())

        E_logit_x_t = E1 - E2

        V1 = 1.0 / (eta * alpha_t * xmin) * (
                (eta * alpha_t * xmax).digamma() - (eta * alpha_t * xmin).digamma())
        V2 = 1.0 / (eta * alpha_t * xmin) * (
                (eta - eta * alpha_t * xmin).digamma() - (eta - eta * alpha_t * xmax).digamma())

        grids = self.scale_shift(torch.arange(0, 101, device=self.device) / 100, type='node')
        alpha_t = alpha_t * grids.unsqueeze(0)

        V3 = ((eta * alpha_t).digamma()) ** 2
        V3[:, 0] = (V3[:, 0] + V3[:, -1]) / 2
        V3 = V3[:, :-1]
        V3 = (V3.mean(dim=1).unsqueeze(1) - E1 ** 2).clamp(0)

        V4 = ((eta - eta * alpha_t).digamma()) ** 2
        V4[:, 0] = (V4[:, 0] + V4[:, -1]) / 2
        V4 = V4[:, :-1]
        V4 = (V4.mean(dim=1).unsqueeze(1) - E2 ** 2).clamp(0)

        std_logit_x_t = (V1 + V2 + V3 + V4).sqrt()
        E_logit_x_t = E_logit_x_t.contiguous().view(bs, -1, 1)
        std_logit_x_t = std_logit_x_t.contiguous().view(bs, -1, 1)
        return E_logit_x_t, std_logit_x_t

    def pre_condition_fn(self, alpha_t, eta, type='node', prior_prob=None, input_space='logit'):

        bounds = torch.tensor([self.Shift[type], self.Scale[type] + self.Shift[type]]).to(self.device)

        if prior_prob is None:
            mean_t, std_t = self.get_logit_beta_stats_con(bounds=bounds, eta=eta, alpha_t=alpha_t)
            return mean_t, std_t

        if isinstance(prior_prob, torch.Tensor):
            # prob = (prior_prob, 1 - prior_prob)
            mean_t_list, std_t_list = [], []
            for neg_prob, prob in zip(1 - prior_prob, prior_prob):
                if input_space == 'logit':
                    mean_t, std_t = self.get_logit_beta_stats(bounds=bounds, eta=eta, alpha_t=alpha_t,
                                                              probs=(neg_prob, prob))
                else:
                    mean_t, std_t = self.get_beta_stats(bounds=bounds, eta=eta, alpha_t=alpha_t, probs=(neg_prob, prob))
                mean_t_list.append(mean_t)
                std_t_list.append(std_t)
            mean_t = torch.concat(mean_t_list, dim=-1)
            std_t = torch.concat(std_t_list, dim=-1)
        else:
            probs = torch.tensor([1 - prior_prob, prior_prob]).to(self.device)

            if input_space == 'logit':
                mean_t, std_t = self.get_logit_beta_stats(bounds=bounds, eta=eta, alpha_t=alpha_t, probs=probs)
            else:
                mean_t, std_t = self.get_beta_stats(bounds=bounds, eta=eta, alpha_t=alpha_t, probs=probs)

        return mean_t, std_t

    def get_eta4x(self, E, eta, deg=None):
        if deg is None:
            deg = E.sum((-1, -2))

        deg = deg.to(self.device)
        norm_deg = deg / torch.max(deg, dim=-1, keepdim=True)[0]
        # norm_deg = F.normalize(deg, p=1, dim=-1)
        x_eta = eta[-1] * torch.ones_like(deg)

        threshold_norm_deg = [1., 0.8, 0.4]  # torch.tensor([1., 0.8, 0.4])
        
        mask = (norm_deg >= threshold_norm_deg[1]) & (norm_deg <= threshold_norm_deg[0])
        x_eta[mask] = eta[0]
        
        mask = (norm_deg >= threshold_norm_deg[2]) & (norm_deg < threshold_norm_deg[1])
        x_eta[mask] = eta[1]
        
        mask = norm_deg < threshold_norm_deg[2]
        x_eta[mask] = eta[2]

        return x_eta

    def get_eta4e(self, E, X, eta, eta_x):
        e_eta = eta[-1] * torch.ones_like(eta_x.expand(-1, -1, X.size(1)))

        return e_eta

    def check_data(self, data):
        if self.cfg.model.noise_feat_type == 'deg':
            data = data[..., 1:-2]
        elif self.cfg.model.noise_feat_type == 'eig':
            data = data[..., -2:]
        elif self.cfg.model.noise_feat_type == 'all':
            data = data[..., 1:]
        else:
            data = torch.ones_like(data)

        return data

    def min_max_scaler(self, x, default_min_max=None):
        if default_min_max is None:
            min_ = torch.min(x, dim=1, keepdim=True)[0]
            max_ = torch.max(x, dim=1, keepdim=True)[0]
        else:
            min_, max_ = default_min_max
        norm_x = (x - min_) / (max_ - min_)
        return norm_x

    def training_step(self, data, i):
        if data.edge_index.numel() == 0:
            self.print("Found a batch with no edges. Skipping.")
            return
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data.X = self.check_data(dense_data.X)
        dense_data = dense_data.mask(node_mask)

        X, E = dense_data.X, dense_data.E[..., 1:]  # remove 'no edge' and 'Aromatic bond'
        if self.cfg.model.noise_feat_type == 'eig':
            for dim in range(2):
                X[..., dim] = self.min_max_scaler(X[..., dim], default_min_max=self.cfg.model.default_min_max)
        
        # Sort nodes by degree, descending
        # if self.cfg.re_order:
        #     node_idx = torch.argsort(E.sum((-1, -2)), dim=1, descending=True)
        #     X, E, node_mask = self.order_graph(X, E, node_mask, node_idx)

        # allpy different eta on edge depends on ordered nodes
        eta_x = self.get_eta4x(E, self.eta['node'])
        eta_x = eta_x.unsqueeze(-1)
        eta_e = self.get_eta4e(E, X, eta=self.eta['edge'], eta_x=eta_x)
        eta_e = eta_e.unsqueeze(-1)
        eta_pair = (eta_x, eta_e)

        if self.cfg.dataset.rand_perm:
            X, E, node_mask, eta_pair = rand_perm_mol(X, E, node_mask, eta_pair)

        X = self.scale_shift(X, type='node')
        E = self.scale_shift(E, type='edge')

        # Follow the steps in Beta Diffusion.
        # noisy_data in original/logit input_space
        noisy_data = self.apply_noise(X, E, data.y, node_mask, etas=eta_pair)

        extra_data = self.compute_extra_data(noisy_data)
        # extra_data.X = torch.cat((extra_feat, extra_data.X), dim=2).float()

        if self.pre_condition:
            if self.cfg.model.noise_feat_type == 'deg':
                mean_logit_X_t, std_logit_X_t = self.pre_condition_fn(noisy_data['alpha_t'], eta_pair[0], type='node', prior_prob=self.prob_X)
                noisy_data['X_t'] = (noisy_data['X_t'] - mean_logit_X_t) / std_logit_X_t
            elif self.cfg.model.noise_feat_type == 'eig':
                mean_logit_X_t, std_logit_X_t = self.pre_condition_fn(noisy_data['alpha_t'], eta_pair[0], type='node')
                noisy_data['X_t'] = (noisy_data['X_t'] - mean_logit_X_t) / std_logit_X_t
            elif self.cfg.model.noise_feat_type == 'all':
                mean_logit_X_deg_t, std_logit_X_deg_t = self.pre_condition_fn(noisy_data['alpha_t'], eta_pair[0], type='node', prior_prob=self.prob_X)
                mean_logit_X_eig_t, std_logit_X_eig_t = self.pre_condition_fn(noisy_data['alpha_t'], eta_pair[0], type='node')
                noisy_data['X_t'][..., :-2] = (noisy_data['X_t'][..., :-2] - mean_logit_X_deg_t) / std_logit_X_deg_t
                noisy_data['X_t'][..., -2:] = (noisy_data['X_t'][..., -2:] - mean_logit_X_eig_t) / std_logit_X_eig_t
            else:
                pass

            mean_logit_E_t, std_logit_E_t = self.pre_condition_fn(noisy_data['alpha_t'].unsqueeze(-1), eta_pair[1], type='edge', prior_prob=self.prob_E)
            noisy_data['E_t'] = (noisy_data['E_t'] - mean_logit_E_t) / std_logit_E_t

            noisy_data['X_t'], noisy_data['E_t'] = self.mask_and_sym(noisy_data['X_t'], noisy_data['E_t'], node_mask)

        assert torch.allclose(noisy_data['E_t'], torch.transpose(noisy_data['E_t'], 1, 2))
        pred = self.forward(noisy_data, extra_data, node_mask)

        coefs = self.calculate_coefs(noisy_data['t'])

        pred_X = self.scale_shift(torch.sigmoid(pred.X), type='node')
        pred_E = self.scale_shift(torch.sigmoid(pred.E), type='edge')

        # MASK FOR TRAINING
        # groundtruth without scale-shift
        X, E, pred_X, pred_E = self.mask4train(X, E, pred_X, pred_E, node_mask)

        loss = self.train_loss(masked_pred_X=pred_X, masked_pred_E=pred_E, pred_y=pred.y,
                               true_X=X, true_E=E, true_y=data.y, coefs=coefs,
                               log=i % self.log_every_steps == 0)

        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, amsgrad=True,
                                 weight_decay=self.cfg.train.weight_decay)

    def load_graph_list(self, dataset):
        if dataset in ['gdss-comm20', 'gdss-ego']:
            pkl_name = 'community_small' if dataset == 'gdss-comm20' else 'ego_small'
            with open(f'../../../../data/{dataset}/raw/{pkl_name}.pkl', 'rb') as f:
                graph_list = pickle.load(f)
            test_size = int(0.2 * len(graph_list))
            self.train_graph_list, self.test_graph_list = graph_list[test_size:], graph_list[:test_size]
        elif dataset in ['sbm', 'planar']:
            with open(f'../../../../data/{dataset}/raw/{dataset}.pkl', 'rb') as f:
                train_graph_list, val_graph_list, test_graph_list = pickle.load(f)
            self.train_graph_list, self.test_graph_list = train_graph_list, test_graph_list
        else:
            raise ValueError('wrong dataset name while loading')

    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        self.print("Size of the input features", self.Xdim, self.Edim, self.ydim)
        # LOAD GRAPH AND SMILES FOR TEST
        print('Load smiles or graph list...')
        self.load_graph_list(self.cfg.dataset.name)

    def on_train_epoch_start(self) -> None:
        self.print("Starting train epoch...")
        self.start_epoch_time = time.time()
        self.train_loss.reset()

    def on_train_epoch_end(self) -> None:
        to_log = self.train_loss.log_epoch_metrics()
        self.print(f"Epoch {self.current_epoch}: X_KLUB: {to_log['train_epoch/x_KLUB'] :.3f}"
                   f" -- E_KLUB: {to_log['train_epoch/E_KLUB'] :.3f} --"
                   f" y_CE: {to_log['train_epoch/y_CE'] :.3f}"
                   f" -- {time.time() - self.start_epoch_time:.1f}s ")

        # Update EMA
        if self.use_ema:
            self.model_ema(self.model)

    def on_validation_epoch_start(self) -> None:
        pass

    def validation_step(self, data, i):
        return {'loss': None}

    def on_validation_epoch_end(self) -> None:
        self.print(f"Validation on Epoch {self.current_epoch}...")
        self.val_counter += 1

        if (self.val_counter != 1) and (self.val_counter % (self.cfg.general.sample_every_val)) == 0:
            self.log2file.info(f"Validation on Epoch {self.current_epoch}...\n")
            # Save model_ema model
            ckpt_dir = os.path.join(os.getcwd(), f'checkpoints/{self.cfg.general.name}')
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            torch.save(self.model_ema, os.path.join(ckpt_dir, f'{self.current_epoch}_ema.pth'))

            start = time.time()
            samples_left_to_generate = self.cfg.general.samples_to_generate
            samples_left_to_save = self.cfg.general.samples_to_save

            Xs, Es = [], []
            ident = 0
            while samples_left_to_generate > 0:
                bs = 200  # self.cfg.general.samples_to_generate // 5  # 2 * self.cfg.train.batch_size
                to_generate = min(samples_left_to_generate, bs)
                to_save = min(samples_left_to_save, bs)

                sample_deg, sample_extra_feat, nodes_num = None, None, None
                if self.cfg.model.eta_from == 'train':
                    sample_deg, nodes_num = self.sample_from_train(to_generate, re_deg=True, re_feat=False)

                with self.ema_scope('Sampling'):
                    graph = self.sample_batch(batch_id=ident, batch_size=to_generate, num_nodes=nodes_num,
                                              save_final=to_save, extra_feat=sample_extra_feat, eta_from=sample_deg,
                                              return_E=True)
                E, n_nodes = graph
                for (e, n_node) in zip(E, n_nodes):
                    Es.append(e[:n_node, :n_node])

                ident += to_generate

                samples_left_to_save -= to_save
                samples_left_to_generate -= to_generate

            self.print("Generated graphs Saved. Computing sampling metrics...")
            gen_graph_list = adjs_to_graphs(Es, True)

            test_graph_num = len(self.test_graph_list)
            test_num = len(gen_graph_list) // test_graph_num

            methods, kernels = load_eval_settings(self.cfg.dataset.name)
            score_all_batch = {}

            average = []
            for k in range(test_num):
                print('Compute score...')
                gen_graph_list_one_batch = gen_graph_list[
                                           k * len(self.test_graph_list): (k + 1) * len(self.test_graph_list)]
                score = eval_graph_list(self.test_graph_list, gen_graph_list_one_batch, methods=methods,
                                        kernels=kernels)
                for key in score.keys():
                    if key in score_all_batch.keys():
                        score_all_batch[key].append(score[key])
                    else:
                        score_all_batch[key] = [score[key]]

            self.log2file.info(f'Sampling after {self.current_epoch} epochs.\n')
            for key in score_all_batch.keys():
                mean = np.mean(score_all_batch[key])
                std = np.std(score_all_batch[key])
                max_ = np.max(score_all_batch[key])
                min_ = np.min(score_all_batch[key])

                self.log2file.info(f'{key}: mean = {mean}    std = {std}    max = {max_}    min = {min_}\n')

            self.log2file.info('=' * 100 + '\n\n')
            print("Validation epoch end ends...")

    def on_test_epoch_start(self) -> None:
        self.print("Starting test...")
        if not hasattr(self, 'test_graph_list'):
            # LOAD GRAPH AND SMILES FOR TEST
            print('Load smiles or graph list...')
            self.load_graph_list(self.cfg.dataset.name)

            # Load EMA
            if self.use_ema:
                ckpt_dir = self.cfg.general.test_only.split('epoch')[0]
                ckpt_name = self.cfg.general.test_only.split('/')[-1].split('=')[1].split('-')[0]
                ema_ckpt_path = os.path.join(ckpt_dir, f'{ckpt_name}_ema.pth')
                self.model_ema = torch.load(ema_ckpt_path)

    def test_step(self, data, i):
        return {'loss': None}

    def on_test_epoch_end(self) -> None:
        """ Measure likelihood on a test set and compute stability metrics. """
        if self.cfg.general.sample_visualization:
            self.sample_visualization(to_sample=self.cfg.general.vis_number, given_t_split=10)
            print('Visilization done.')
            return

        self.print(f"Test on Epoch {self.current_epoch}...")

        samples_left_to_generate = self.cfg.general.final_model_samples_to_generate
        samples_left_to_save = self.cfg.general.final_model_samples_to_save

        Xs, Es = [], []
        ident = 0
        while samples_left_to_generate > 0:
            bs = 300  # 3096  # self.cfg.general.samples_to_generate // 5  # 2 * self.cfg.train.batch_size
            to_generate = min(samples_left_to_generate, bs)
            to_save = min(samples_left_to_save, bs)

            sample_deg, sample_extra_feat, nodes_num = None, None, None
            if self.cfg.model.eta_from == 'train':
                sample_deg, nodes_num = self.sample_from_train(to_generate, re_deg=True, re_feat=False)

            # with self.ema_scope('Sampling'):
            graph = self.sample_batch(batch_id=ident, batch_size=to_generate, num_nodes=nodes_num,
                                      save_final=to_save, extra_feat=sample_extra_feat, eta_from=sample_deg,
                                      return_E=True)
            E, n_nodes = graph
            for (e, n_node) in zip(E, n_nodes):
                Es.append(e[:n_node, :n_node])

            ident += to_generate

            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate

        self.print("Generated graphs Saved. Computing sampling metrics...")
        gen_graph_list = adjs_to_graphs(Es, True)

        test_graph_num = len(self.test_graph_list)
        test_num = len(gen_graph_list) // test_graph_num

        methods, kernels = load_eval_settings(self.cfg.dataset.name, kernel='tv')
        score_all_batch = {}
        # for key in methods:
        #     score_all_batch[key] = []
        # score_all_batch = {'degree': [], 'cluster': [], 'orbit': [], 'spectral': [], 'connected': [], 'V.U.N.': []}
        ckpt_dir = self.cfg.general.test_only.split('checkpoints')[0]
        ckpt_name = self.cfg.general.test_only.split('/')[-1].split('=')[1].split('-')[0]
        graph_ckpt_path = os.path.join(ckpt_dir, f'{ckpt_name}.pkl')
        with open(graph_ckpt_path, 'wb') as f:
            pickle.dump(obj=gen_graph_list, file=f, protocol=pickle.HIGHEST_PROTOCOL)

        # plot_graphs_list(graphs=sample_graph_list, title=save_name, max_num=16, save_dir=self.log_folder_name)
        average = []
        for k in range(test_num):
            print('Compute score...')
            gen_graph_list_one_batch = gen_graph_list[
                                       k * test_graph_num: (k + 1) * test_graph_num]
            score = eval_graph_list(self.test_graph_list, gen_graph_list_one_batch, methods=methods,
                                    kernels=kernels)
            for key in score.keys():
                if key in score_all_batch.keys():
                    score_all_batch[key].append(score[key])
                else:
                    score_all_batch[key] = [score[key]]

        self.log2file.info(f'Sampling after {self.current_epoch} epochs.\n')
        for key in score_all_batch.keys():
            mean = np.mean(score_all_batch[key])
            std = np.std(score_all_batch[key])
            max_ = np.max(score_all_batch[key])
            min_ = np.min(score_all_batch[key])

            self.log2file.info(f'{key}: mean = {mean}    std = {std}    max = {max_}    min = {min_}\n')

        self.log2file.info('=' * 100 + '\n\n')
        print("Done testing")

    def log_gamma(self, alpha):
        return torch.log(torch._standard_gamma(alpha.to(torch.float32)).clamp(MIN))

    def gamma(self, alpha):
        return torch._standard_gamma(alpha.to(torch.float32)).clamp(MIN)

    def apply_noise(self, X, E, y, node_mask, given_t_split=None, etas=None):
        """ Sample noise and apply it to the data. """
        eta_x, eta_e = etas[0], etas[1]
        # Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately
        # lowest_t = 0 if self.training else 1
        if given_t_split is None:
            t_float = torch.rand([X.size(0), 1], device=X.device).float() * (1e-5 - 1.) + 1
        else:  # for visualization
            t_float = torch.linspace(1e-5, 1, given_t_split).view(-1, 1)
            t_float = t_float.repeat(X.size(0), 1).reshape(-1, 1)  # [num_t * X.size(0), 1]
            X = torch.concat([x.repeat(given_t_split, 1, 1) for x in X], dim=0)
            E = torch.concat([e.repeat(given_t_split, 1, 1, 1) for e in E], dim=0)
            node_mask = torch.concat([nm.repeat(given_t_split, 1) for nm in node_mask], dim=0)
            etas = torch.concat([eta.repeat(given_t_split, 1) for eta in etas], dim=0)
            etas = etas.unsqueeze(-1)

        logit_alpha_t = self.sigmoid_start + (self.sigmoid_end - self.sigmoid_start) * (t_float ** self.sigmoid_power)
        alpha_t = torch.sigmoid(logit_alpha_t).unsqueeze(2)

        if etas is None:
            eta_x = eta_e = self.eta

        log_u = self.log_gamma((eta_x * alpha_t * X).to(torch.float32))
        log_v = self.log_gamma((eta_x - eta_x * alpha_t * X).to(torch.float32))
        logit_x_t = log_u - log_v

        log_u = self.log_gamma(((eta_e * alpha_t.unsqueeze(-1)) * E).to(torch.float32))
        log_v = self.log_gamma((eta_e - (eta_e * alpha_t.unsqueeze(-1)) * E).to(torch.float32))
        logit_e_t = log_u - log_v

        if self.input_space == 'logit':
            X = logit_x_t
            E = logit_e_t
        else:
            # X and E (in original data input_space)
            X = logit_x_t.sigmoid()
            E = logit_e_t.sigmoid()

        X, E = self.mask_and_sym(X, E, node_mask)

        z_t = utils.PlaceHolder(X=X, E=E, y=y).type_as(X).mask(node_mask)

        noisy_data = {'t': t_float, 'alpha_t': alpha_t, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y,
                      'node_mask': node_mask}

        return noisy_data

    def forward(self, noisy_data, extra_data, node_mask):
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
        return self.model(X, E, y, node_mask)

    def mask_and_sym(self, X, E, node_mask, mask=True, diag=True, sym=True):
        if mask:
            bs, n, _ = X.shape
            # Noise X
            # The masked rows should define probability distributions with zero
            X[~node_mask] = 0.

            # Noise E
            # The masked rows should define probability distributions as well
            inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
            diag_mask = torch.eye(n).unsqueeze(0).expand(bs, -1, -1)

            # The masked position should define probability distributions with zero
            E[inverse_edge_mask] = 0.
            if diag:
                E[diag_mask.bool()] = 0.

        if sym:
            upper_triangular_mask = torch.zeros_like(E)
            indices = torch.triu_indices(row=E.size(1), col=E.size(2), offset=1)
            upper_triangular_mask[:, indices[0], indices[1], :] = 1

            E = E * upper_triangular_mask
            E = (E + torch.transpose(E, 1, 2))

        return X, E

    def scale_shift(self, input, type='node'):

        new_input = self.Scale[type] * input + self.Shift[type]

        return new_input

    def shift_scale(self, input, type='node'):
        new_input = (input - self.Shift[type]) / self.Scale[type]
        return new_input

    @torch.no_grad()
    def sample_batch(self, batch_id: int = 0, batch_size: int = 32,
                     save_final: int = 10, alpha_T=None,
                     num_nodes=None, extra_feat=None, eta_from=None, return_E=False, num_chain_step=None,
                     keep_chain=None):
        """
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(batch_size, device=self.device, dtype=torch.int)
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_max = self.dataset_info.datamodule.test_dataset.max_node_num  # torch.max(n_nodes).item()
        # Build the masks
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)

        # Build noise schedule
        sample_steps = 1 - torch.arange(self.T) / (self.T - 1) * (1. - 1e-5)
        sample_steps = torch.cat([sample_steps, torch.zeros(1)]).to(self.device)

        logit_alpha = self.sigmoid_start + (self.sigmoid_end - self.sigmoid_start) * (
                    sample_steps ** self.sigmoid_power)
        alpha = torch.sigmoid(logit_alpha)

        alpha_T = alpha_T if alpha_T is not None else alpha[0]

        N = node_mask.size(1)
        # Sample noise  -- x_next has size (n_samples, n_nodes, n_features)
        # Initiate E_T adn X_T
        # y = torch.empty((batch_size, 0))
        y = torch.zeros((batch_size, 0)).to(self.device)

        # TODO: replace 0.5 with data.mean()
        if self.noise_feat_type is None:
            max_feat_num = 1
        else:
            max_feat_num = self.max_feat_num

        X_s = 0.01 * torch.ones(batch_size, N, max_feat_num, device=self.device)
        E_s = 0.01 * torch.ones(batch_size, N, N, 1, device=self.device)

        eta_x = self.get_eta4x(E_s, eta=self.eta['node'], deg=eta_from)
        eta_x = eta_x.unsqueeze(-1)
        eta_e = self.get_eta4e(E_s, X_s, eta=self.eta['edge'], eta_x=eta_x)
        eta_e = eta_e.unsqueeze(-1)
        eta_pair = (eta_x, eta_e)

        X_s = self.scale_shift(X_s, type='node')
        E_s = self.scale_shift(E_s, type='edge')

        log_u_x = self.log_gamma((eta_x * alpha_T * X_s).to(torch.float32)).to(torch.float64)
        log_v_x = self.log_gamma((eta_x - eta_x * alpha_T * X_s).to(torch.float32)).to(torch.float64)
        logit_X_s = log_u_x - log_v_x

        log_u_e = self.log_gamma((eta_e * alpha_T * E_s).to(torch.float32)).to(torch.float64)
        log_v_e = self.log_gamma((eta_e - eta_e * alpha_T * E_s).to(torch.float32)).to(torch.float64)
        logit_E_s = log_u_e - log_v_e

        if self.input_space == 'logit':
            X_s = logit_X_s
            E_s = logit_E_s
        else:
            X_s = F.logsigmoid(-logit_X_s)
            E_s = F.logsigmoid(-logit_E_s)

        if num_chain_step is not None:
            assert num_chain_step < self.T
            chain_X_size = torch.Size((num_chain_step, keep_chain, X_s.size(-2), X_s.size(-1)))
            chain_E_size = torch.Size((num_chain_step, keep_chain, E_s.size(-3), E_s.size(-2), E_s.size(-1)))

            chain_X = torch.zeros(chain_X_size)
            chain_E = torch.zeros(chain_E_size)

        for i, (logit_alpha_t, logit_alpha_s) in enumerate(
                tqdm(zip(logit_alpha[:-1], logit_alpha[1:]))):  # 0, ..., self.T - 1
            alpha_t, alpha_s = torch.sigmoid(logit_alpha_t), torch.sigmoid(logit_alpha_s)

            # z_hat = f(z_t, t)
            t = sample_steps[i].unsqueeze(-1).repeat(batch_size, 1)

            with torch.no_grad():
                # Neural net predictions
                noisy_data = {'X_t': X_s, 'E_t': E_s, 'y_t': y, 't': t,
                              'alpha_t': alpha_t.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1),
                              'node_mask': node_mask}

                if self.input_space == 'original':  # not in logit input space
                    noisy_data['X_t'] = - X_s.expm1()
                    noisy_data['E_t'] = - E_s.expm1()

                extra_data = self.compute_extra_data(noisy_data)

                if self.pre_condition:
                    if self.cfg.model.noise_feat_type == 'deg':
                        mean_logit_X_t, std_logit_X_t = self.pre_condition_fn(noisy_data['alpha_t'], eta_pair[0], type='node', prior_prob=self.prob_X)
                        noisy_data['X_t'] = (noisy_data['X_t'] - mean_logit_X_t) / std_logit_X_t
                    elif self.cfg.model.noise_feat_type == 'eig':
                        mean_logit_X_t, std_logit_X_t = self.pre_condition_fn(noisy_data['alpha_t'], eta_pair[0], type='node')
                        noisy_data['X_t'] = (noisy_data['X_t'] - mean_logit_X_t) / std_logit_X_t
                    elif self.cfg.model.noise_feat_type == 'all':
                        mean_logit_X_deg_t, std_logit_X_deg_t = self.pre_condition_fn(noisy_data['alpha_t'], eta_pair[0], type='node', prior_prob=self.prob_X)
                        mean_logit_X_eig_t, std_logit_X_eig_t = self.pre_condition_fn(noisy_data['alpha_t'], eta_pair[0], type='node')
                        noisy_data['X_t'][..., :-2] = (noisy_data['X_t'][..., :-2] - mean_logit_X_deg_t) / std_logit_X_deg_t
                        noisy_data['X_t'][..., -2:] = (noisy_data['X_t'][..., -2:] - mean_logit_X_eig_t) / std_logit_X_eig_t
                    else:
                        pass

                    mean_logit_E_t, std_logit_E_t = self.pre_condition_fn(noisy_data['alpha_t'].unsqueeze(-1), eta_e, type='edge', prior_prob=self.prob_E)
                    noisy_data['E_t'] = (noisy_data['E_t'] - mean_logit_E_t) / std_logit_E_t

                    noisy_data['X_t'], noisy_data['E_t'] = self.mask_and_sym(noisy_data['X_t'], noisy_data['E_t'],
                                                                             node_mask)

                pred = self.forward(noisy_data, extra_data, node_mask)

            X = self.scale_shift(torch.sigmoid(pred.X), type='node')
            E = self.scale_shift(torch.sigmoid(pred.E), type='edge')

            if self.input_space == 'logit':
                #  new logit_X and logit_E
                X_s = self.sample_new_logit_x(X_s, X, alpha_t, alpha_s, eta_x)
                E_s = self.sample_new_logit_x(E_s, E, alpha_t, alpha_s, eta_e)
            else:
                #  new X and E
                X_s = self.sample_new_x(X_s, X, alpha_t, alpha_s, eta_e)
                E_s = self.sample_new_x(E_s, E, alpha_t, alpha_s, eta_e)

            if num_chain_step is not None:
                # Save the first keep_chain graphs
                chain_index = int((num_chain_step * (self.T - i)) / self.T - 1)
                chain_X[chain_index] = self.shift_scale(X[:keep_chain])
                chain_E[chain_index] = self.shift_scale(E[:keep_chain])

        # Sample
        X = self.shift_scale(X, type='node')
        E = self.shift_scale(E, type='edge')
        # X = X * 2 - 0
        # X = self.shift_scale(torch.sigmoid(logit_X_s)/alpha_s)
        # E = self.shift_scale(torch.sigmoid(logit_E_s)/alpha_s)

        beta_sample_s = diffusion_utils.sample_beta_features(X, E, node_mask, threshold=0.9, is_node_feat=True)
        out_one_hot = beta_sample_s.mask(node_mask, argmax=True, collapse=True)
        X, E, y = out_one_hot.X, out_one_hot.E, out_one_hot.y

        # Prepare the chain for saving
        if (num_chain_step is not None) and (keep_chain > 0):

            chain_X = diffusion_utils.reverse_tensor(chain_X)
            chain_E = diffusion_utils.reverse_tensor(chain_E)

            return chain_X, chain_E, node_mask

        molecule_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        # Visualize chains
        if self.visualization_tools is not None:
            self.print('\nVisualizing molecules...')

            # Visualize the final molecules
            current_path = os.getcwd()
            result_path = os.path.join(current_path,
                                       f'graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}/')
            self.visualization_tools.visualize(result_path, molecule_list, save_final)
            self.print("Done.")

        if return_E:
            return (E, n_nodes)
        return molecule_list

    def sample_new_x(self, x, x_hat, alpha_t, alpha_s, eta):
        alpha_t2s = eta * (alpha_s - alpha_t) * x_hat
        beta_t2s = eta - eta * alpha_s * x_hat
        log_u = self.log_gamma(alpha_t2s.to(torch.float32))
        log_v = self.log_gamma(beta_t2s.to(torch.float32))
        new_x = x + F.logsigmoid(log_v - log_u)

        return new_x

    def sample_new_logit_x(self, x, x_hat, alpha_t, alpha_s, eta):
        alpha_t2s = eta * (alpha_s - alpha_t) * x_hat
        beta_t2s = eta - eta * alpha_s * x_hat
        log_u = self.log_gamma(alpha_t2s.to(torch.float32))
        log_v = self.log_gamma(beta_t2s.to(torch.float32))
        log_delta = log_u - log_v

        concatenated = torch.cat([x.unsqueeze(-1), log_delta.unsqueeze(-1), (x + log_delta).unsqueeze(-1)], dim=-1)
        new_x = torch.logsumexp(concatenated, dim=-1)

        return new_x

    def compute_extra_data(self, noisy_data):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """
        if self.cfg.model.extra_features is not None:
            logit_X_t, logit_E_t, y_t, node_mask = noisy_data['X_t'], noisy_data['E_t'], noisy_data['y_t'], noisy_data[
                'node_mask']
            X_t = self.shift_scale(torch.sigmoid(logit_X_t))
            E_t = self.shift_scale(torch.sigmoid(logit_E_t))
            sample_t = diffusion_utils.sample_beta_features_mol(X_t, E_t, node_mask, threshold=0.5)

            noisy_data_discrete = {}
            noisy_data_discrete['X_t'] = sample_t.X
            noisy_data_discrete['E_t'] = sample_t.E
            noisy_data_discrete['y_t'] = y_t
            noisy_data_discrete['node_mask'] = node_mask

            extra_features = self.extra_features(noisy_data_discrete)
            extra_molecular_features = self.domain_features(noisy_data_discrete)
        else:
            extra_features = self.extra_features(noisy_data)
            extra_molecular_features = self.domain_features(noisy_data)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        if self.cfg.model.high_order:
            E_t = self.shift_scale(torch.sigmoid(noisy_data['E_t'][..., 0]), type='edge').clone()
            deg_inv_sqrt = E_t.sum(dim=-1).clamp(min=1).pow(-0.5)
            E_t_normalized = deg_inv_sqrt.unsqueeze(-1) * E_t * deg_inv_sqrt.unsqueeze(-2)
            E_t = torch.bmm(E_t, E_t_normalized)
            mask = torch.ones([E_t.shape[-1], E_t.shape[-1]]) - torch.eye(E_t.shape[-1])
            E_t = E_t * mask.unsqueeze(0).to(E_t.device)
            E_t = E_t.unsqueeze(-1)
            extra_E = torch.cat((extra_E, E_t), dim=-1)

        t = noisy_data['t']
        extra_y = torch.cat((extra_y, t), dim=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)

    def get_E_and_V(self, alpha):
        xmin = self.Shift
        xmax = self.Shift + self.Scale

        E1 = 1.0 / (self.eta * alpha * self.Scale) * (
                (self.eta * alpha * xmax).lgamma() - (self.eta * alpha * xmin).lgamma())
        E2 = 1.0 / (self.eta * alpha * self.Scale) * (
                (self.eta - self.eta * alpha * xmin).lgamma() - (self.eta - self.eta * alpha * xmax).lgamma())

        E_logit_x_t = E1 - E2

        V1 = 1.0 / (self.eta * alpha * self.Scale) * (
                (self.eta * alpha * xmax).digamma() - (self.eta * alpha * xmin).digamma())
        V2 = 1.0 / (self.eta * alpha * self.Scale) * (
                (self.eta - self.eta * alpha * xmin).digamma() - (self.eta - self.eta * alpha * xmax).digamma())

        grids = (torch.arange(0, 101, device=self.device) / 100) * self.Scale + self.Shift
        alpha_x = alpha[:, :, 0] * grids.unsqueeze(0)

        V3 = ((self.eta * alpha_x).digamma()) ** 2
        V3[:, 0] = (V3[:, 0] + V3[:, -1]) / 2
        V3 = V3[:, :-1]
        V3 = (V3.mean(dim=1).unsqueeze(1).unsqueeze(2) - E1 ** 2).clamp(0)

        V4 = ((self.eta - self.eta * alpha_x).digamma()) ** 2
        V4[:, 0] = (V4[:, 0] + V4[:, -1]) / 2
        V4 = V4[:, :-1]
        V4 = (V4.mean(dim=1).unsqueeze(1).unsqueeze(2) - E2 ** 2).clamp(0)

        std_logit_x_t = (V1 + V2 + V3 + V4).sqrt()

        return E_logit_x_t, std_logit_x_t

    def sample_from_train(self, sample_num, re_deg=False, re_feat=False, ):
        sample_idx = np.random.randint(0, len(self.train_graph_list), sample_num)
        sample_degree = self.degree_counts[sample_idx]
        sample_degree, _ = torch.sort(sample_degree, dim=-1, descending=True)
        feat = F.one_hot(sample_degree, num_classes=self.dataset_info.max_deg_num).to(torch.float32).to(self.device)
        assert sample_num == feat.shape[0]

        num_nodes = feat[..., 1:].sum(dim=(-1, -2)).to(torch.long).to(self.device)
        feat = feat[..., 1:]
        if re_deg and re_feat:
            return sample_degree, feat, num_nodes
        elif re_deg and not re_feat:
            return sample_degree, num_nodes
        else:
            return feat, num_nodes

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

    def visualization_graph_matrix(self, X, E, node_mask, given_t_split=50, graph=True, matrix=True, sample_vis=False,
                                   order=False):

        if not sample_vis:
            file_name = 'forward_process'
            node_idx = torch.argsort(E.sum((-1, -2)), dim=1, descending=True)
            X, E, node_mask = self.order_graph(X, E, node_mask, node_idx)
            x_eta = self.get_eta4x(E, self.eta)

            # Follow the steps in Beta Diffusion.
            # noisy_data in original/logit input_space

            y = torch.zeros((X.size(0), 0)).to(self.device)
            X = self.scale_shift(X)
            E = self.scale_shift(E)
            noisy_data = self.apply_noise(X, E, y, node_mask, given_t_split=given_t_split, etas=x_eta)
            if self.input_space == 'logit':
                noisy_data['X_t'] = noisy_data['X_t'].sigmoid()
                noisy_data['E_t'] = noisy_data['E_t'].sigmoid()

            noisy_data['X_t'] = self.shift_scale(noisy_data['X_t'])
            noisy_data['E_t'] = self.shift_scale(noisy_data['E_t'])

            con_X, con_E = self.mask_and_sym(noisy_data['X_t'], noisy_data['E_t'],
                                             noisy_data['node_mask'])
            X, E = con_X.clone(), con_E.clone()
            node_mask = noisy_data['node_mask']
        else:
            file_name = 'sample_process'
            if order:
                tmp_beta_sample_s = diffusion_utils.sample_beta_features(X.clone(), E.clone(), node_mask, threshold=0.5,
                                                                         is_node_feat=True)
                tmp_X, tmp_E, tmp_y = tmp_beta_sample_s.X, tmp_beta_sample_s.E, tmp_beta_sample_s.y
                tmp_E = tmp_E[..., 1:]
                final_E = [tmp_E[(idx + 1) * given_t_split - 1] for idx in range(tmp_X.size(0) // given_t_split)]

                node_idx = [torch.argsort(fE.sum((-1, -2)).view(1, -1), dim=1, descending=True) for fE in final_E]
                node_idx = torch.concat([nm.repeat(given_t_split, 1) for nm in node_idx], dim=0)

                X, E, node_mask = self.order_graph(X, E, node_mask.cpu(), node_idx)

            con_X, con_E = self.mask_and_sym(X, E, node_mask)

        # Get number of nodes in each graph
        n_nodes = node_mask.sum(-1)

        # Visualization
        print('Visualization process with {} graphs for {} time steps...'.format(X.size(0), given_t_split))

        # Graph Visualization
        if graph:
            beta_sample_s = diffusion_utils.sample_beta_features(X, E, node_mask, threshold=0.5, is_node_feat=True)
            out_one_hot = beta_sample_s.mask(node_mask, argmax=True, collapse=True)
            X, E, y = out_one_hot.X, out_one_hot.E, out_one_hot.y

            Es = []
            for i in range(X.size(0)):
                n = n_nodes[i]
                # atom_types = X[i, :n].cpu()
                Es.append(E[i, :n, :n].cpu())
            gen_graph_list = adjs_to_graphs(Es, True)

            ckpt_dir = self.cfg.general.test_only.split('checkpoints')[0]
            ckpt_name = self.cfg.general.test_only.split('/')[-1].split('=')[1].split('-')[0]
            graph_ckpt_path = os.path.join(ckpt_dir, f'{ckpt_name}.pkl')
            with open(graph_ckpt_path, 'wb') as f:
                pickle.dump(obj=gen_graph_list, file=f, protocol=pickle.HIGHEST_PROTOCOL)



            # molecule_list = []
            # for i in range(X.size(0)):
            #     n = n_nodes[i]
            #     atom_types = X[i, :n].cpu()
            #     edge_types = E[i, :n, :n].cpu()
            #     molecule_list.append([atom_types, edge_types])
            #
            # # Visualize chains
            # if self.visualization_tools is not None:
            #     for idx in range(X.size(0) // given_t_split):
            #         print(f'\nVisualizing Graphs {idx}...')
            #
            #         # Visualize the final molecules
            #         current_path = os.getcwd()
            #
            #         result_path = os.path.join(current_path,
            #                                    f'{file_name}/graph_{idx}/')
            #         self.visualization_tools.visualize(result_path,
            #                                            molecule_list[idx * given_t_split: (idx + 1) * given_t_split],
            #                                            given_t_split)
            #         print("Done.")
        # Matrix Visualization
        if matrix:
            cmap_colors = [(1, 1, 1), (0, 0, 0)]
            cmap = LinearSegmentedColormap.from_list('CustomCmap', cmap_colors)
            graph_num = con_E.size(0) // given_t_split
            for idx in range(graph_num):
                num_rows = 2
                num_cols = given_t_split // num_rows
                fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
                for state_id in range(given_t_split):
                    adj = con_E[idx * given_t_split + state_id, ..., 0].numpy()

                    ax = axes[math.floor(state_id / num_cols), state_id % num_cols]
                    ax.imshow(adj, cmap=cmap, interpolation='nearest', vmin=0, vmax=1)
                    ax.set_xticks([])
                    ax.set_yticks([])
                # plt.subplots_adjust(wspace=0, hspace=0)
                current_path = os.getcwd()

                result_path = os.path.join(current_path,
                                           f'{file_name}/graph_{idx}/')
                if not os.path.exists(result_path):
                    os.makedirs(result_path)

                plt.tight_layout()
                plt.savefig(os.path.join(result_path, 'Matrix_Process.png', ))
                # plt.show()

    def process_visualization(self, datamodule, given_t_split=20):

        data = next(iter(datamodule.test_dataloader()))

        dense, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr,
                                          data.batch, max_num_nodes=datamodule.train_dataset.max_node_num)
        dense_data = dense.mask(node_mask)
        X, E = dense_data.X, dense_data.E[..., 1:]  # remove 'no edge' and 'Aromatic bond'
        X, E = X[:self.cfg.general.vis_number], E[:self.cfg.general.vis_number]
        node_mask = node_mask[:self.cfg.general.vis_number]

        # X = utils.init_node_feats(X, E, feat_type=self.noise_feat_type, max_feat_num=self.max_feat_num)

        # TODO Visualization doesn't need scale_shift?
        # X = self.scale_shift(X)
        # E = self.scale_shift(E)

        self.visualization_graph_matrix(X, E, node_mask, given_t_split, graph=True, matrix=True)

    def sample_visualization(self, to_sample=5, given_t_split=20):
        if self.cfg.model.eta_from == 'train':
            sample_deg, nodes_num = self.sample_from_train(to_sample, re_deg=True, re_feat=False)
        # else:
        #     sample_extra_feat, nodes_num = None, None
        chain_X, chain_E, node_mask = self.sample_batch(batch_size=to_sample, num_nodes=nodes_num, eta_from=sample_deg,
                                                        num_chain_step=given_t_split, keep_chain=to_sample)

        # X = torch.stack(chain_X, dim=0)
        X = torch.concat([chain_X[:, i, ...] for i in range(chain_X.size(1))], dim=0)
        # E = torch.stack(chain_E, dim=0)
        E = torch.concat([chain_E[:, i, ...] for i in range(chain_E.size(1))], dim=0)

        node_mask = torch.concat([nm.repeat(given_t_split, 1) for nm in node_mask], dim=0)

        self.visualization_graph_matrix(X, E, node_mask, given_t_split, graph=True, matrix=True, sample_vis=True,
                                        order=True)
