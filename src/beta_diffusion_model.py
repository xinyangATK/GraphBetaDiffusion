import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import time
import wandb
import os
import numpy as np
from models.transformer_model import GraphTransformer
from diffusion.noise_schedule import DiscreteUniformTransition, PredefinedBetaNoiseScheduleDiscrete,\
    MarginalUniformTransition
from src.diffusion import diffusion_utils
from metrics.train_metrics import TrainLossBeta, TrainLoss, TrainLossDiscrete
from metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchKL, NLL
from src import utils
EPS=torch.finfo(torch.float32).eps
MIN=torch.finfo(torch.float32).tiny
clamp_min = 0.405465
clamp_max = 4.595119



class BetaDiffusion(pl.LightningModule):
    def __init__(self, cfg, dataset_infos, train_metrics, sampling_metrics, visualization_tools, extra_features,
                 domain_features):
        super().__init__()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        self.cfg = cfg
        self.name = cfg.general.name
        self.model_dtype = torch.float32
        self.T = cfg.model.diffusion_steps

        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        self.node_dist = nodes_dist

        self.dataset_info = dataset_infos

        self.train_loss = TrainLossBeta(self.cfg.model.lambda_train)
        # self.train_loss = TrainLossDiscrete(self.cfg.model.lambda_train)

        self.val_nll = NLL()
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()

        self.test_nll = NLL()
        self.test_X_kl = SumExceptBatchKL()
        self.test_E_kl = SumExceptBatchKL()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics

        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features

        self.model = GraphTransformer(n_layers=cfg.model.n_layers,
                                      input_dims=input_dims,
                                      hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                      hidden_dims=cfg.model.hidden_dims,
                                      output_dims=output_dims,
                                      act_fn_in=nn.ReLU(),
                                      act_fn_out=nn.ReLU())

        # self.beta_noise_schedule = PredefinedBetaNoiseScheduleDiscrete(cfg.model.diffusion_noise_schedule,
        #                                                       timesteps=cfg.model.diffusion_steps)

        if cfg.model.transition == 'uniform':
            self.transition_model = DiscreteUniformTransition(x_classes=self.Xdim_output, e_classes=self.Edim_output,
                                                              y_classes=self.ydim_output)
            x_limit = torch.ones(self.Xdim_output) / self.Xdim_output
            e_limit = torch.ones(self.Edim_output) / self.Edim_output
            y_limit = torch.ones(self.ydim_output) / self.ydim_output
            self.limit_dist = utils.PlaceHolder(X=x_limit, E=e_limit, y=y_limit)
        elif cfg.model.transition == 'marginal':

            node_types = self.dataset_info.node_types.float()
            x_marginals = node_types / torch.sum(node_types)

            edge_types = self.dataset_info.edge_types.float()
            e_marginals = edge_types / torch.sum(edge_types)
            print(f"Marginal distribution of the classes: {x_marginals} for nodes, {e_marginals} for edges")
            self.transition_model = MarginalUniformTransition(x_marginals=x_marginals, e_marginals=e_marginals,
                                                              y_classes=self.ydim_output)
            self.limit_dist = utils.PlaceHolder(X=x_marginals, E=e_marginals,
                                                y=torch.ones(self.ydim_output) / self.ydim_output)

        self.save_hyperparameters(ignore=['train_metrics', 'sampling_metrics'])
        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.best_val_nll = 1e8
        self.val_counter = 0

        # Hyper-parameters for BetaDIffusion
        self.sigmoid_start = 9
        self.sigmoid_end = -9
        self.sigmoid_power = 0.5
        self.beta_max = 20
        self.beta_min = 0.1
        self.Scale = 0.39
        self.Shift = 0.60
        self.eta = torch.tensor(30, dtype=torch.float32)
        self.use_fea = True
    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    def calculate_coefs(self, t):
        t_norm = t
        s_norm = t_norm * 0.95

        # apply noise schedule
        logit_alpha_t = self.sigmoid_start + (self.sigmoid_end - self.sigmoid_start) * (
                    t_norm ** self.sigmoid_power)
        logit_alpha_s = self.sigmoid_start + (self.sigmoid_end - self.sigmoid_start) * (
                    s_norm ** self.sigmoid_power)

        alpha_t, alpha_s = torch.sigmoid(logit_alpha_t), torch.sigmoid(logit_alpha_s)
        delta = (logit_alpha_s.to(torch.float64).sigmoid() - logit_alpha_t.to(torch.float64).sigmoid()).to(torch.float32)

        coefs = {'alpha_t': alpha_t, 'alpha_s': alpha_s, 'delta': delta}

        return coefs

    def training_step(self, data, i):
        if data.edge_index.numel() == 0:
            self.print("Found a batch with no edges. Skipping.")
            return
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E[..., -1].unsqueeze(-1)
        # X = (X - 0) / 2
        X = self.scale_shift(X)
        E = self.scale_shift(E)

        # Follow the steps in Beta Diffusion.
        # noisy_data in logit space
        noisy_data = self.apply_noise(X, E, data.y, node_mask)

        extra_data = self.compute_extra_data(noisy_data, use_fea=self.use_fea)

        # pred in logit space
        pred = self.forward(noisy_data, extra_data, node_mask, use_fea=self.use_fea)

        coefs = self.calculate_coefs(noisy_data['t'])

        pred_X = self.scale_shift(torch.sigmoid(pred.X))
        pred_E = self.scale_shift(torch.sigmoid(pred.E))


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


        loss = self.train_loss(masked_pred_X=pred_X, masked_pred_E=pred_E, pred_y=pred.y,
                               true_X=X, true_E=E, true_y=data.y, coefs=coefs,
                               log=i % self.log_every_steps == 0)

        self.train_metrics(masked_pred_X=pred_X, masked_pred_E=pred_E, true_X=X, true_E=E,
                           log=i % self.log_every_steps == 0)

        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, amsgrad=True,
                                 weight_decay=self.cfg.train.weight_decay)

    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        self.print("Size of the input features", self.Xdim, self.Edim, self.ydim)
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)

    def on_train_epoch_start(self) -> None:
        self.print("Starting train epoch...")
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        to_log = self.train_loss.log_epoch_metrics()
        self.print(f"Epoch {self.current_epoch}: X_KLUB: {to_log['train_epoch/x_KLUB'] :.3f}"
                      f" -- E_KLUB: {to_log['train_epoch/E_KLUB'] :.3f} --"
                      f" y_CE: {to_log['train_epoch/y_CE'] :.3f}"
                      f" -- {time.time() - self.start_epoch_time:.1f}s ")
        epoch_at_metrics, epoch_bond_metrics = self.train_metrics.log_epoch_metrics()
        self.print(f"Epoch {self.current_epoch}: {epoch_at_metrics} -- {epoch_bond_metrics}")
        # print(torch.cuda.memory_summary())

    def on_validation_epoch_start(self) -> None:
        # self.val_nll.reset()
        # self.val_X_kl.reset()
        # self.val_E_kl.reset()
        # self.val_X_logp.reset()
        # self.val_E_logp.reset()
        self.sampling_metrics.reset()

    def validation_step(self, data, i):
        return {'loss': None}
        # dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        # dense_data = dense_data.mask(node_mask)
        #
        # # Follow the steps in Beta Diffusion.
        # noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        # extra_data = self.compute_extra_data(noisy_data)
        # pred = self.forward(noisy_data, extra_data, node_mask)
        # coefs = self.calculate_coefs(pred.t)
        # nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y, node_mask, test=False)

        # loss = self.train_loss(masked_pred_X=pred.X, masked_pred_E=pred.E, pred_y=pred.y,
        #                        true_X=X, true_E=E, true_y=data.y, coefs=coefs,
        #                        log=i % self.log_every_steps == 0)
        # return {'loss': nll}

    def on_validation_epoch_end(self) -> None:
        self.print(f"Validation on Epoch {self.current_epoch}...")
        self.val_counter += 1
        if self.val_counter % self.cfg.general.sample_every_val == 0:
            start = time.time()
            samples_left_to_generate = self.cfg.general.samples_to_generate
            samples_left_to_save = self.cfg.general.samples_to_save
            chains_left_to_save = self.cfg.general.chains_to_save

            samples = []

            ident = 0
            while samples_left_to_generate > 0:
                bs = 2 * self.cfg.train.batch_size
                to_generate = min(samples_left_to_generate, bs)
                to_save = min(samples_left_to_save, bs)
                chains_save = min(chains_left_to_save, bs)
                samples.extend(self.sample_batch(batch_id=ident, batch_size=to_generate, num_nodes=None,
                                                 save_final=to_save,
                                                 keep_chain=chains_save,
                                                 number_chain_steps=self.number_chain_steps))
                ident += to_generate

                samples_left_to_save -= to_save
                samples_left_to_generate -= to_generate
                chains_left_to_save -= chains_save
            self.print("Computing sampling metrics...")
            self.sampling_metrics.forward(samples, self.name, self.current_epoch, val_counter=-1, test=False,
                                          local_rank=self.local_rank)
            self.print(f'Done. Sampling took {time.time() - start:.2f} seconds\n')
            print("Validation epoch end ends...")

    def on_test_epoch_start(self) -> None:
        self.print("Starting test...")
        # self.test_nll.reset()
        # self.test_X_kl.reset()
        # self.test_E_kl.reset()
        # self.test_X_logp.reset()
        # self.test_E_logp.reset()
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)

    def test_step(self, data, i):
        return {'loss': None}
        # dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        # dense_data = dense_data.mask(node_mask)
        # noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        # extra_data = self.compute_extra_data(noisy_data)
        # pred = self.forward(noisy_data, extra_data, node_mask)
        # nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y, node_mask, test=True)
        # return {'loss': nll}

    def on_test_epoch_end(self) -> None:
        """ Measure likelihood on a test set and compute stability metrics. """
        # metrics = [self.test_nll.compute(), self.test_X_kl.compute(), self.test_E_kl.compute(),
        #            self.test_X_logp.compute(), self.test_E_logp.compute()]
        # if wandb.run:
        #     wandb.log({"test/epoch_NLL": metrics[0],
        #                "test/X_kl": metrics[1],
        #                "test/E_kl": metrics[2],
        #                "test/X_logp": metrics[3],
        #                "test/E_logp": metrics[4]}, commit=False)
        #
        # self.print(f"Epoch {self.current_epoch}: Test NLL {metrics[0] :.2f} -- Test Atom type KL {metrics[1] :.2f} -- ",
        #            f"Test Edge type KL: {metrics[2] :.2f}")

        # test_nll = metrics[0]
        # if wandb.run:
        #     wandb.log({"test/epoch_NLL": test_nll}, commit=False)
        #
        # self.print(f'Test loss: {test_nll :.4f}')
        self.print(f"Test on Epoch {self.current_epoch}...")

        samples_left_to_generate = self.cfg.general.final_model_samples_to_generate
        samples_left_to_save = self.cfg.general.final_model_samples_to_save
        chains_left_to_save = self.cfg.general.final_model_chains_to_save

        # samples = []
        id = 0
        d, c, o = [], [], []
        while samples_left_to_generate > 0:
            self.print(f'Samples left to generate: {samples_left_to_generate}/'
                       f'{self.cfg.general.final_model_samples_to_generate}', end='', flush=True)
            bs = 20 #2 * self.cfg.train.batch_size
            to_generate = min(samples_left_to_generate, bs)
            to_save = min(samples_left_to_save, bs)
            chains_save = min(chains_left_to_save, bs)
            samples = self.sample_batch(id, to_generate, num_nodes=None, save_final=to_save,
                                             keep_chain=chains_save, number_chain_steps=self.number_chain_steps)
            id += to_generate
            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate
            chains_left_to_save -= chains_save
        # with open(filename, 'w') as f:
        #     for item in samples:
        #         f.write(f"N={item[0].shape[0]}\n")
        #         atoms = item[0].tolist()
        #         f.write("X: \n")
        #         for at in atoms:
        #             f.write(f"{at} ")
        #         f.write("\n")
        #         f.write("E: \n")
        #         for bond_list in item[1]:
        #             for bond in bond_list:
        #                 f.write(f"{bond} ")
        #             f.write("\n")
        #         f.write("\n")
            self.print("Generated graphs Saved. Computing sampling metrics...")
            d_, c_, o_ = self.sampling_metrics(samples, self.name, self.current_epoch, self.val_counter, test=True, local_rank=self.local_rank)
            d.append(d_)
            c.append(c_)
            o.append(o_)
        d = np.array(d)
        c = np.array(c)
        o = np.array(o)
        print(f"degree:\n mean = {d.mean()}  std = {d.std()}")
        print(f"clustering:\n mean = {c.mean()}  std = {c.std()}")
        print(f"orbit:\n mean = {o.mean()}  std = {o.std()}")

        self.print("Done testing.")

    def log_gamma(self, alpha):
        return torch.log(torch._standard_gamma(alpha.to(torch.float32)).clamp(MIN))

    def kl_prior(self, X, E, node_mask):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((X.size(0), 1), device=X.device)
        Ts = self.T * ones
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts)  # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)
        assert probX.shape == X.shape

        bs, n, _ = probX.shape

        limit_X = self.limit_dist.X[None, None, :].expand(bs, n, -1).type_as(probX)
        limit_E = self.limit_dist.E[None, None, None, :].expand(bs, n, n, -1).type_as(probE)

        # Make sure that masked rows do not contribute to the loss
        limit_dist_X, limit_dist_E, probX, probE = diffusion_utils.mask_distributions(true_X=limit_X.clone(),
                                                                                      true_E=limit_E.clone(),
                                                                                      pred_X=probX,
                                                                                      pred_E=probE,
                                                                                      node_mask=node_mask)

        kl_distance_X = F.kl_div(input=probX.log(), target=limit_dist_X, reduction='none')
        kl_distance_E = F.kl_div(input=probE.log(), target=limit_dist_E, reduction='none')

        return diffusion_utils.sum_except_batch(kl_distance_X) + \
               diffusion_utils.sum_except_batch(kl_distance_E)

    def compute_Lt(self, X, E, y, pred, noisy_data, node_mask, test):
        pred_probs_X = F.softmax(pred.X, dim=-1)
        pred_probs_E = F.softmax(pred.E, dim=-1)
        pred_probs_y = F.softmax(pred.y, dim=-1)

        Qtb = self.transition_model.get_Qt_bar(noisy_data['alpha_t_bar'], self.device)
        Qsb = self.transition_model.get_Qt_bar(noisy_data['alpha_s_bar'], self.device)
        Qt = self.transition_model.get_Qt(noisy_data['beta_t'], self.device)

        # Compute distributions to compare with KL
        bs, n, d = X.shape
        prob_true = diffusion_utils.posterior_distributions(X=X, E=E, y=y, X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_true.E = prob_true.E.reshape((bs, n, n, -1))
        prob_pred = diffusion_utils.posterior_distributions(X=pred_probs_X, E=pred_probs_E, y=pred_probs_y,
                                                            X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_pred.E = prob_pred.E.reshape((bs, n, n, -1))

        # Reshape and filter masked rows
        prob_true_X, prob_true_E, prob_pred.X, prob_pred.E = diffusion_utils.mask_distributions(true_X=prob_true.X,
                                                                                                true_E=prob_true.E,
                                                                                                pred_X=prob_pred.X,
                                                                                                pred_E=prob_pred.E,
                                                                                                node_mask=node_mask)
        kl_x = (self.test_X_kl if test else self.val_X_kl)(prob_true.X, torch.log(prob_pred.X))
        kl_e = (self.test_E_kl if test else self.val_E_kl)(prob_true.E, torch.log(prob_pred.E))
        return self.T * (kl_x + kl_e)

    def reconstruction_logp(self, t, X, E, node_mask):
        # Compute noise values for t = 0.
        t_zeros = torch.zeros_like(t)
        beta_0 = self.noise_schedule(t_zeros)
        Q0 = self.transition_model.get_Qt(beta_t=beta_0, device=self.device)

        probX0 = X @ Q0.X  # (bs, n, dx_out)
        probE0 = E @ Q0.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled0 = diffusion_utils.sample_discrete_features(probX=probX0, probE=probE0, node_mask=node_mask)

        X0 = F.one_hot(sampled0.X, num_classes=self.Xdim_output).float()
        E0 = F.one_hot(sampled0.E, num_classes=self.Edim_output).float()
        y0 = sampled0.y
        assert (X.shape == X0.shape) and (E.shape == E0.shape)

        sampled_0 = utils.PlaceHolder(X=X0, E=E0, y=y0).mask(node_mask)

        # Predictions
        noisy_data = {'X_t': sampled_0.X, 'E_t': sampled_0.E, 'y_t': sampled_0.y, 'node_mask': node_mask,
                      't': torch.zeros(X0.shape[0], 1).type_as(y0)}
        extra_data = self.compute_extra_data(noisy_data)
        pred0 = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        probX0 = F.softmax(pred0.X, dim=-1)
        probE0 = F.softmax(pred0.E, dim=-1)
        proby0 = F.softmax(pred0.y, dim=-1)

        # Set masked rows to arbitrary values that don't contribute to loss
        probX0[~node_mask] = torch.ones(self.Xdim_output).type_as(probX0)
        probE0[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))] = torch.ones(self.Edim_output).type_as(probE0)

        diag_mask = torch.eye(probE0.size(1)).type_as(probE0).bool()
        diag_mask = diag_mask.unsqueeze(0).expand(probE0.size(0), -1, -1)
        probE0[diag_mask] = torch.ones(self.Edim_output).type_as(probE0)

        return utils.PlaceHolder(X=probX0, E=probE0, y=proby0)

    def apply_noise(self, X, E, y, node_mask):
        """ Sample noise and apply it to the data. """

        # Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately
        # lowest_t = 0 if self.training else 1

        t_float = torch.rand([X.size(0), 1], device=X.device).float() * (1e-5 - 1.) + 1

        # logit_alpha_t = self.noise_schedule(t_normalized=t_float)
        logit_alpha_t = self.sigmoid_start + (self.sigmoid_end - self.sigmoid_start) * (t_float ** self.sigmoid_power)
        alpha_t = torch.sigmoid(logit_alpha_t).unsqueeze(2)

        # X (in logit space)
        log_u = self.log_gamma(self.eta * alpha_t * X)
        log_v = self.log_gamma(self.eta - self.eta * alpha_t * X)
        logit_X = log_u - log_v


        # E (in logit space)
        log_u = self.log_gamma((self.eta * alpha_t.unsqueeze(-1)) * E)
        log_v = self.log_gamma(self.eta - (self.eta * alpha_t.unsqueeze(-1)) * E)
        logit_E = (log_u - log_v)  #.clamp(clamp_min, clamp_max)

        # E_logit, std_logit = self.get_E_and_V(alpha_t)
        #
        # # logit_X = (logit_X - E.unsqueeze(-1)) / (std.unsqueeze(-1))
        # logit_E = (logit_E - (E_logit.unsqueeze(-1))) / (std_logit.unsqueeze(-1))

        logit_X, logit_E = self.mask_and_sym(logit_X, logit_E, node_mask)



        logit_z_t = utils.PlaceHolder(X=logit_X, E=logit_E, y=y).type_as(X).mask(node_mask)

        noisy_data = {'t': t_float, 'logit_alpha_t': logit_alpha_t, 'X_t': logit_z_t.X, 'E_t': logit_z_t.E, 'y_t': logit_z_t.y, 'node_mask': node_mask}

        # X = torch.sigmoid(logit_X)
        # E = torch.sigmoid(logit_E)
        # X = self.shift_scale(X)
        # E = self.shift_scale(E)  # value in (0, 1)

        # X, E = self.mask_and_sym(X, E, node_mask)
        #
        # # X = self.shift_scale(X)
        # # E = self.shift_scale(E)
        #
        # # sampled_t = diffusion_utils.sample_beta_features(probX=X, probE=E, node_mask=node_mask)
        #
        # z_t = utils.PlaceHolder(X=X, E=E, y=y).type_as(X).mask(node_mask)
        #
        # noisy_data = {'t': t_float,
        #               # 'logit_X': logit_X, 'logit_E': logit_E,
        #               'alpha_t': alpha_t, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}

        return noisy_data

    def compute_val_loss(self, pred, noisy_data, X, E, y, node_mask, test=False):
        """Computes an estimator for the variational lower bound.
           pred: (batch_size, n, total_features)
           noisy_data: dict
           X, E, y : (bs, n, dx),  (bs, n, n, de), (bs, dy)
           node_mask : (bs, n)
           Output: nll (size 1)
       """
        t = noisy_data['t']

        # 1.
        N = node_mask.sum(1).long()
        log_pN = self.node_dist.log_prob(N)

        # 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
        kl_prior = self.kl_prior(X, E, node_mask)

        # 3. Diffusion loss
        loss_all_t = self.compute_Lt(X, E, y, pred, noisy_data, node_mask, test)

        # 4. Reconstruction loss
        # Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
        prob0 = self.reconstruction_logp(t, X, E, node_mask)

        loss_term_0 = self.val_X_logp(X * prob0.X.log()) + self.val_E_logp(E * prob0.E.log())

        # Combine terms
        nlls = - log_pN + kl_prior + loss_all_t - loss_term_0
        assert len(nlls.shape) == 1, f'{nlls.shape} has more than only batch dim.'

        # Update NLL metric object and return batch nll
        nll = (self.test_nll if test else self.val_nll)(nlls)        # Average over the batch

        if wandb.run:
            wandb.log({"kl prior": kl_prior.mean(),
                       "Estimator loss terms": loss_all_t.mean(),
                       "log_pn": log_pN.mean(),
                       "loss_term_0": loss_term_0,
                       'batch_test_nll' if test else 'val_nll': nll}, commit=False)
        return nll

    def forward(self, noisy_data, extra_data, node_mask, use_fea=False):
        if use_fea:
            X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
            E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        else:
            X = noisy_data['X_t'].float()
            E = noisy_data['E_t'].float()
        y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
        return self.model(X, E, y, node_mask)

    def mask_and_sym(self, X, E, node_mask, mask=True, diag=True):
        if mask:
            bs, n, _ = X.shape
            # Noise X
            # The masked rows should define probability distributions with zero
            X[~node_mask] = 0.

            # Noise E
            # The masked rows should define probability distributions as well
            inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
            diag_mask = torch.eye(n).unsqueeze(0).expand(bs, -1, -1)

            # The masked pos should define probability distributions with zero
            E[inverse_edge_mask] = 0.
            if diag:
                E[diag_mask.bool()] = 0.

        E = torch.triu(E[..., 0], diagonal=1)
        E = (E + torch.transpose(E, 1, 2))
        E = E.unsqueeze(-1)

        return X, E

    def scale_shift(self, input):
        new_input = self.Scale * input + self.Shift
        return new_input

    def shift_scale(self, input):
        new_input = (input - self.Shift) / self.Scale
        return new_input

    @torch.no_grad()
    def sample_batch(self, batch_id: int, batch_size: int, keep_chain: int, number_chain_steps: int,
                     save_final: int, alpha_T=None,
                     num_nodes=None):
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
        n_max = torch.max(n_nodes).item()
        # Build the masks
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)

        # Build noise schedule
        sample_steps = 1 - torch.arange(self.T) / (self.T - 1) * (1. - 1e-5)
        sample_steps = torch.cat([sample_steps, torch.zeros(1)]).to(self.device)

        logit_alpha = self.sigmoid_start + (self.sigmoid_end - self.sigmoid_start) * (sample_steps ** self.sigmoid_power)
        alpha = torch.sigmoid(logit_alpha)

        alpha_T = alpha_T if alpha_T is not None else alpha[0]

        N = node_mask.size(1)
        # Sample noise  -- x_next has size (n_samples, n_nodes, n_features)
        # Initiate E_T adn X_T
        # y = torch.empty((batch_size, 0))
        y = torch.zeros((batch_size, 0)).to(self.device)

        # TODO: replace 0.5 with data.mean()
        X_s = 0 * torch.ones(batch_size, N, 1, device=self.device)
        E_s = 0 * torch.ones(batch_size, N, N, 1, device=self.device)
        # X_s, E_s = self.mask_and_sym(X_s, E_s, node_mask, mask=True)
        X_s = self.scale_shift(X_s)
        E_s = self.scale_shift(E_s)

        log_u_x = self.log_gamma((self.eta * alpha_T * X_s).to(torch.float32) ).to(torch.float64)
        log_v_x = self.log_gamma((self.eta - self.eta * alpha_T * X_s).to(torch.float32) ).to(torch.float64)
        logit_X_s = (log_u_x - log_v_x).to(self.device)

        log_u_e = self.log_gamma((self.eta * alpha_T * E_s).to(torch.float32) ).to(torch.float64)
        log_v_e = self.log_gamma((self.eta - self.eta * alpha_T * E_s).to(torch.float32)).to(torch.float64)
        logit_E_s = (log_u_e - log_v_e)  # .clamp(clamp_min, clamp_max).to(self.device)

        assert number_chain_steps < self.T
        # chain_X_size = torch.Size((number_chain_steps, keep_chain, logit_X_s.size(1)))
        # chain_E_size = torch.Size((number_chain_steps, keep_chain, logit_E_s.size(1), logit_E_s.size(2)))

        # chain_X = torch.zeros(chain_X_size)
        # chain_E = torch.zeros(chain_E_size)

        for i, (logit_alpha_t, logit_alpha_s) in enumerate(zip(logit_alpha[:-1], logit_alpha[1:])):  # 0, ..., self.T - 1
            alpha_t, alpha_s = torch.sigmoid(logit_alpha_t), torch.sigmoid(logit_alpha_s)

            # logit_E_t = logit_E_s  # [B, N, N, 2]
            # logit_E_t = (logit_E_t + logit_E_t.transpose(1, 2))

            # logit_X_t = logit_X_s  # B * N * 1

            # beta_sample_s = diffusion_utils.sample_beta_features(X, E, node_mask)
            # X, E = beta_sample_s.X, beta_sample_s.E

            # MASK input
            # GET NEXT INPUT
            # E_logit, std_logit = self.get_E_and_V(alpha_t.unsqueeze(0).unsqueeze(1).unsqueeze(2))
            # logit_E_s_norm = (logit_E_s - (E_logit.unsqueeze(-1))) / (std_logit.unsqueeze(-1))
            # X_s = torch.sigmoid(X_s)
            # E_s = torch.sigmoid(E_s)
            # X_s = self.shift_scale(X_s)
            # E_s = self.shift_scale(E_s)

            logit_X_t, logit_E_t = self.mask_and_sym(logit_X_s, logit_E_s, node_mask)

            # z_hat = f(z_t, t)
            t = sample_steps[i].unsqueeze(-1).repeat(batch_size, 1)

            with torch.no_grad():
                # Neural net predictions
                noisy_data = {'X_t': logit_X_t, 'E_t': logit_E_t, 'y_t': y, 't': t, 'logit_alpha_t': logit_alpha_t.unsqueeze(0).unsqueeze(0).expand(batch_size, -1), 'node_mask': node_mask}
                extra_data = self.compute_extra_data(noisy_data, use_fea=self.use_fea)
                pred = self.forward(noisy_data, extra_data, node_mask, use_fea=self.use_fea)

            X = torch.sigmoid(pred.X)
            E = torch.sigmoid(pred.E)
            X = self.scale_shift(X)
            E = self.scale_shift(E)

            X, E = self.mask_and_sym(X, E, node_mask)
            # beta_sample_s = diffusion_utils.sample_beta_features(X, E, node_mask, threshold=0.5)

            #  new logit_X and logit_E
            logit_X_s = self.sample_new_logit_x(logit_X_s, X, alpha_t, alpha_s)
            logit_E_s = self.sample_new_logit_x(logit_E_s, E, alpha_t, alpha_s)


        # Sample
        X = self.shift_scale(X)
        E = self.shift_scale(E)
        # X = X * 2 - 0
        # X = self.shift_scale(torch.sigmoid(logit_X_s)/alpha_s)
        # E = self.shift_scale(torch.sigmoid(logit_E_s)/alpha_s)

        beta_sample_s = diffusion_utils.sample_beta_features(X, E, node_mask, threshold=0.5)
        out_one_hot = beta_sample_s.mask(node_mask, collapse=True)
        X, E, y = out_one_hot.X, out_one_hot.E, out_one_hot.y

        # Prepare the chain for saving
        # if keep_chain > 0:
        #     final_X_chain = X[:keep_chain]
        #     final_E_chain = E[:keep_chain]
        #
        #     chain_X[0] = final_X_chain                  # Overwrite last frame with the resulting X, E
        #     chain_E[0] = final_E_chain
        #
        #     chain_X = diffusion_utils.reverse_tensor(chain_X)
        #     chain_E = diffusion_utils.reverse_tensor(chain_E)
        #
        #     # Repeat last frame to see final sample better
        #     chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
        #     chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
        #     assert chain_X.size(0) == (number_chain_steps + 10)

        molecule_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        # Visualize chains
        if self.visualization_tools is not None:
            # self.print('Visualizing chains...')
            # current_path = os.getcwd()
            # num_molecules = chain_X.size(1)       # number of molecules
            # for i in range(num_molecules):
            #     result_path = os.path.join(current_path, f'chains/{self.cfg.general.name}/'
            #                                              f'epoch{self.current_epoch}/'
            #                                              f'chains/molecule_{batch_id + i}')
            #     if not os.path.exists(result_path):
            #         os.makedirs(result_path)
            #         _ = self.visualization_tools.visualize_chain(result_path,
            #                                                      chain_X[:, i, :].numpy(),
            #                                                      chain_E[:, i, :].numpy())
            #     self.print('\r{}/{} complete'.format(i+1, num_molecules), end='', flush=True)
            self.print('\nVisualizing molecules...')

            # Visualize the final molecules
            current_path = os.getcwd()
            result_path = os.path.join(current_path,
                                       f'graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}/')
            self.visualization_tools.visualize(result_path, molecule_list, save_final)
            self.print("Done.")

        return molecule_list

    def sample_new_logit_x(self, x, x_hat, alpha_t, alpha_s):
        alpha_t2s = self.eta * (alpha_s - alpha_t) * x_hat
        beta_t2s = self.eta - self.eta * alpha_s * x_hat
        log_u = self.log_gamma(alpha_t2s.to(torch.float32))
        log_v = self.log_gamma(beta_t2s.to(torch.float32))
        log_delta = log_u - log_v

        concatenated = torch.cat([x.unsqueeze(-1), log_delta.unsqueeze(-1), (x + log_delta).unsqueeze(-1)], dim=-1)
        new_x = torch.logsumexp(concatenated, dim=-1)

        return new_x


    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t, node_mask):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
        bs, n, dxs = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # Retrieve transitions matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        # Neural net predictions
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 'node_mask': node_mask}
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)               # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)               # bs, n, n, d0

        p_s_and_t_given_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=X_t,
                                                                                           Qt=Qt.X,
                                                                                           Qsb=Qsb.X,
                                                                                           Qtb=Qtb.X)

        p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=E_t,
                                                                                           Qt=Qt.E,
                                                                                           Qsb=Qsb.E,
                                                                                           Qtb=Qtb.E)
        # Dim of these two tensors: bs, N, d0, d_t-1
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X         # bs, n, d0, d_t-1
        unnormalized_prob_X = weighted_X.sum(dim=2)                     # bs, n, d_t-1
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1

        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E        # bs, N, d0, d_t-1
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, node_mask=node_mask)

        X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))

        return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t)

    def compute_extra_data(self, noisy_data, use_fea=False):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """
        if use_fea:
            logit_X_t, logit_E_t, y_t, node_mask = noisy_data['X_t'], noisy_data['E_t'], noisy_data['y_t'], noisy_data['node_mask']
            X_t = self.shift_scale(torch.sigmoid(logit_X_t))
            E_t = self.shift_scale(torch.sigmoid(logit_E_t))
            sample_t = diffusion_utils.sample_beta_features(X_t, E_t, node_mask, threshold=0.25)
            # X_t = sample_t.X.unsqueeze(-1)
            # E_t = sample_t.E.unsqueeze(-1)
            noisy_data_discrete = {}
            noisy_data_discrete['X_t'] = sample_t.X
            noisy_data_discrete['E_t'] = sample_t.E
            noisy_data_discrete['y_t'] = y_t
            noisy_data_discrete['node_mask'] = node_mask

            extra_features = self.extra_features(noisy_data_discrete)
            extra_molecular_features = self.domain_features(noisy_data_discrete)

            extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
            extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
            extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

            t = noisy_data['t']
            extra_y = torch.cat((extra_y, t), dim=1)
        else:
            extra_X = noisy_data['X_t']
            extra_E = noisy_data['E_t']
            extra_y = noisy_data['t']

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

        V3 = ((self.eta * alpha_x).digamma())**2
        V3[:, 0] = (V3[:, 0] + V3[:, -1]) / 2
        V3 = V3[:, :-1]
        V3 = (V3.mean(dim=1).unsqueeze(1).unsqueeze(2) - E1**2).clamp(0)

        V4 = ((self.eta - self.eta * alpha_x).digamma())**2
        V4[:, 0] = (V4[:, 0] + V4[:, -1]) / 2
        V4 = V4[:, :-1]
        V4 = (V4.mean(dim=1).unsqueeze(1).unsqueeze(2) - E2**2).clamp(0)

        std_logit_x_t = (V1 + V2 + V3 + V4).sqrt()

        return E_logit_x_t, std_logit_x_t




























