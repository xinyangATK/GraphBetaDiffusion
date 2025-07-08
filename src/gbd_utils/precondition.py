import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PreConditionMoudle(object):
    def __init__(self, dataset_name, dataset_info, Shift, Scale):
        self.dataset_name = dataset_name
        self.Shift = Shift
        self.Scale = Scale

        prob = self.get_discrete_prob(self.dataset_name, dataset_info, num_classes=2)
        if isinstance(prob, tuple):
            self.prob_X, self.prob_E = prob
        else:
            self.prob_X = self.prob_E = prob

    def get_discrete_prob(self, dataset_name, dataset_info, prob=None, num_classes=None):
        if prob is not None:
            return prob
        
        if dataset_name == 'comm20':
            prob_X = torch.tensor([0.0000, 0.0393, 0.2376, 0.2376, 0.2192, 0.1394, 0.0910, 0.0340, 0.0020])
            prob_E = torch.tensor([0.2914]) 
        elif dataset_name == 'ego':
            ValueError('Not implenmted. The code will Coming soon.')
        elif dataset_name == 'planar':
            ValueError('Not implenmted. The code will Coming soon.')
        elif dataset_name == 'sbm':
            ValueError('Not implenmted. The code will Coming soon.')
        elif dataset_name == 'qm9':
            prob_X = dataset_info.node_types
            prob_E = dataset_info.edge_types_new
        elif dataset_name == 'zinc250k':
            ValueError('Not implenmted. The code will Coming soon.')
        else:
            raise ValueError('Invalid dataset')
        
        return (prob_X, prob_E)
        

    def scale_shift(self, input, type='node'):
        new_input = self.Scale[type] * input + self.Shift[type]
        return new_input

    def shift_scale(self, input, type='node'):
        new_input = (input - self.Shift[type]) / self.Scale[type]
        return new_input

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

        E1 = 1.0 / (eta * alpha_t * (xmax - xmin)) * (
                (eta * alpha_t * xmax).lgamma() - (eta * alpha_t * xmin).lgamma())
        E2 = 1.0 / (eta * alpha_t * (xmax - xmin)) * (
                (eta - eta * alpha_t * xmin).lgamma() - (eta - eta * alpha_t * xmax).lgamma())

        E_logit_x_t = E1 - E2

        V1 = 1.0 / (eta * alpha_t * (xmax - xmin)) * (
                (eta * alpha_t * xmax).digamma() - (eta * alpha_t * xmin).digamma())
        V2 = 1.0 / (eta * alpha_t * (xmax - xmin)) * (
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

        bounds = torch.tensor([self.Shift[type], self.Scale[type] + self.Shift[type]])

        if prior_prob is None:
            # continuous version of pre-conditioning
            mean_t, std_t = self.get_logit_beta_stats_con(bounds=bounds, eta=eta, alpha_t=alpha_t)
            return mean_t, std_t
        

        if isinstance(prior_prob, torch.Tensor):
            prior_prob = prior_prob.to(alpha_t.device)
            if input_space == 'logit':
                mean_t, std_t = self.get_logit_beta_stats(bounds=bounds, eta=eta, alpha_t=alpha_t, probs=(1 - prior_prob, prior_prob))
            else:
                mean_t, std_t = self.get_beta_stats(bounds=bounds, eta=eta, alpha_t=alpha_t, probs=(1 - prior_prob, prior_prob))
        else:
            probs = torch.tensor([1 - prior_prob, prior_prob]).to(self.device)

            if input_space == 'logit':
                mean_t, std_t = self.get_logit_beta_stats(bounds=bounds, eta=eta, alpha_t=alpha_t, probs=probs)
            else:
                mean_t, std_t = self.get_beta_stats(bounds=bounds, eta=eta, alpha_t=alpha_t, probs=probs)

        return mean_t, std_t


