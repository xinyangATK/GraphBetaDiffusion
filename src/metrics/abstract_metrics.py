import torch
from torch import Tensor
from torch.nn import functional as F
from torchmetrics import Metric, MeanSquaredError


class TrainAbstractMetricsDiscrete(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, masked_pred_X, masked_pred_E, true_X, true_E, log: bool):
        pass

    def reset(self):
        pass

    def log_epoch_metrics(self):
        return None, None


class TrainAbstractMetrics(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, masked_pred_epsX, masked_pred_epsE, pred_y, true_epsX, true_epsE, true_y, log):
        pass

    def reset(self):
        pass

    def log_epoch_metrics(self):
        return None, None


class SumExceptBatchMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_value', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, values) -> None:
        self.total_value += torch.sum(values)
        self.total_samples += values.shape[0]

    def compute(self):
        return self.total_value / self.total_samples


class SumExceptBatchMSE(MeanSquaredError):
    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        assert preds.shape == target.shape
        sum_squared_error, n_obs = self._mean_squared_error_update(preds, target)

        self.sum_squared_error += sum_squared_error
        self.total += n_obs

    def _mean_squared_error_update(self, preds: Tensor, target: Tensor):
            """ Updates and returns variables required to compute Mean Squared Error. Checks for same shape of input
            tensors.
                preds: Predicted tensor
                target: Ground truth tensor
            """
            diff = preds - target
            sum_squared_error = torch.sum(diff * diff)
            n_obs = preds.shape[0]
            return sum_squared_error, n_obs


class SumExceptBatchKL(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_value', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, p, q) -> None:
        self.total_value += F.kl_div(q, p, reduction='sum')
        self.total_samples += p.size(0)

    def compute(self):
        return self.total_value / self.total_samples


class CrossEntropyMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_ce', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """ Update state with predictions and targets.
            preds: Predictions from model   (bs * n, d) or (bs * n * n, d)
            target: Ground truth values     (bs * n, d) or (bs * n * n, d). """
        target = torch.argmax(target, dim=-1)
        output = F.cross_entropy(preds, target, reduction='sum')
        self.total_ce += output
        self.total_samples += preds.size(0)

    def compute(self):
        return self.total_ce / self.total_samples

class KLUBs(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_klub', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.eta = torch.tensor(10000, dtype=torch.float32)

    def KL_gamma(*args):
        """
        Calculates the KL divergence between two Gamma distributions.
        args[0]: alpha_p, the shape of the first Gamma distribution Gamma(alpha_p,beta_p).
        args[1]: alpha_q,the shape of the second Gamma distribution Gamma(alpha_q,beta_q).
        args[2]: beta_p, the rate (inverse scale) of the first Gamma distribution Gamma(alpha_p,beta_p).
        args[3]: beta_q, the rate (inverse scale) of the second Gamma distribution Gamma(alpha_q,beta_q).
        """
        alpha_p = args[1] + 0.01
        alpha_q = args[2] + 0.01
        KL = (alpha_p - alpha_q) * torch.digamma(alpha_p) - torch.lgamma(alpha_p) + torch.lgamma(alpha_q)
        # if KL.isnan().any():
        #     print('Stop!')
        if len(args) > 3:
            beta_p = args[3] + torch.finfo(torch.float32).eps
            beta_q = args[4] + torch.finfo(torch.float32).eps
            KL = KL + alpha_q * (torch.log(beta_p) - torch.log(beta_q)) + alpha_p * (beta_q / beta_p - 1.0)
        return KL

    def update(self, preds: Tensor, target: Tensor, coefs: dict, z_mask: Tensor, n: int) -> None:
        """ Update state with predictions and targets.
            preds: Predictions from model   (bs * n, d) or (bs * n * n, d)
            target: Ground truth values     (bs * n, d) or (bs * n * n, d). """

        alpha_t = coefs['alpha_t'].unsqueeze(2).expand(-1, n, -1)  # [B, n, 1]
        alpha_s = coefs['alpha_s'].unsqueeze(2).expand(-1, n, -1)
        delta = coefs['delta'].unsqueeze(2).expand(-1, n, -1)
        alpha_t = torch.reshape(alpha_t, (-1, alpha_t.size(-1)))  # [B*n, 1]
        alpha_s = torch.reshape(alpha_s, (-1, alpha_t.size(-1)))
        delta = torch.reshape(delta, (-1, delta.size(-1)))
        alpha_t_mask = alpha_t[z_mask, :]
        alpha_s_mask = alpha_s[z_mask, :]
        delta_mask = delta[z_mask, :]

        alpha_p = self.eta * delta_mask * target
        alpha_q = self.eta * delta_mask * preds
        beta_p = self.eta - self.eta * alpha_s_mask * target
        beta_q = self.eta - self.eta * alpha_s_mask * preds

        _alpha_p = self.eta * alpha_t_mask * target
        _alpha_q = self.eta * alpha_t_mask * preds
        _beta_p = self.eta - self.eta * alpha_t_mask * target
        _beta_q = self.eta - self.eta * alpha_t_mask * preds

        KLUB_conditional = (self.KL_gamma(alpha_q, alpha_p).clamp(0) \
                            + self.KL_gamma(beta_q, beta_p).clamp(0) \
                            - self.KL_gamma(alpha_q + beta_q, alpha_p + beta_p).clamp(0)).clamp(0)
        # KLUB_conditional = KLUB_conditional #* z_mask

        KLUB_marginal = (self.KL_gamma(_alpha_q, _alpha_p).clamp(0) \
                         + self.KL_gamma(_beta_q, _beta_p).clamp(0) \
                         - self.KL_gamma(_alpha_q + _beta_q, _alpha_p + _beta_p).clamp(0)).clamp(0)
        # KLUB_marginal = KLUB_marginal #* z_mask

        KLUB = (.99 * KLUB_conditional + .01 * KLUB_marginal)

        n_nan = 0
        if KLUB.isnan().any():
            pos = torch.where(KLUB.isnan())
            n_nan += pos[0].size(0)
            KLUB[pos] = 0.

        up_scale = (z_mask.numel() + n_nan) / z_mask.sum()

        loss = KLUB.sum()  # * up_scale

        self.total_klub += loss
        self.total_samples += preds.size(0)

    def compute(self):
        return self.total_klub / self.total_samples


class ProbabilityMetric(Metric):
    def __init__(self):
        """ This metric is used to track the marginal predicted probability of a class during training. """
        super().__init__()
        self.add_state('prob', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: Tensor) -> None:
        self.prob += preds.sum()
        self.total += preds.numel()

    def compute(self):
        return self.prob / self.total


class NLL(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_nll', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, batch_nll) -> None:
        self.total_nll += torch.sum(batch_nll)
        self.total_samples += batch_nll.numel()

    def compute(self):
        return self.total_nll / self.total_samples