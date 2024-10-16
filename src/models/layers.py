import torch
import torch.nn as nn


class Xtoy(nn.Module):
    def __init__(self, dx, dy):
        """ Map node features to global features """
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)

    def forward(self, X):
        """ X: bs, n, dx. """
        m = X.mean(dim=1)
        mi = X.min(dim=1)[0]
        ma = X.max(dim=1)[0]
        std = X.std(dim=1)
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out


class Etoy(nn.Module):
    def __init__(self, d, dy):
        """ Map edge features to global features. """
        super().__init__()
        self.lin = nn.Linear(4 * d, dy)

    def forward(self, E):
        """ E: bs, n, n, de
            Features relative to the diagonal of E could potentially be added.
        """
        m = E.mean(dim=(1, 2))
        mi = E.min(dim=2)[0].min(dim=1)[0]
        ma = E.max(dim=2)[0].max(dim=1)[0]
        std = torch.std(E, dim=(1, 2))
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out


def masked_softmax(x, mask, **kwargs):
    if mask.sum() == 0:
        return x
    x_masked = x.clone()
    x_masked[mask == 0] = -float("inf")
    return torch.softmax(x_masked, **kwargs)

def pow_tensor(x, cnum, from_logit=True):
    # x : B x N x N
    x_ = x.clone()
    xc = [x.unsqueeze(1)]
    for _ in range(cnum-1):
        deg_inv_sqrt = x.abs().sum(dim=-1).clamp(min=1).pow(-0.5)
        x_normalized = deg_inv_sqrt.unsqueeze(-1) * x * deg_inv_sqrt.unsqueeze(-2)
        x_ = torch.bmm(x_, x_normalized)
        mask = torch.ones([x.shape[-1], x.shape[-1]]) - torch.eye(x.shape[-1])
        x_ = x_ * mask.unsqueeze(0).to(x.device)
        xc.append(x_.unsqueeze(1))
    xc = torch.cat(xc, dim=1)
    return xc