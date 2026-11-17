import torch


def grad_norm(loss, params):
    """L2 norm of d(loss)/d(params), keeping the graph for the training backward pass."""
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    sq = [g.pow(2).sum() for g in grads if g is not None]
    return torch.stack(sq).sum().sqrt().item() if sq else 0.0


class LossBalancer:
    """Adaptive weight for a representation loss that shares the phi network with the TD loss.

    With back_critic_grad, phi is trained by the representation loss and the TD loss together.
    The TD gradient on phi grows over training while the representation gradient does not, so a
    fixed weight lets the TD loss take over phi. Every `every` steps, `update` measures both
    gradient norms on the phi parameters and smooths them with an EMA (`ema`). It returns the
    weight ratio * |g_td| / |g_rep|, clipped to `clip`, which holds the weighted representation
    gradient at `ratio` times the TD gradient. Off when ratio <= 0.

    Config keys (all optional): balance_ratio (0 = off), balance_every (50), balance_ema (0.95),
    balance_range ([1, 1000])."""

    def __init__(self, ratio=0.0, every=50, ema=0.95, clip=(1.0, 1000.0)):
        self.ratio, self.every, self.ema = ratio, every, ema
        self.lo, self.hi = clip
        self.g_rep = self.g_td = None

    @classmethod
    def from_cfg(cls, cfg):
        return cls(
            ratio=cfg.get("balance_ratio", 0.0),
            every=cfg.get("balance_every", 50),
            ema=cfg.get("balance_ema", 0.95),
            clip=tuple(cfg.get("balance_range", [1.0, 1000.0])),
        )

    def due(self, step):
        return self.ratio > 0 and step % self.every == 0

    def update(self, rep_loss, td_loss, params):
        """Returns (weight, |g_rep|, |g_td|), the gradient norms being this step's raw values."""
        g_rep, g_td = grad_norm(rep_loss, params), grad_norm(td_loss, params)
        if self.g_rep is None:
            self.g_rep, self.g_td = g_rep, g_td
        else:
            b = self.ema
            self.g_rep = b * self.g_rep + (1 - b) * g_rep
            self.g_td = b * self.g_td + (1 - b) * g_td
        weight = float(min(max(self.ratio * self.g_td / max(self.g_rep, 1e-8), self.lo), self.hi))
        return weight, g_rep, g_td
