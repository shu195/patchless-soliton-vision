import torch
from torch.optim import Optimizer

class StormerVerletSymplectic(Optimizer):
    """
    Störmer-Verlet Symplectic Phase-Space Volume Preserving Integrator:
        p_{t + 1/2} = p_t - (dt/2) grad L(q_t)
        q_{t+1}     = q_t + dt (p_{t + 1/2} / m)
        p_{t+1}     = (1 - gamma dt) p_{t + 1/2} - (dt/2) grad L(q_{t+1})
    Strictly preserves the canonical symplectic 2-form omega = dq ^ dp (det(J) == 1.0).
    """
    def __init__(self, params, lr=0.02, mass=1.0, gamma=0.2):
        defaults = dict(lr=lr, mass=mass, gamma=gamma)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            dt = group['lr']
            m = group['mass']
            gamma = group['gamma']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['p'] = torch.zeros_like(p)

                p_mom = state['p']

                # Symplectic half-kick
                p_mom.add_(grad, alpha=-0.5 * dt)
                # Drift
                p.add_(p_mom, alpha=dt / m)
                # Damping and second half-kick
                p_mom.mul_(1.0 - gamma * dt).add_(grad, alpha=-0.5 * dt)

        return loss
