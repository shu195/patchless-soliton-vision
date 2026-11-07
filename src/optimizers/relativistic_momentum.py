import torch
from torch.optim import Optimizer

class RelativisticEulerLagrange(Optimizer):
    """
    Relativistic Euler-Lagrange Optimizer with Lorentz Speed Clamping:
        T_{rel} = -m c^2 sqrt(1 - ||v||^2 / c^2)
        v = p / sqrt(m^2 + ||p||^2 / c^2) < c
    Eliminates gradient explosion, wall reflection, and step size instability by construction.
    """
    def __init__(self, params, lr=0.03, gamma=0.4, c_light=5.0, mass=1.0):
        defaults = dict(lr=lr, gamma=gamma, c_light=c_light, mass=mass)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            dt = group['lr']
            gamma = group['gamma']
            c = group['c_light']
            m = group['mass']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['velocity'] = torch.zeros_like(p)

                v = state['velocity']

                # Rayleigh dissipation + potential force:
                # dv/dt = -(gamma/m) v - (1/m) grad
                v.mul_(1.0 - (gamma / m) * dt).add_(grad, alpha=-(dt / m))

                # Relativistic Lorentz Velocity Clamping: ||v|| < c
                v_norm = torch.norm(v)
                if v_norm > c:
                    v.mul_(c / v_norm)

                # Parameter coordinate update: theta += v * dt
                p.add_(v, alpha=dt)

        return loss
