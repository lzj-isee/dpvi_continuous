import torch
import numpy as np

class SGLD(object):
    def __init__(self, opts, init_particles, init_mass) -> None:
        super().__init__()
        self.particles = init_particles
        self.mass = init_mass

    def one_step_update(self, step_size = None, grad_fn = None, **kw):
        grads = grad_fn(self.particles) * kw['annealing'] if 'annealing' in kw.keys() else grad_fn(self.particles)
        self.particles += step_size * grads + torch.randn_like(self.particles) * np.sqrt(2 * step_size)

    def get_state(self):
        return self.particles, self.mass