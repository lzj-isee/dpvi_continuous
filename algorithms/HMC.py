import torch
import numpy as np

class HMC(object):
    def __init__(self,opts, particles, mass) -> None:
        super().__init__()
        self.particles = particles
        self.mass = mass
        self.accu_accept_ratio = 0

    def one_step_update(self, step_size = None, leap_frog_num = None, grad_fn = None, potential_fn = None, **kw):
        q = self.particles.clone()
        velocity = torch.randn_like(self.particles)
        p = velocity.clone()
        grads = grad_fn(q) * kw['annealing'] if 'annealing' in kw.keys() else grad_fn(q)
        p = p + 0.5 * step_size * grads
        for k in range(leap_frog_num):
            q = q + step_size * p
            grads = grad_fn(q) * kw['annealing'] if 'annealing' in kw.keys() else grad_fn(q)
            if k != (leap_frog_num - 1): p = p + step_size * grads
        p = p + 0.5 * step_size * grads
        p = -p
        curr_u = potential_fn(self.particles)
        curr_k = velocity.pow(2).sum(1) / 2
        prop_u = potential_fn(q)
        prop_k = p.pow(2).sum(1) / 2
        accept_prob = torch.minimum(torch.exp(curr_u + curr_k - prop_u - prop_k), torch.ones(q.shape[0], device = q.device))
        rand = torch.rand(q.shape[0], device = q.device)
        self.particles[rand < accept_prob] = q[rand < accept_prob].clone()
        self.accu_accept_ratio += accept_prob.mean()

    def get_state(self):
        return self.particles, self.mass

    def get_accept_ratio(self):
        return self.accu_accept_ratio