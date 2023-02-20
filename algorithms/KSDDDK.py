import torch
import numpy as np
from algorithms.KSDD import KSDD
from algorithms.KSDDCA import KSDDCA
from ._funcs import kernel_func, duplicate_kill_particles

class KSDDDK(KSDD):
    def __init__(self, opts, init_particles, init_mass) -> None:
        super().__init__(opts, init_particles, init_mass)

    def one_step_update(self, step_size = None, alpha = None, grad_fn = None, **kw):
        self.particles.requires_grad = True
        grads = grad_fn(self.particles) * kw['annealing'] if 'annealing' in kw.keys() else grad_fn(self.particles)
        _, _, bw_h = kernel_func(self.particles, self.knType, self.bwType, self.bwVal, bw_only = True)
        vector_field = KSDD.get_vector_field(self.particles, self.mass, grads, bw_h)
        with torch.no_grad():
            self.particles -= step_size * vector_field
        self.particles.requires_grad = False
        self.particles.grad.zero_()
        avg_first_variation = KSDDCA.get_avg_first_variation(self.particles, self.mass, grads, bw_h)
        prob_list = 1 - torch.exp( - avg_first_variation.abs() * alpha * step_size)
        if alpha == 0: return # keep the random seed consistent
        self.particles = duplicate_kill_particles(prob_list, avg_first_variation > 0, self.particles, noise_amp = np.sqrt(2 * step_size))
        

