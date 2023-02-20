import torch
from algorithms.GFSD import GFSD
from algorithms.GFSDCA import GFSDCA
from ._funcs import kernel_func, safe_log, duplicate_kill_particles
import numpy as np

class GFSDDK(GFSD):
    def __init__(self, opts, init_particles, init_mass) -> None:
        super().__init__(opts, init_particles, init_mass)

    def one_step_update(self, step_size = None, alpha = None, grad_fn = None, potential_fn = None, **kw):
        grads = grad_fn(self.particles) * kw['annealing'] if 'annealing' in kw.keys() else grad_fn(self.particles)
        potential = potential_fn(self.particles)
        kernel, nabla_kernel, _ = kernel_func(self.particles, self.knType, self.bwType, self.bwVal, bw_only = None)
        self.particles += step_size * GFSD.get_vector_field(
            self.mass, grads, kernel, nabla_kernel
        )
        avg_first_variation = GFSDCA.get_avg_first_variation(self.mass, potential, kernel)
        prob_list = 1 - torch.exp( - avg_first_variation.abs() * alpha * step_size)
        self.particles = duplicate_kill_particles(prob_list, avg_first_variation > 0, self.particles, noise_amp = np.sqrt(2 * step_size))
