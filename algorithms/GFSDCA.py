import torch
from algorithms.GFSD import GFSD
from ._funcs import kernel_func, safe_log, duplicate_kill_particles

class GFSDCA(GFSD):
    def __init__(self, opts, init_particles, init_mass) -> None:
        super().__init__(opts, init_particles, init_mass)

    @classmethod
    def get_avg_first_variation(cls, mass, potential, kernel):
        beta = safe_log((mass[None, :] * kernel).sum(1)) + potential
        beta_bar = beta - (beta * mass).sum()
        return beta_bar

    def one_step_update(self, step_size = None, alpha = None, grad_fn = None, potential_fn = None, kernel_fn = None, **kw):
        grads = grad_fn(self.particles) * kw['annealing'] if 'annealing' in kw.keys() else grad_fn(self.particles)
        potential = potential_fn(self.particles)
        kernel, nabla_kernel, _ = kernel_func(self.particles, self.knType, self.bwType, self.bwVal, bw_only = None)
        self.particles += step_size * GFSD.get_vector_field(
            self.mass, grads, kernel, nabla_kernel
        )
        self.mass *= 1 - step_size * alpha * GFSDCA.get_avg_first_variation(
            self.mass, potential, kernel
        )
        self.mass = self.mass / self.mass.sum() # eliminate numerical error
