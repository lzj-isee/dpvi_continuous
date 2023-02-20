import torch
from algorithms.KSDD import KSDD
from ._funcs import kernel_func, gaussian_stein_kernel

class KSDDCA(KSDD):
    def __init__(self, opts, init_particles, init_mass) -> None:
        super().__init__(opts, init_particles, init_mass)
        
    @classmethod
    def get_avg_first_variation(cls, particles, mass, grads, bw_h):
        score = grads
        with torch.no_grad(): ksd_matrix = gaussian_stein_kernel(particles, particles, score, score, bw_h)
        beta = torch.matmul(mass, ksd_matrix)
        beta_bar = beta - (beta * mass).sum()
        return beta_bar

    def one_step_update(self, step_size = None, alpha = None, grad_fn = None, **kw):
        self.particles.requires_grad = True
        grads = grad_fn(self.particles) * kw['annealing'] if 'annealing' in kw.keys() else grad_fn(self.particles)
        _, _, bw_h = kernel_func(self.particles, self.knType, self.bwType, self.bwVal, bw_only = True)
        vector_field = KSDD.get_vector_field(self.particles, self.mass, grads, bw_h)
        with torch.no_grad():
            self.particles -= step_size * vector_field
        self.particles.requires_grad = False
        self.particles.grad.zero_()
        self.mass *= 1 - step_size * alpha * KSDDCA.get_avg_first_variation(
            self.particles, self.mass, grads, bw_h
        )
        self.mass = self.mass / self.mass.sum()

