import torch
from ._funcs import kernel_func, gaussian_stein_kernel

class KSDD(object):
    def __init__(self, opts, init_particles, init_mass) -> None:
        super().__init__()
        self.knType = 'rbf'
        self.bwType = 'fix'
        self.bwVal = opts.bwVal
        self.particles = init_particles
        self.mass = init_mass

    @classmethod
    def get_vector_field(cls, particles: torch.Tensor, mass, grads, bw_h):
        score = grads
        g_kernel = gaussian_stein_kernel(particles.detach(), particles, score.detach(), score, bw_h)
        loss = torch.matmul(mass, g_kernel).sum()
        loss.backward()
        return particles.grad
        
    def one_step_update(self, step_size = None, grad_fn = None, **kw):
        self.particles.requires_grad = True
        grads = grad_fn(self.particles) * kw['annealing'] if 'annealing' in kw.keys() else grad_fn(self.particles)
        _, _, bw_h = kernel_func(self.particles, self.knType, self.bwType, self.bwVal, bw_only = True)
        vector_field = KSDD.get_vector_field(self.particles, self.mass, grads, bw_h)
        with torch.no_grad():
            self.particles -= step_size * vector_field
        self.particles.requires_grad = False
        self.particles.grad.zero_()

    def get_state(self):
        return self.particles, self.mass