from ._funcs import kernel_func

class BLOB(object):
    def __init__(self, opts, init_particles, init_mass) -> None:
        super().__init__()
        self.knType = opts.knType
        self.bwType = opts.bwType
        self.bwVal = opts.bwVal
        self.particles = init_particles
        self.mass = init_mass

    @classmethod
    def get_vector_field(cls, mass, grads, kernel, nabla_kernel):
        repulsive = (nabla_kernel * mass[None, :, None]).sum(1) / (kernel * mass[None, :]).sum(1, keepdim = True) + (nabla_kernel * mass[None, :, None] / (kernel * mass[None, :]).sum(1)[None,:, None]).sum(1)
        return grads - repulsive

    def one_step_update(self, step_size = None, grad_fn = None, **kw):
        grads = grad_fn(self.particles) * kw['annealing'] if 'annealing' in kw.keys() else grad_fn(self.particles)
        kernel, nabla_kernel, _ = kernel_func(self.particles, self.knType, self.bwType, self.bwVal, bw_only = None)
        self.particles += step_size * BLOB.get_vector_field(
            self.mass, grads, kernel, nabla_kernel
        )

    def get_state(self):
        return self.particles, self.mass