import torch
import os
import numpy as np
import ot
import matplotlib.pyplot as plt
from algorithms._funcs import safe_log

class single_gaussian(object):
    def __init__(self, opts, **kw) -> None:
        torch.set_default_dtype(torch.float64)
        self.device = opts.device
        self.particle_num = opts.particle_num
        # parameter setting
        self.model_dim, self.var, self.cov = opts.model_dim, 1.0, 0.8 * 1.0
        self.init_mu, self.init_std = 0, 0.5
        self.ref_particle_num = 5000 # number of particles to calculate 2-wasserstein distance
        self.plot_size, self.min_ratio, self.max_ratio = 50, 0.1, 10
        # model setting
        self.cov_matrix = self.cov * torch.ones((self.model_dim, self.model_dim)).to(self.device)
        self.cov_matrix = self.cov_matrix - torch.diag(torch.diag(self.cov_matrix)) + \
            torch.diag(self.var * torch.ones(self.model_dim).to(self.device))
        self.in_cov_matrix = torch.inverse(self.cov_matrix)
        self.in_cov_matrix_2d = torch.inverse(self.cov_matrix[0:2,0:2])
        self.mean = torch.zeros((1, self.model_dim)).to(self.device)
        p = np.random.multivariate_normal(
            mean = np.zeros(self.model_dim), 
            cov = self.cov_matrix.cpu().numpy(), 
            size = self.ref_particle_num 
        )
        self.ref_particles = torch.Tensor(p).to(self.device) # particles to calculate the 2-Wasserstein distance
        self.w2_value_list = [] # record the result
        self.particles_list = []
        self.mass_list = []

    @property
    @torch.no_grad()
    def init_particles(self):
        result = torch.randn(self.particle_num, self.model_dim, device = self.device) * self.init_std + self.init_mu
        return result

    # @torch.no_grad()
    # def pdf_calc(self, particles):  # M * dim matrix
    #     result = torch.exp( - (torch.matmul((particles-self.mean), self.in_cov_matrix) * (particles-self.mean)).sum(1) / 2)
    #     return result   # M array

    # @torch.no_grad()
    # def likelihood_calc(self, particles, features = None, labels = None): # len(particles.shape) must be 2
    #     results = self.pdf_calc(particles)
    #     return results

    @torch.no_grad()
    def potential(self, particles, features = None, labels = None):
        potential = (torch.matmul((particles-self.mean), self.in_cov_matrix) * (particles-self.mean)).sum(1) / 2
        return potential

    def grad_logp(self, particles, features = None, labels = None): # M * dim matrix
        result = - torch.matmul((particles - self.mean), self.in_cov_matrix)
        return result   # M*dim matrix

    @torch.no_grad()
    def evaluation(self, particles, mass, writer, logger, count: int, save_folder):
        cost_matrix  = (torch.cdist(particles, self.ref_particles) ** 2).cpu().numpy()
        # evaluate w2
        mass_numpy = mass.cpu().numpy().astype(np.float64) # from tensor to numpy, need extra normalization
        transport_plan = ot.emd(mass_numpy / mass_numpy.sum(), ot.unif(self.ref_particle_num), cost_matrix)
        w2_value = np.sqrt((cost_matrix * transport_plan).sum())
        self.particles_list.append(particles.cpu().numpy())
        self.mass_list.append(mass.cpu().numpy())
        self.w2_value_list.append(w2_value)
        writer.add_scalar('w2', self.w2_value_list[-1], global_step = count)
        logger.info('count: {}, w2: {:.2e}'.format(count, self.w2_value_list[-1]))

    def final_process(self, particles, mass, writer, logger, save_folder, isSave):
        device = particles.device
        if self.model_dim == 1:
            pass
        else:
            particles = particles[:, :2]   
        particles = particles.cpu().numpy()
        mass = mass.cpu().numpy()
        fig = self.plot_result(particles, mass, device = self.device)        
        plt.savefig(os.path.join(save_folder, 'figure.png'), pad_inches = 0.0)
        writer.add_figure(tag = 'samples', figure = fig)
        plt.close()
        # save the results
        np.save(os.path.join(save_folder, 'w2.npy'), np.array(self.w2_value_list))
        if isSave:
            np.save(os.path.join(save_folder, 'particles.npy'), np.array(self.particles_list))
            np.save(os.path.join(save_folder, 'mass.npy'), np.array(self.mass_list))

    def plot_result(self, particles, mass, device):
        # create mesh
        x, y = torch.linspace(-4, 4, 101, device = device), torch.linspace(-4, 4, 101, device = device)
        grid_X, grid_Y = torch.meshgrid(x, y, indexing = 'ij')
        loc = torch.cat([grid_X.unsqueeze(2), grid_Y.unsqueeze(2)], dim = 2)
        num = len(loc)
        log_pdf = - self.potential_2d(loc.view(-1, 2)).view(num, num)
        fig = plt.figure(figsize=(9.6, 4.8))
        # plot the particles for algorithms
        plt.subplot(121)
        mass = np.clip(mass * particles.shape[0], self.min_ratio, self.max_ratio)
        size_list = (mass * self.plot_size).astype(np.int)
        plt.scatter(particles[:, 0], particles[:, 1], alpha = 0.5, s = size_list, c = 'r',  zorder = 2)
        plt.contour(grid_X.cpu().numpy(), grid_Y.cpu().numpy(), log_pdf.cpu().numpy(), levels = 30,  zorder = 1)
        plt.ylim([-4, 4])
        plt.xlim([-4, 4])
        # plot the particles of reference
        plt.subplot(122)
        ref_particles_np = self.ref_particles.cpu().numpy()
        plt.scatter(ref_particles_np[:, 0], ref_particles_np[:, 1], alpha = 0.5, s = self.plot_size * self.min_ratio, c = 'r',  zorder = 2)
        plt.contour(grid_X.cpu().numpy(), grid_Y.cpu().numpy(), log_pdf.cpu().numpy(), levels = 30,  zorder = 1)
        plt.ylim([-4, 4])
        plt.xlim([-4, 4])
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        return fig

    def potential_2d(self, particles, features = None, labels = None):
        return (torch.matmul((particles - self.mean[:,:2]), self.in_cov_matrix_2d) * (particles - self.mean[:,:2])).sum(1) / 2
