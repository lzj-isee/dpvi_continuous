from dataloader import myDataLoader
import torch
import os
import numpy as np
import ot
import matplotlib.pyplot as plt

class functions(myDataLoader):
    def __init__(self, opts) -> None:
        super().__init__(opts)
        # parameter setting
        self.model_dim, self.var, self.cov = 10, 6, 0.9 * 6
        self.init_mu, self.init_std = 0, 0.5
        self.ref_particle_num = 20000 # number of particles to calculate 2-wasserstein distance
        self.plot_size, self.min_ratio, self.max_ratio = 50, 0.2, 4
        # model setting
        self.cov_matrix = self.cov * torch.ones((self.model_dim, self.model_dim)).to(self.device)
        self.cov_matrix = self.cov_matrix - torch.diag(torch.diag(self.cov_matrix)) + \
            torch.diag(self.var * torch.ones(self.model_dim).to(self.device))
        self.in_cov_matrix = torch.inverse(self.cov_matrix)
        # self.in_cov_matrix_2dim = torch.inverse(self.cov_matrix[0:2,0:2])
        self.mean = torch.zeros((1, self.model_dim)).to(self.device)
        p = np.random.multivariate_normal(
            mean = np.zeros(self.model_dim), 
            cov = self.cov_matrix.cpu().numpy(), 
            size = self.ref_particle_num 
        )
        self.ref_samples = torch.Tensor(p).to(self.device) # particles to calculate the 2-Wasserstein distance
        self.record_particles = [] # record the evolution of particles
        self.record_mass = []
        self.w2_values = [] # record the result

    @torch.no_grad()
    def init_net(self, shape):
        result = torch.randn(shape, device = self.device) * self.init_std + self.init_mu
        return result

    @torch.no_grad()
    def pdf_calc(self, particles):  # M * dim matrix
        result = torch.exp( - (torch.matmul((particles-self.mean), self.in_cov_matrix) * (particles-self.mean)).sum(1) / 2)
        return result   # M array

    @torch.no_grad()
    def likelihood_calc(self, particles, features = None, labels = None): # len(particles.shape) must be 2
        results = self.pdf_calc(particles)
        return results

    @torch.no_grad()
    def potential_calc(self, particles, features, labels):
        potential = - torch.log(self.likelihood_calc(particles, features, labels) + 1e-6)
        return potential

    
    def nl_grads_calc(self, particles, features = None, labels = None): # M * dim matrix
        result = torch.matmul((particles - self.mean), self.in_cov_matrix)
        return result   # M*dim matrix

    @torch.no_grad()
    def evaluation(self, particles, mass):
        cost_matrix  = (torch.cdist(particles, self.ref_samples) ** 2).cpu().numpy()
        # evaluate w2
        mass_numpy = mass.cpu().numpy().astype(np.float64) # from tensor to numpy, need extra normalization
        transport_plan = ot.emd(mass_numpy / mass_numpy.sum(), ot.unif(self.ref_particle_num), cost_matrix)
        self.w2_value = np.sqrt((cost_matrix * transport_plan).sum())
        self.record_particles.append(particles.cpu().numpy())
        self.record_mass.append(mass.cpu().numpy())
        return [particles, mass]

    def save_eval_to_tensorboard(self, writer, results, curr_iter_count):
        particles, mass = results[0].cpu().numpy(), results[1].cpu().numpy()
        # particles = self.ref_samples.cpu().numpy()   # debug
        writer.add_scalar('W2', self.w2_value, curr_iter_count)
        self.w2_values.append(self.w2_value)
        # # calc mesh
        # x, y = torch.linspace(-6, 6, 121, device = self.device), torch.linspace(-6, 6, 121, device = self.device)
        # grid_X, grid_Y = torch.meshgrid(x,y)
        # loc = torch.cat([grid_X.unsqueeze(2), grid_Y.unsqueeze(2)], dim = 2)
        # num = len(loc)
        # pdf = self.pdf_calc(loc.view(-1, 2))
        # pdf = pdf.view(num, num)
        # fig = plt.figure(num = 1)
        # plt.contour(grid_X.cpu().numpy(), grid_Y.cpu().numpy(), pdf.cpu().numpy())
        # # set particle size
        # mass = np.clip(mass * self.opts.particle_num, self.min_ratio, self.max_ratio)
        # size_list = (mass * self.plot_size).astype(np.int)
        # plt.scatter(particles[:, 0], particles[:, 1], alpha = 0.5, s = size_list, c = 'r')
        # # plt.scatter(particles[:, 0], particles[:, 1], alpha = 0.5, s = 15) # debug for testing ref particles
        # plt.ylim([ - 6, 6])
        # plt.xlim([ - 6, 6])
        # plt.tight_layout()
        # writer.add_figure(tag = 'samples', figure = fig, global_step = curr_iter_count)
        # plt.close()

    def save_final_results(self, writer, save_folder, results):
        particles = np.array(self.record_particles)
        mass = np.array(self.record_mass)
        np.save(os.path.join(save_folder, 'w2_values.npy'), np.array(self.w2_values))