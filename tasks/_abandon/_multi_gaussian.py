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
        self.model_dim, self.var, self.cov = 2, 2.5, 0.8 * 2.5 # variance and covariance of the gaussian distribution
        self.weight = torch.Tensor([1/2, 1/4, 1/4], device = self.device) # weight of mixture gaussian
        self.init_mu, self.init_std = 0.0, 1 # init distribution of particles
        self.ref_particle_num = 200 # number of particles to calculate 2-wasserstein distance
        self.plot_size, self.min_ratio, self.max_ratio = 50, 0.2, 4 # particle size to display
        # model setting
        self.g_means = torch.Tensor([[0,1.0],[-1.2,-1.0],[1.2,-1.0]], device = self.device) # \mu of gaussian mixture
        u, stretch, v = torch.linalg.svd(torch.Tensor([[self.var, self.cov],[self.cov, self.var]], device = self.device))
        self.cov_1 = torch.diag(stretch)
        self.cov_2 = torch.Tensor([[self.var, -self.cov], [-self.cov, self.var]], device = self.device)
        self.cov_3 = torch.Tensor([[self.var, self.cov], [self.cov, self.var]], device = self.device)
        self.g_covs = torch.cat([self.cov_1.unsqueeze(0), self.cov_2.unsqueeze(0), self.cov_3.unsqueeze(0)], dim = 0)
        self.inv_1, self.inv_2, self.inv_3 = torch.inverse(self.cov_1), torch.inverse(self.cov_2), torch.inverse(self.cov_3)
        self.g_invs = torch.cat([self.inv_1.unsqueeze(0), self.inv_2.unsqueeze(0), self.inv_3.unsqueeze(0)], dim = 0)
        # generate ref particles
        p = []
        for i in range(3): 
            p.append(np.random.multivariate_normal(
                mean = self.g_means[i].cpu().numpy(), 
                cov = self.g_covs[i].cpu().numpy(), 
                size = int(self.ref_particle_num * self.weight[i].item()) 
            ))
        p = np.concatenate(p, axis = 0)
        self.ref_samples = torch.Tensor(p).to(self.device) # particles to calculate the 2-Wasserstein distance
        self.record_particles = [] # record the evolution of particles
        self.record_mass = []
        self.w2_values = [] # record the result

    @torch.no_grad()
    def init_net(self, shape):
        result = torch.randn(shape, device = self.device) * self.init_std + self.init_mu
        return result

    @torch.no_grad()
    def pdf_calc(self, particles):  # M * D matrix
        result = self.weight[0] * torch.exp(-(torch.matmul(particles - self.g_means[0], self.g_invs[0]) * (particles - self.g_means[0])).sum(1) / 2) + \
            self.weight[1] * torch.exp(-(torch.matmul(particles - self.g_means[1], self.g_invs[1]) * (particles - self.g_means[1])).sum(1) / 2) + \
            self.weight[2] * torch.exp(-(torch.matmul(particles - self.g_means[2], self.g_invs[2]) * (particles - self.g_means[2])).sum(1) / 2)
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
        part1 = self.weight[0] * torch.exp(-(torch.matmul(particles - self.g_means[0], self.g_invs[0]) * (particles - self.g_means[0])).sum(1) / 2).unsqueeze(1)
        part2 = self.weight[1] * torch.exp(-(torch.matmul(particles - self.g_means[1], self.g_invs[1]) * (particles - self.g_means[1])).sum(1) / 2).unsqueeze(1)
        part3 = self.weight[2] * torch.exp(-(torch.matmul(particles - self.g_means[2], self.g_invs[2]) * (particles - self.g_means[2])).sum(1) / 2).unsqueeze(1)
        result = torch.matmul((particles - self.g_means[0]), self.g_invs[0]) / ((part1 + part2 + part3) / part1) + \
            torch.matmul((particles - self.g_means[1]), self.g_invs[1]) / ((part1 + part2 + part3) / part2) + \
            torch.matmul((particles - self.g_means[2]), self.g_invs[2]) / ((part1 + part2 + part3) / part3)
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
        # calc mesh
        x, y = torch.linspace(-6, 6, 121, device = self.device), torch.linspace(-6, 6, 121, device = self.device)
        grid_X, grid_Y = torch.meshgrid(x,y)
        loc = torch.cat([grid_X.unsqueeze(2), grid_Y.unsqueeze(2)], dim = 2)
        num = len(loc)
        pdf = self.pdf_calc(loc.view(-1, 2))
        pdf = pdf.view(num, num)
        fig = plt.figure(num = 1)
        plt.contour(grid_X.cpu().numpy(), grid_Y.cpu().numpy(), pdf.cpu().numpy())
        # set particle size
        mass = np.clip(mass * self.opts.particle_num, self.min_ratio, self.max_ratio)
        size_list = (mass * self.plot_size).astype(np.int)
        plt.scatter(particles[:, 0], particles[:, 1], alpha = 0.5, s = size_list, c = 'r')
        # plt.scatter(particles[:, 0], particles[:, 1], alpha = 0.5, s = 15) # debug for testing ref particles
        plt.ylim([ - 6, 6])
        plt.xlim([ - 6, 6])
        plt.tight_layout()
        writer.add_figure(tag = 'samples', figure = fig, global_step = curr_iter_count)
        plt.close()

    def save_final_results(self, writer, save_folder, results):
        particles = np.array(self.record_particles)
        mass = np.array(self.record_mass)
        if self.opts.save_particles == True:
            np.save(os.path.join(save_folder, 'particles.npy'), particles)
            np.save(os.path.join(save_folder, 'mass.npy'), mass)
        np.save(os.path.join(save_folder, 'w2_values.npy'), np.array(self.w2_values))