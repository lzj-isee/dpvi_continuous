import torch
import os
import numpy as np
import ot
import matplotlib.pyplot as plt
from algorithms._funcs import safe_log
from torch.utils.data import WeightedRandomSampler

class demo(object):
    def __init__(self, opts, **kw) -> None:
        torch.set_default_dtype(torch.float64)
        self.device = opts.device
        self.particle_num = opts.particle_num
        # parameter setting
        self.model_dim = 2
        self.std = 1
        self.init_mu, self.init_std = 0, 0.5 # init distribution of particles
        self.ref_particle_num = 5000 # number of particles to calculate 2-wasserstein distance
        self.plot_size, self.min_ratio, self.max_ratio = 50, 0.1, 10 # particle size to display
        # generate reference
        self.weight = torch.Tensor([3, 1, 2]).to(self.device) # weight of mixture gaussian, 3 mode
        self.weight = self.weight / self.weight.sum()
        index = torch.as_tensor(list(WeightedRandomSampler(self.weight, self.ref_particle_num, replacement = True)), device = self.device)
        self.skew = 5.5
        self.ref_particles = torch.randn(self.ref_particle_num, self.model_dim, device = self.device) * self.std
        self.ref_particles = torch.cat([self.ref_particles[index == 0] - self.skew, self.ref_particles[index == 1], self.ref_particles[index == 2] + self.skew], dim = 0)
        self.w2_value_list = [] # record the result
        self.particles_list = []
        self.mass_list = []

    @property
    @torch.no_grad()
    def init_particles(self):
        pars = torch.randn(self.particle_num, self.model_dim, device = self.device) * self.init_std
        return pars

    def grad_logp(self, particles, features = None, labels = None):
        part0 = self.weight[0] * torch.exp(-(particles + self.skew).pow(2).sum(1) / 2 / self.std**2)
        part1 = self.weight[1] * torch.exp(-(particles).pow(2).sum(1) / 2 / self.std**2)
        part2 = self.weight[2] * torch.exp(-(particles - self.skew).pow(2).sum(1) / 2 / self.std**2)
        result = - (particles + self.skew) / self.std**2 / ((part0 + part1 + part2) / part0)[:, None] - particles / self.std**2 / ((part0 + part1 + part2) / part1)[:, None] - (particles - self.skew) / self.std**2 / ((part0 + part1 + part2) / part2)[:, None]
        if torch.isnan(result).sum() > 0:
            raise ValueError('Nan')
        return result   # M * dim matrix

    def potential(self, particles, features = None, labels = None):
        part0 = self.weight[0] * torch.exp(-(particles + self.skew).pow(2).sum(1) / 2 / self.std**2)
        part1 = self.weight[1] * torch.exp(-(particles).pow(2).sum(1) / 2 / self.std**2)
        part2 = self.weight[2] * torch.exp(-(particles - self.skew).pow(2).sum(1) / 2 / self.std**2)
        results = - safe_log(part0 + part1 + part2)
        if torch.isnan(results).sum() > 0: raise ValueError('Nan')
        return results

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
        # only for debug
        # if self.model_dim == 1:
        #     pass
        # else:
        #     particles = particles[:, :2]   
        # particles = particles.cpu().numpy()
        # mass = mass.cpu().numpy()
        # fig = self.plot_result(particles, mass, device = self.device)        
        # plt.savefig(os.path.join(save_folder, 'figure_%d.png'%count), pad_inches = 0.0)
        # plt.close()

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
        xlim, ylim = 10, 10
        # create mesh
        x, y = torch.linspace(-xlim, xlim, 200, device = device), torch.linspace(-ylim, ylim, 200, device = device)
        grid_X, grid_Y = torch.meshgrid(x, y, indexing = 'ij')
        loc = torch.cat([grid_X.unsqueeze(2), grid_Y.unsqueeze(2)], dim = 2)
        num = len(loc)
        log_pdf = - self.potential(loc.view(-1, 2)).view(num, num)
        log_pdf = log_pdf.exp()
        fig = plt.figure(figsize=(9.6, 4.8))
        # plot the particles for algorithms
        plt.subplot(121)
        mass = np.clip(mass * particles.shape[0], self.min_ratio, self.max_ratio)
        size_list = (mass * self.plot_size).astype(np.int)
        plt.scatter(particles[:, 0], particles[:, 1], alpha = 0.5, s = size_list, c = 'r',  zorder = 2)
        plt.contour(grid_X.cpu().numpy(), grid_Y.cpu().numpy(), log_pdf.cpu().numpy(), levels = 10,  zorder = 1)
        plt.ylim([-ylim, ylim])
        plt.xlim([-xlim, xlim])
        # plot the particles of reference
        plt.subplot(122)
        ref_particles_np = self.ref_particles.cpu().numpy()
        plt.scatter(ref_particles_np[:, 0], ref_particles_np[:, 1], alpha = 0.5, s = self.plot_size * self.min_ratio, c = 'r',  zorder = 2)
        plt.contour(grid_X.cpu().numpy(), grid_Y.cpu().numpy(), log_pdf.cpu().numpy(), levels = 10,  zorder = 1)
        plt.ylim([-ylim, ylim])
        plt.xlim([-xlim, xlim])
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        return fig
        