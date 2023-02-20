import torch
import os
import numpy as np
import ot
import matplotlib.pyplot as plt

class gaussian_process(object):
    def __init__(self, opts, **kw) -> None:
        super().__init__()
        self.device = opts.device
        self.particle_num = opts.particle_num
        self.model_dim = kw['info_source'].model_dim
        self.sigma = 0.2 # parameter of this gaussian process task
        self.plot_size, self.min_ratio, self.max_ratio = 50, 0.2, 4
        self.particles_list = []
        self.mass_list = []
        self.w2_value_list = []
        self.train_features = kw['info_source'].train_features
        self.train_labels = kw['info_source'].train_labels
        self.train_num = kw['info_source'].train_num
        self.sq_c00_dist = torch.cdist(self.train_features, self.train_features, p = 2).pow(2)
        self.diag_index = torch.linspace(0, self.train_num - 1, self.train_num, dtype = torch.long, device = self.device)
        if hasattr(opts, 'reference_path'):
            self.ref_particles = torch.load(opts.reference_path).to(self.device)

    @property
    @torch.no_grad()
    def init_particles(self):
        result = torch.randn(self.particle_num, self.model_dim, device = self.device) * 0.3 + torch.as_tensor([[0, -10]]).to(self.device)
        return result

    def potential(self, particles, features = None, labels = None):
        param_1 = particles[:,1].view(-1,1).unsqueeze(1)
        param_0 = particles[:,0].view(-1,1).unsqueeze(1)
        K_f = torch.exp(param_0) * torch.exp( - torch.exp(param_1) * self.sq_c00_dist.unsqueeze(0))
        K_y = K_f + (self.sigma**2) * torch.eye(self.train_num, device = self.device) # M * N * N tensor
        log_ll = -0.5 * (self.train_labels.t()@(torch.inverse(K_y)@self.train_labels)).squeeze() - 0.5 * torch.logdet(K_y)
        # prior
        log_pr = - torch.log(particles.pow(2) + 1).sum(1)
        log_po = log_ll + log_pr
        return  - log_po

    def grad_logp(self, particles, features = None, labels = None): # M * 2 matrix
        param_1 = particles[:,1].view(-1,1).unsqueeze(1)
        param_0 = particles[:,0].view(-1,1).unsqueeze(1)
        K_f = torch.exp(param_0) * torch.exp( - torch.exp(param_1) * self.sq_c00_dist.unsqueeze(0))
        dC0 = torch.exp(param_0) * torch.exp( - torch.exp(param_1) * self.sq_c00_dist.unsqueeze(0)) # M * N * N tensor
        dC1 = torch.exp(param_0) * torch.exp( - torch.exp(param_1) * self.sq_c00_dist.unsqueeze(0)) * \
            ( - torch.exp(param_1) * self.sq_c00_dist.unsqueeze(0)) # M * N * N tensor
        K_y = K_f + (self.sigma**2) * torch.eye(self.train_num, device = self.device) # M * N * N tensor
        alpha =  torch.inverse(K_y)@self.train_labels # M * N * 1 tensor
        temp = alpha * alpha.transpose(-1,-2) - torch.inverse(K_y)
        score_0 = 0.5 * (temp@dC0)[:, self.diag_index, self.diag_index].sum(1) - 2 * particles[:,0] / (1 + particles[:,0].pow(2))
        score_1 = 0.5 * (temp@dC1)[:, self.diag_index, self.diag_index].sum(1) - 2 * particles[:,1] / (1 + particles[:,1].pow(2))
        score = torch.cat([score_0.view(-1, 1), score_1.view(-1,1)], dim = 1)
        return score

    def evaluation(self, particles, mass, writer, logger, count: int, save_folder):
        cost_matrix  = (torch.cdist(particles, self.ref_particles) ** 2)
        # evaluate w2
        uniform = torch.ones(self.ref_particles.shape[0], device = mass.device, dtype = mass.dtype) / self.ref_particles.shape[0]
        transport_plan = ot.emd(mass, uniform, cost_matrix)
        w2_value = np.sqrt((cost_matrix * transport_plan).sum().item())
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
        x, y = torch.linspace(-5, 5, 100, device = device), torch.linspace(-13, -7, 100, device = device)
        grid_X, grid_Y = torch.meshgrid(x, y, indexing = 'ij')
        loc = torch.cat([grid_X.unsqueeze(2), grid_Y.unsqueeze(2)], dim = 2)
        num = len(loc)
        log_pdf = - self.potential(loc.view(-1, 2)).view(num, num)
        mass = np.clip(mass * particles.shape[0], self.min_ratio, self.max_ratio)
        size_list = (mass * self.plot_size).astype(np.int)
        if not hasattr(self, 'ref_particles'):
            fig = plt.figure(figsize=(4.8, 4.8))
            plt.scatter(particles[:, 0], particles[:, 1], alpha = 0.5, s = size_list, c = 'r',  zorder = 2)
            plt.contour(grid_X.cpu().numpy(), grid_Y.cpu().numpy(), log_pdf.cpu().numpy(), levels = 30,  zorder = 1)
            plt.ylim([ - 13, -7])
            plt.xlim([ - 5, 5])
        else:
            fig = plt.figure(figsize=(9.6, 4.8))
            plt.subplot(121)
            plt.scatter(particles[:, 0], particles[:, 1], alpha = 0.5, s = size_list, c = 'r',  zorder = 2)
            plt.contour(grid_X.cpu().numpy(), grid_Y.cpu().numpy(), log_pdf.cpu().numpy(), levels = 30,  zorder = 1)
            plt.ylim([ - 13, -7])
            plt.xlim([ - 5, 5])
            plt.subplot(122)
            plt.scatter(self.ref_particles.cpu().numpy()[:, 0], self.ref_particles.cpu().numpy()[:, 1], alpha = 0.5, s = self.plot_size * self.min_ratio, c = 'r',  zorder = 2)
            plt.contour(grid_X.cpu().numpy(), grid_Y.cpu().numpy(), log_pdf.cpu().numpy(), levels = 30,  zorder = 1)
            plt.ylim([ - 13, -7])
            plt.xlim([ - 5, 5])
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        return fig