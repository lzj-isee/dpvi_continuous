from dataloader import myDataLoader
import torch
import common.utils as utils
import numpy as np
import tasks
import os
from scipy import signal
from sklearn.decomposition import FastICA

class functions(myDataLoader):
    def __init__(self, opts) -> None:
        super().__init__(opts)
        self.p = self.opts.dim
        self.weight_decay = 1e-5
        self.init_std = 2 
        self.test_nll = []

    @torch.no_grad()
    def init_net(self, shape):
        #init = torch.cat([torch.ones(self.p, device = self.device).diag() * torch.randn(self.p, device = self.device).abs() for i in range(shape[0])], dim = 0).reshape(shape[0], shape[1]) * self.init_std
        inits = torch.randn(shape, device = self.device) * self.init_std
        # ica = FastICA(random_state = self.opts.seed)
        # _ = ica.fit_transform(self.train_features.cpu().numpy())
        # A_ =  torch.from_numpy(ica.mixing_.astype(np.float32)).to(self.device)
        # init_w = torch.inverse(A_)
        # inits = torch.randn(shape, device = self.device) * self.init_std + init_w.reshape(1,-1)
        return inits
    
    #------------------------------------------ likelihood and grad ------------------------------------------------------------------

    @torch.no_grad()
    def likelihood_calc(self, particles, X): # len(particles.shape) must be
        par_list = particles.reshape(particles.shape[0], self.p, self.p)
        results = torch.det(par_list).abs().view(-1,1) * (1 / (4 * torch.cosh(par_list.matmul(X) / 2)**2)).prod(dim = 1)
        return results

    @torch.no_grad()
    def nll_calc(self, particles, X):
        par_list = particles.reshape(particles.shape[0], self.p, self.p)
        log_ll = torch.log(torch.det(par_list).abs()+1e-6).view(-1,1) + \
            ( - np.log(4) + 2 * torch.log(1 / torch.cosh(par_list.matmul(X) / 2) + 1e-6)).sum(dim = 1)
        return -log_ll

    @torch.no_grad()
    def potential_calc(self, particles, features, labels):
        data = self.train_features.t()
        return self.nll_calc(particles, data).mean(1) + particles.pow(2).sum(1) * self.weight_decay / 2

    
    def nl_grads_calc(self, particles, features, labels): # len(particles.shape) mush be 2
        data = self.train_features.t()
        par_list = particles.reshape(particles.shape[0], self.p, self.p)
        N = data.shape[1]
        psi = torch.tanh(par_list.matmul(data) / 2).matmul(data.t()) / N - torch.inverse(par_list).transpose(-1, -2)
        sc = - psi - self.weight_decay * par_list
        grads = - sc
        return grads.reshape(particles.shape[0], self.p**2)

    @torch.no_grad()
    def evaluation(self, particles, mass):
        train_nlls = self.nll_calc(particles, self.train_features.t())
        test_nlls = self.nll_calc(particles, self.test_features.t())
        train_nll_avg = torch.matmul(mass, train_nlls).mean()
        test_nll_avg = torch.matmul(mass, test_nlls).mean()
        sq_distance = torch.cdist(particles, particles, p = 2) ** 2
        med_dis = torch.median(sq_distance + 1e-5)
        mass_min = mass.min() * self.opts.particle_num
        return {'train_nll': train_nll_avg.item(), 'test_nll': test_nll_avg.item(), 'med_dis': med_dis.item(), 'mass_min': mass_min.item()}

    #------------------------------------ save the results ----------------------------------------------------------------------------

    def save_final_results(self, writer, save_folder, result_dict):
        utils.save_final_results(save_folder, result_dict)
        np.save(os.path.join(save_folder, 'test_nll.npy'), np.array(self.test_nll))

    def save_eval_to_tensorboard(self, writer, results, global_step):
        writer.add_scalar('train_nll', results['train_nll'], global_step = global_step)
        writer.add_scalar('test_nll', results['test_nll'], global_step = global_step)
        writer.add_scalar('med dis', results['med_dis'], global_step = global_step)
        writer.add_scalar('mass_min', results['mass_min'], global_step = global_step)
        self.test_nll.append(results['test_nll'])