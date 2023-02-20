from dataloader import myDataLoader
import torch
import common.utils as utils
import numpy as np
import tasks
import os
from scipy import signal

class functions(myDataLoader):
    def __init__(self, opts) -> None:
        super().__init__(opts)
        self.p = 2
        self.model_dim = self.p**2
        self.weight_decay = 1e-5
        self.init_std = 1
        #self.W = torch.randn(self.p, self.p, device = self.device)
        time = np.linspace(0, 8, self.opts.train_num)
        s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
        s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
        s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal
        S = np.c_[s1, s2]
        S += 0.2 * np.random.normal(size = S.shape)  # Add noise
        self.S = torch.from_numpy(S.astype(np.float32)).to(self.device).t()
        self.A = torch.Tensor([[1,1],[0.5,2]]).to(self.device)
        self.W = torch.inverse(self.A)
        self.train_features = torch.matmul(self.A, self.S)
        self.amari_dists, self.ksds = [], []

    @torch.no_grad()
    def init_net(self, shape):
        init = torch.randn(shape, device = self.device) * self.init_std
        # ica = FastICA(random_state = self.opts.seed)
        # _ = ica.fit_transform(self.train_features.t().cpu().numpy())
        # A_ =  torch.from_numpy(ica.mixing_.astype(np.float32)).to(self.device)
        # init_w = torch.inverse(A_)
        # inits = torch.randn(shape, device = self.device) * self.init_std + init_w.reshape(1,-1)
        return init
    
    #------------------------------------------ likelihood and grad ------------------------------------------------------------------

    @torch.no_grad()
    def likelihood_calc(self, particles, X): # len(particles.shape) must be
        par_list = particles.reshape(particles.shape[0], self.p, self.p)
        results = torch.det(par_list).abs().view(-1,1) * (1 / 4 / torch.cosh(par_list.matmul(X) / 2)**2).prod(dim = 1)
        return results

    @torch.no_grad()
    def nll_calc(self, particles, X):
        par_list = particles.reshape(particles.shape[0], self.p, self.p)
        log_ll = torch.log(torch.det(par_list).abs()+1e-6).view(-1,1) + 2 * torch.log(1 / torch.cosh(par_list.matmul(X) / 2) + 1e-6).sum(dim = 1)
        return -log_ll

    @torch.no_grad()
    def potential_calc(self, particles, features, labels):
        return self.nll_calc(particles, self.train_features).mean(1) + 0.5 * np.log(2 * np.pi / self.weight_decay) + particles.pow(2).sum(1) * self.weight_decay / 2

    
    def nl_grads_calc(self, particles, features, labels): # len(particles.shape) mush be 2
        par_list = particles.reshape(particles.shape[0], self.p, self.p)
        N = self.train_features.shape[1]
        psi = torch.tanh(par_list.matmul(self.train_features) / 2).matmul(self.train_features.t()) / N - torch.inverse(par_list).transpose(-1, -2)
        sc = - psi - self.weight_decay * par_list
        grads = - sc
        return grads.reshape(particles.shape[0], self.p**2)

    def amari_distance(self, pars, A):
        p = torch.matmul(pars, A.unsqueeze(0))
        def s(r): # M * p * p dim
            return torch.sum(torch.sum(r, dim = 2) / torch.max(r, dim = 2)[0] -1, dim = 1)
        return (s(torch.abs(p)) + s(torch.abs(p.transpose(-1,-2)))) / (2 * self.p)


    @torch.no_grad()
    def evaluation(self, particles, mass):
        particles_m = particles.reshape(particles.shape[0], self.p, self.p)
        nll = self.nll_calc(particles, self.train_features)
        amari_dists = self.amari_distance(particles_m, self.A)
        avg_amari_dists = torch.matmul(amari_dists, mass)
        nll = torch.matmul(mass, nll).mean()
        sq_distance = torch.cdist(particles, particles, p = 2) ** 2
        med_dis = torch.median(sq_distance + 1e-5)
        ksds = tasks.ksd_matrix(particles, score =  - self.nl_grads_calc(
            particles, features = self.train_features, labels = None), 
            bandwidth_h = self.opts.ksd_h, sq_distance = sq_distance)
        ksd_value = torch.matmul(torch.matmul(mass, ksds), mass).sqrt()
        mass_min = mass.min() * self.opts.particle_num
        return {'avg_amari': avg_amari_dists.item(), 'nll': nll.item(), 'ksd': ksd_value.item(), 'med_dis': med_dis.item(), 'mass_min': mass_min.item()}

    #------------------------------------ save the results ----------------------------------------------------------------------------

    def save_final_results(self, writer, save_folder, result_dict):
        utils.save_final_results(save_folder, result_dict)
        np.save(os.path.join(save_folder, 'amari_dist.npy'), np.array(self.amari_dists))
        np.save(os.path.join(save_folder, 'ksd.npy'), np.array(self.ksds))

    def save_eval_to_tensorboard(self, writer, results, global_step):
        writer.add_scalar('avg_amari', results['avg_amari'], global_step = global_step)
        writer.add_scalar('nll', results['nll'], global_step = global_step)
        writer.add_scalar('ksd', results['ksd'], global_step = global_step)
        writer.add_scalar('med dis', results['med_dis'], global_step = global_step)
        writer.add_scalar('mass_min', results['mass_min'], global_step = global_step)
        self.amari_dists.append(results['avg_amari'])
        self.ksds.append(results['ksd'])