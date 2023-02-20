from dataloader import myDataLoader
import torch
import common.utils as utils
import numpy as np
import tasks
from tqdm import tqdm
import os
import ot
# we use full batch in this experiment

class functions(myDataLoader):
    def __init__(self, opts) -> None:
        super().__init__(opts)
        self.weight_decay = 1
        self.hmc_particle_num, self.hmc_inner_iter, self.hmc_outer_iter, self.hmc_burn_in = 50, 50, 200, 2000
        self.hmc_gd_init_iter = 5000
        self.hmc_eval_interval = 10
        if not os.path.exists('./hmc_reference/{}.pth'.format(opts.dataset)):
            self.hmc()
        self.state = torch.load('./hmc_reference/{}.pth'.format(opts.dataset))
        self.posterior_mean = self.state['particles'].mean(0)
        self.w2, self.ksd = [], []

    @torch.no_grad()
    def init_net(self, shape):
        net = torch.randn(shape, device = self.device) / np.sqrt(self.model_dim)
        # net = self.state['particles'][:self.opts.particle_num, :].clone()
        return net
    
    #------------------------------------------ likelihood and grad ------------------------------------------------------------------

    @torch.no_grad()
    def likelihood_calc(self, particles, features, labels): # len(particles.shape) must be 2
        logits = torch.matmul(features, particles.t()) * labels.view(-1, 1) # B*M matrix
        results = torch.sigmoid(logits).t() # M*B matrix 
        return results

    @torch.no_grad()
    def potential_calc(self, particles, features, labels):
        potential = - torch.log(self.likelihood_calc(particles, self.train_features, self.train_labels) + 1e-6).sum(1)
        potential += 0.5 * np.log(2 * np.pi / self.weight_decay) + particles.pow(2).sum(1) * self.weight_decay / 2
        return potential

    
    def nl_grads_calc_batch(self, particles, features, labels): # len(particles.shape) mush be 2
        batch_size = len(labels)
        logits = - torch.matmul(features, particles.t()) * labels.view(-1, 1) # B*M matrix
        probs = torch.sigmoid(logits) # B*M matrix
        temp = - (probs * labels.view(-1, 1)).t() # M*B matrix
        grads = torch.matmul(temp, features) * (self.train_num / batch_size) + self.weight_decay * particles
        return grads

    def nl_grads_calc(self, particles, features, labels): 
        return self.nl_grads_calc_batch(particles, self.train_features, self.train_labels)

    @torch.no_grad()
    def evaluation(self, particles, mass):
        train_outputs = self.likelihood_calc(particles, self.train_features, self.train_labels)
        test_outputs = self.likelihood_calc(particles, self.test_features, self.test_labels)
        train_outputs_avg = torch.matmul(mass, train_outputs)
        test_outputs_avg = torch.matmul(mass, test_outputs)
        train_nll = ( - torch.log(train_outputs_avg)).mean() # instance average
        train_error = 1 - ((torch.round(train_outputs_avg)).sum() / self.train_num) # instance average
        test_nll = ( - torch.log(test_outputs_avg)).mean() # instance average
        test_error = 1 - ((torch.round(test_outputs_avg)).sum() / self.test_num) # instance average
        mass_max, mass_min, mass_std = mass.max() * self.opts.particle_num, mass.min() * self.opts.particle_num, mass.std() * self.opts.particle_num
        # evaluate KSD
        sq_distance = torch.cdist(particles, particles, p = 2) ** 2
        ksds = tasks.ksd_matrix(particles, score =  - self.nl_grads_calc(
            particles, features = self.train_features, labels = self.train_labels), 
            bandwidth_h = self.state['median_dist'], sq_distance = sq_distance)
        ksd_value = torch.matmul(torch.matmul(mass, ksds), mass)
        # evaluate W2
        hmc_reference = self.state['particles']
        cost_matrix  = (torch.cdist(particles, hmc_reference) ** 2).cpu().numpy()
        mass_numpy = mass.cpu().numpy().astype(np.float64) # from tensor to numpy, need extra normalization
        transport_plan = ot.emd(mass_numpy / mass_numpy.sum(), ot.unif(hmc_reference.shape[0]), cost_matrix)
        w2_value = np.sqrt((cost_matrix * transport_plan).sum())
        # eval mse
        mse_value = (torch.matmul(mass, particles) - self.posterior_mean).pow(2).sum()
        return {'train_nll':train_nll.item(), 'train_error':train_error.item(),\
            'test_nll':test_nll.item(), 'test_error':test_error.item(), \
            'mass_min':mass_min.item(), 'ksd': ksd_value.item(), 'w2': w2_value, 'mse': mse_value.item()}

    #------------------------------------ save the results ----------------------------------------------------------------------------

    def save_final_results(self, writer, save_folder, result_dict):
        utils.save_final_results(save_folder, result_dict)
        np.save(os.path.join(save_folder, 'ksd.npy'), np.array(self.ksd))
        np.save(os.path.join(save_folder, 'w2.npy'), np.array(self.w2))

    def save_eval_to_tensorboard(self, writer, results, global_step):
        writer.add_scalar('train nll', results['train_nll'], global_step = global_step)
        writer.add_scalar('train error', results['train_error'], global_step = global_step)
        writer.add_scalar('test nll', results['test_nll'], global_step = global_step)
        writer.add_scalar('test error', results['test_error'], global_step = global_step)
        writer.add_scalar('mass min', results['mass_min'], global_step = global_step)
        writer.add_scalar('ksd', results['ksd'], global_step = global_step)
        writer.add_scalar('w2', results['w2'], global_step = global_step)
        writer.add_scalar('mse', results['mse'], global_step = global_step)
        self.ksd.append(results['ksd'])
        self.w2.append(results['w2'])

    #------------------------------------------ HMC as reference ------------------------------------------------------------
    def hmc(self):
        particles = torch.randn((self.hmc_particle_num, self.model_dim), device = self.device) / np.sqrt(self.model_dim)
        pars = []
        accu_accept_ratio = 0.0
        # ------------------------------------------------ GD find a good initialization --------------------------------------
        for i in tqdm(range(self.hmc_gd_init_iter)):
           grads = self.nl_grads_calc(particles, features = None, labels = None) 
           particles = particles - self.opts.lr_gd_init * grads
        for i in tqdm(range(self.hmc_burn_in + self.hmc_outer_iter)):
            q = particles.clone()
            velocity = torch.randn_like(particles, device = self.device)
            p = velocity.clone()
            grads = self.nl_grads_calc(q, features = None, labels = None)
            p = p - 0.5 * self.opts.lr_hmc * grads
            for k in range(self.hmc_inner_iter):
                q = q + self.opts.lr_hmc * p
                grads = self.nl_grads_calc(q, features = None, labels = None)
                if k != (self.hmc_inner_iter - 1): p = p - self.opts.lr_hmc * grads
            p = p - 0.5 * self.opts.lr_hmc * grads
            p = -p
            curr_u = self.potential_calc(particles, None, None)
            curr_k = velocity.pow(2).sum(1) / 2
            prop_u = self.potential_calc(q, None, None)
            prop_k = p.pow(2).sum(1) / 2
            accept_prob = torch.minimum(torch.exp(curr_u + curr_k - prop_u - prop_k), torch.ones(self.hmc_particle_num, device = self.device))
            accu_accept_ratio += accept_prob.mean()
            rand = torch.rand(self.hmc_particle_num, device = self.device)
            particles[rand < accept_prob] = q[rand < accept_prob].clone() # accept
            if i >= self.hmc_burn_in: pars.append(particles.clone())
            if (i + 1) % self.hmc_eval_interval == 0: 
                self.writer.add_scalar('min_acc_prob', accept_prob.min(), global_step = i)
                ll = self.likelihood_calc(particles, self.test_features, self.test_labels)
                self.writer.add_scalar('nll',  -torch.log(ll.mean(0)).mean().item(), global_step = i)
        pars = torch.cat(pars, dim = 0)
        sq_dist = torch.cdist(pars, pars, p = 2)**2
        state = {'particles': pars, 'median_dist': sq_dist.median()}
        utils.create_dirs_if_not_exist('./hmc_reference')
        torch.save(state, './hmc_reference/{}.pth'.format(self.opts.dataset))

    # def mala(self):
    #     particles = torch.randn((self.hmc_particle_num, self.model_dim), device = self.device) / np.sqrt(self.model_dim)
    #     particles[:,0] = 0
    #     pars = []
    #     accu_accept_ratio = 0.0
    #     # ------------------------------------------------ GD find a good initialization --------------------------------------
    #     for i in tqdm(range(self.hmc_gd_init_iter)):
    #        grads = self.nl_grads_calc(particles, features = None, labels = None) 
    #        particles = particles - self.opts.lr_gd_init * grads
    #     for i in tqdm(range(self.hmc_burn_in + self.hmc_outer_iter)):
    #         for k in range(self.hmc_inner_iter):
    #             grads_curr = self.nl_grads_calc(particles, features = None, labels = None)
    #             par_prop = particles - self.opts.lr_hmc * grads_curr + \
    #                 torch.randn_like(particles, device = self.device) * np.sqrt(2 * self.opts.lr_hmc)
    #             grads_prop = self.nl_grads_calc(par_prop, features = None, labels = None)
    #             curr_logp = - self.potential_calc(particles, None, None)
    #             prop_logp = - self.potential_calc(par_prop, None, None)
    #             part1 = torch.exp(prop_logp - curr_logp)
    #             part2 = torch.exp(

    #             )


        


    
