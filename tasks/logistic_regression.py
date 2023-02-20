import torch, numpy as np, os, torch.nn.functional as F
from algorithms._funcs import safe_log
# we use full batch in this experiment

class logistic_regression(object):
    def __init__(self, opts, **kw) -> None:
        self.weight_decay = 0.0005 # prior inverse_variance
        self.device = opts.device
        self.particle_num = opts.particle_num
        self.test_nll_list = []
        self.test_err_list = []
        self.train_nll_list = []
        self.train_err_list = []
        self.train_features = kw['info_source'].train_features
        self.train_labels = kw['info_source'].train_labels
        self.test_features = kw['info_source'].test_features
        self.test_labels = kw['info_source'].test_labels
        self.train_num = kw['info_source'].train_num
        self.test_num = kw['info_source'].test_num
        self.model_dim = kw['info_source'].model_dim

    @property
    @torch.no_grad()
    def init_particles(self):
        params = torch.randn(self.particle_num, self.model_dim, device = self.device) / np.sqrt(self.model_dim)
        return params
    
    #------------------------------------------ likelihood and grad ------------------------------------------------------------------

    @torch.no_grad()
    def likelihood(self, particles, features, labels): # len(particles.shape) must be 2
        logits = torch.matmul(features, particles.t()) * labels.view(-1, 1) # B*M matrix
        results = torch.sigmoid(logits).t() # M*B matrix 
        return results

    # @torch.no_grad()
    # def potential(self, particles, features = None, labels = None):
    #     potential = - torch.log(self.likelihood_calc(particles, self.train_features, self.train_labels) + 1e-6).sum(1)
    #     potential += 0.5 * np.log(2 * np.pi / self.weight_decay) + particles.pow(2).sum(1) * self.weight_decay / 2
    #     return potential
    @torch.no_grad()
    def _potential(self, particles, features, labels):
        logits = torch.matmul(features, particles.t()) * labels.view(-1, 1) # B * M matrix
        p1 = - F.logsigmoid(logits.t()).sum(1)
        p2 = 0.5 * np.log(2 * np.pi / self.weight_decay) + particles.pow(2).sum(1) * self.weight_decay / 2
        return p1 + p2

    @torch.no_grad()
    def potential(self, particles, features = None, labels = None):
        return self._potential(particles, self.train_features, self.train_labels)

    @torch.no_grad()
    def _grad_logp(self, particles, features, labels):
        batch_size = len(labels)
        logits = - torch.matmul(features, particles.t()) * labels.view(-1, 1) # B*M matrix
        probs = torch.sigmoid(logits) # B*M matrix
        temp = - (probs * labels.view(-1, 1)).t() # M*B matrix
        grads = torch.matmul(temp, features) * (self.train_num / batch_size) + self.weight_decay * particles
        return -grads
    
    @torch.no_grad()
    def grad_logp(self, particles, features = None, labels = None):
        return self._grad_logp(particles, self.train_features, self.train_labels)
    
    # def nl_grads_calc_batch(self, particles, features, labels): # len(particles.shape) mush be 2
    #     batch_size = len(labels)
    #     logits = - torch.matmul(features, particles.t()) * labels.view(-1, 1) # B*M matrix
    #     probs = torch.sigmoid(logits) # B*M matrix
    #     temp = - (probs * labels.view(-1, 1)).t() # M*B matrix
    #     grads = torch.matmul(temp, features) * (self.train_num / batch_size) + self.weight_decay * particles
    #     return grads

    # def nl_grads_calc(self, particles, features, labels): 
    #     return self.nl_grads_calc_batch(particles, self.train_features, self.train_labels)

    @torch.no_grad()
    def evaluation(self, particles, mass, writer, logger, count: int, save_folder):
        train_outputs = self.likelihood(particles, self.train_features, self.train_labels)
        test_outputs = self.likelihood(particles, self.test_features, self.test_labels)
        train_outputs_avg = torch.matmul(mass, train_outputs)
        test_outputs_avg = torch.matmul(mass, test_outputs)
        train_nll = ( - safe_log(train_outputs_avg)).mean() # instance average
        train_error = 1 - ((torch.round(train_outputs_avg)).sum() / self.train_num) # instance average
        test_nll = ( - safe_log(test_outputs_avg)).mean() # instance average
        test_error = 1 - ((torch.round(test_outputs_avg)).sum() / self.test_num) # instance average
        self.train_nll_list.append(train_nll.item())
        self.train_err_list.append(train_error.item())
        self.test_nll_list.append(test_nll.item())
        self.test_err_list.append(test_error.item())
        writer.add_scalar('train_nll', train_nll.item(), global_step = count)
        writer.add_scalar('train_err', train_error.item() * 100, global_step = count)
        writer.add_scalar('test_nll', test_nll.item(), global_step = count)
        writer.add_scalar('test_err', test_error.item() * 100, global_step = count)
        logger.info('count: %d, test_err: %.3f, test_nll: %.2e, train_err: %.3f, train_nll: %.2e'\
            %(count, test_error.item(), test_nll.item(), train_error.item(), train_nll.item()))
        

    @torch.no_grad()
    def final_process(self, particles, mass, writer, logger, save_folder, isSave):
        np.save(os.path.join(save_folder, 'test_err.npy'), np.array(self.test_err_list))
        np.save(os.path.join(save_folder, 'test_nll.npy'), np.array(self.test_nll_list))
        np.save(os.path.join(save_folder, 'train_err.npy'), np.array(self.train_err_list))
        np.save(os.path.join(save_folder, 'train_nll.npy'), np.array(self.train_nll_list))
        logger.info('mass_max: %.2f, mass_min: %.2f'%(mass.max().item() * self.particle_num, mass.min().item() * self.particle_num))

    # def save_eval_to_tensorboard(self, writer, results, global_step):
    #     writer.add_scalar('train nll', results['train_nll'], global_step = global_step)
    #     writer.add_scalar('train error', results['train_error'], global_step = global_step)
    #     writer.add_scalar('test nll', results['test_nll'], global_step = global_step)
    #     writer.add_scalar('test error', results['test_error'], global_step = global_step)
    #     writer.add_scalar('mass min', results['mass_min'], global_step = global_step)
    #     writer.add_scalar('ksd', results['ksd'], global_step = global_step)
    #     writer.add_scalar('w2', results['w2'], global_step = global_step)
    #     writer.add_scalar('mse', results['mse'], global_step = global_step)
    #     self.ksd.append(results['ksd'])
    #     self.w2.append(results['w2'])

    # #------------------------------------------ HMC as reference ------------------------------------------------------------
    # def hmc(self):
    #     particles = torch.randn((self.hmc_particle_num, self.model_dim), device = self.device) / np.sqrt(self.model_dim)
    #     pars = []
    #     accu_accept_ratio = 0.0
    #     # ------------------------------------------------ GD find a good initialization --------------------------------------
    #     for i in tqdm(range(self.hmc_gd_init_iter)):
    #        grads = self.nl_grads_calc(particles, features = None, labels = None) 
    #        particles = particles - self.opts.lr_gd_init * grads
    #     for i in tqdm(range(self.hmc_burn_in + self.hmc_outer_iter)):
    #         q = particles.clone()
    #         velocity = torch.randn_like(particles, device = self.device)
    #         p = velocity.clone()
    #         grads = self.nl_grads_calc(q, features = None, labels = None)
    #         p = p - 0.5 * self.opts.lr_hmc * grads
    #         for k in range(self.hmc_inner_iter):
    #             q = q + self.opts.lr_hmc * p
    #             grads = self.nl_grads_calc(q, features = None, labels = None)
    #             if k != (self.hmc_inner_iter - 1): p = p - self.opts.lr_hmc * grads
    #         p = p - 0.5 * self.opts.lr_hmc * grads
    #         p = -p
    #         curr_u = self.potential_calc(particles, None, None)
    #         curr_k = velocity.pow(2).sum(1) / 2
    #         prop_u = self.potential_calc(q, None, None)
    #         prop_k = p.pow(2).sum(1) / 2
    #         accept_prob = torch.minimum(torch.exp(curr_u + curr_k - prop_u - prop_k), torch.ones(self.hmc_particle_num, device = self.device))
    #         accu_accept_ratio += accept_prob.mean()
    #         rand = torch.rand(self.hmc_particle_num, device = self.device)
    #         particles[rand < accept_prob] = q[rand < accept_prob].clone() # accept
    #         if i >= self.hmc_burn_in: pars.append(particles.clone())
    #         if (i + 1) % self.hmc_eval_interval == 0: 
    #             self.writer.add_scalar('min_acc_prob', accept_prob.min(), global_step = i)
    #             ll = self.likelihood_calc(particles, self.test_features, self.test_labels)
    #             self.writer.add_scalar('nll',  -torch.log(ll.mean(0)).mean().item(), global_step = i)
    #     pars = torch.cat(pars, dim = 0)
    #     sq_dist = torch.cdist(pars, pars, p = 2)**2
    #     state = {'particles': pars, 'median_dist': sq_dist.median()}
    #     utils.create_dirs_if_not_exist('./hmc_reference')
    #     torch.save(state, './hmc_reference/{}.pth'.format(self.opts.dataset))

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


        


    
