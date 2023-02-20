import torch
import numpy as np
import os

class bnn_regression(object):
    def __init__(self, opts, **kw) -> None:
        super().__init__()
        self.param_a, self.param_b = 1, 0.1
        self.test_rmse_list = []
        self.test_nll_list= []
        self.test_nll_con_list = []
        self.data_dim = kw['info_source'].data_dim
        self.out_dim = kw['info_source'].out_dim
        self.model_dim = kw['info_source'].model_dim
        self.n_hidden = kw['info_source'].n_hidden
        self.device = opts.device
        self.train_num = kw['info_source'].train_num
        self.train_features = kw['info_source'].train_features
        self.train_labels = kw['info_source'].train_labels
        self.test_num = kw['info_source'].test_num
        self.test_features = kw['info_source'].test_features
        self.test_labels = kw['info_source'].test_labels
        self.std_labels = kw['info_source'].std_labels
        self.mean_labels = kw['info_source'].mean_labels
        self.particle_num = opts.particle_num

    def pack_params(self, w1, b1, w2, b2, log_gamma, log_lambda):
        params = torch.cat([w1.flatten(), b1, w2.flatten(), b2, log_gamma.view(-1), log_lambda.view(-1)])
        return params

    def unpack_params(self, params):
        w1 = params[:self.data_dim * self.n_hidden].view(self.data_dim, self.n_hidden)
        b1 = params[self.data_dim * self.n_hidden : (self.data_dim + 1) * self.n_hidden]
        temp = params[(self.data_dim + 1) * self.n_hidden:]
        w2, b2 = temp[:self.n_hidden * self.out_dim].view(self.n_hidden, self.out_dim), temp[-2-self.out_dim:-2]
        log_gamma, log_lambda = temp[-2], temp[-1]
        return w1, b1, w2, b2, log_gamma, log_lambda

    def init_params(self):
        w1 = 1.0 / np.sqrt(self.data_dim + 1) * torch.randn((self.data_dim, self.n_hidden), device = self.device)
        b1 = torch.zeros((self.n_hidden, ), device = self.device)
        w2 = 1.0 / np.sqrt(self.n_hidden + 1) * torch.randn((self.n_hidden, self.out_dim), device = self.device)
        b2 = torch.zeros((self.out_dim, ), device = self.device)
        log_gamma = torch.log(torch.ones((1, ), device = self.device) * np.random.gamma(self.param_a, self.param_b))
        log_lambda = torch.log(torch.ones((1, ), device = self.device) * np.random.gamma(self.param_a, self.param_b))
        return w1, b1, w2, b2, log_gamma, log_lambda

    @property
    @torch.no_grad()
    def init_particles(self):
        param_group = torch.zeros(self.particle_num, self.model_dim, device = self.device)
        for i in range(len(param_group)):
            w1, b1, w2, b2, log_gamma, log_lambda = self.init_params()
            # use better initiaization for gamma
            index = torch.multinomial(1 / torch.ones(self.train_num, device = self.device), \
            np.min([self.train_num, 1000]), replacement = False)
            y_predic = self.prediction_single(
                self.pack_params(w1, b1, w2, b2, log_gamma, log_lambda), 
                self.train_features[index]
            )
            log_gamma = - torch.log((y_predic - self.train_labels[index].view(-1, self.out_dim)).pow(2).mean())
            param_group[i, :] = self.pack_params(w1, b1, w2, b2, log_gamma, log_lambda)
        return param_group

    # ------------------------------------------ likelihood and grad ------------------------------------------------------------------
    def prediction_single(self, x, features):
        w1, b1, w2, b2, _, _ = self.unpack_params(x)
        outputs = torch.matmul(torch.relu(torch.matmul(features, w1) + b1), w2) + b2
        return outputs 

    def prediction(self, x, features):
        num_particles = len(x)
        w1_s = x[:, :self.data_dim * self.n_hidden].view(num_particles, self.data_dim, self.n_hidden)
        b1_s = x[:, self.data_dim * self.n_hidden : (self.data_dim + 1) * self.n_hidden].unsqueeze(1)
        temp = x[:, (self.data_dim + 1) * self.n_hidden:]
        w2_s = temp[:, :self.n_hidden * self.out_dim].view(num_particles, self.n_hidden, self.out_dim)
        b2_s = temp[:, -2 - self.out_dim : -2].unsqueeze(1)
        log_gamma, log_lambda = temp[:, -2], temp[:, -1]
        outputs = torch.matmul(torch.relu(torch.matmul(features, w1_s) + b1_s), w2_s) + b2_s
        return outputs, [w1_s, b1_s, w2_s, b2_s, log_gamma, log_lambda] # return the params for the following calc

    def log_posteriors_calc(self, x, features, labels):
        batch_size = len(features)
        predictions, unpack_params = self.prediction(x, features)
        w1_s, b1_s, w2_s, b2_s, log_gamma, log_lambda = unpack_params
        log_likeli_data = - 0.5 * batch_size * (np.log(2*np.pi) - log_gamma) - (log_gamma.exp()/2) * \
            (predictions - labels.view(-1, self.out_dim)).pow(2).sum((1,2))
        log_prior_data = (self.param_a - 1) * log_gamma - self.param_b * log_gamma.exp() + log_gamma
        log_prior_w = -0.5 * (self.model_dim - 2) * (np.log(2*np.pi) - log_gamma) - (log_gamma.exp()/2) * \
            (w1_s.pow(2).sum((1,2))+w2_s.pow(2).sum((1,2))+b1_s.pow(2).sum((1,2))+b2_s.pow(2).sum((1,2))) + \
            (self.param_a - 1) * log_lambda - self.param_b * log_lambda.exp() + log_lambda
        log_posteriors = log_likeli_data * self.train_num / batch_size + log_prior_data + log_prior_w
        return log_posteriors
    
    def potential(self, particles, features, labels):
        return -self.log_posteriors_calc(particles, features, labels)

    def nl_grads_calc_batch(self, x, features, labels): # use the auto grad of pytorch
        particles = x.detach()
        particles = particles.requires_grad_(True)
        loss = - self.log_posteriors_calc(particles, features, labels)
        grad_list = torch.autograd.grad(
            outputs = loss.sum(),
            inputs = particles
        )[0]
        return grad_list

    def grad_logp(self, particles, features, labels):
        return -self.nl_grads_calc_batch(particles, features, labels)

    # def nl_grads_calc(self, particles, features, labels): 
    #     return self.nl_grads_calc_batch(particles, features, labels)

    @torch.no_grad()
    def likelihoods_predicts_calc(self, particles, features, labels, is_Test = False):
        predicts, _ = self.prediction(particles, features)
        if is_Test: predicts = predicts * self.std_labels + self.mean_labels
        log_gamma_s = particles[:, -2].view(-1, 1)
        part_1 = log_gamma_s.exp().sqrt() / np.sqrt(2 * np.pi)
        part_2 = torch.exp(- (predicts - labels.view(-1, self.out_dim)).pow(2).sum(2) / 2 * log_gamma_s.exp())
        likelihoods = part_1 * part_2 + 1e-5
        return likelihoods, predicts
    
    @torch.no_grad()
    def evaluation(self, particles, mass, writer, logger, count: int, save_folder):
        train_likeli, train_predicts = self.likelihoods_predicts_calc(particles, self.train_features, self.train_labels, is_Test=False)
        test_likeli, test_predicts = self.likelihoods_predicts_calc(particles, self.test_features, self.test_labels, is_Test=True)
        temp = mass.unsqueeze(1).unsqueeze(1)
        train_predict_avg, test_predict_avg = (temp * train_predicts).sum(0), (temp * test_predicts).sum(0)
        train_likeli_avg, test_likeli_avg = torch.matmul(mass, train_likeli), torch.matmul(mass, test_likeli)
        train_nll, test_nll = - train_likeli_avg.log().mean(), - test_likeli_avg.log().mean()
        train_std = torch.std(self.train_labels, unbiased = True) / 5
        test_std = torch.std(self.test_labels, unbiased = True) / 5
        train_nll_convert =  torch.mean((train_predict_avg.squeeze() - self.train_labels).pow(2) / 2 / train_std**2)
        test_nll_convert = torch.mean( (test_predict_avg.squeeze() - self.test_labels).pow(2) / 2 / test_std**2)
        train_rmse = (train_predict_avg.squeeze() - self.train_labels).pow(2).mean().sqrt()
        test_rmse = (test_predict_avg.squeeze() - self.test_labels).pow(2).mean().sqrt()
        # mass_max, mass_min, mass_std = mass.max() * self.opts.particle_num, mass.min() * self.opts.particle_num, mass.std() * self.opts.particle_num
        # return {'train_nll':train_nll.item(), 'train_rmse':train_rmse.item(),\
        #     'test_nll':test_nll.item(), 'test_rmse':test_rmse.item(),\
        #     'mass_max':mass_max.item(), 'mass_min':mass_min.item(), 'mass_std':mass_std.item()}
        self.test_rmse_list.append(test_rmse.item())
        self.test_nll_list.append(test_nll.item())
        self.test_nll_con_list.append(test_nll_convert.item())
        writer.add_scalar('test_rmse', test_rmse.item(), global_step = count)
        writer.add_scalar('test_nll', test_nll.item(), global_step = count)
        writer.add_scalar('test_nll_con', test_nll_convert.item(), global_step = count)
        writer.add_scalar('train_rmse', train_rmse.item(), global_step = count)
        writer.add_scalar('train_nll', train_nll.item(), global_step = count)
        writer.add_scalar('train_nll_con', train_nll_convert.item(), global_step = count)
        logger.info('count: %d, test_rmse: %.3e test_nll: %.3e, test_nll_con: %.3e, train_rmse: %.3e, train_nll: %.3e, train_nll_con: %.3e'\
            %(count, test_rmse.item(), test_nll.item(), test_nll_convert.item(), train_rmse.item(), train_nll.item(), train_nll_convert.item()))

    @torch.no_grad()
    def final_process(self, particles, mass, writer, logger, save_folder, isSave):
        np.save(os.path.join(save_folder, 'test_rmse.npy'), np.array(self.test_rmse_list))
        np.save(os.path.join(save_folder, 'test_nll.npy'), np.array(self.test_nll_list))
        np.save(os.path.join(save_folder, 'test_nll_con.npy'), np.array(self.test_nll_con_list))
    
    # #------------------------------------ save the results -----------------------------------------------------

    # def save_final_results(self, writer, save_folder, result_dict):
    #     utils.save_final_results(save_folder, result_dict)
    #     np.save(os.path.join(save_folder, 'test_rmses.npy'), np.array(self.test_rmses))

    # def save_eval_to_tensorboard(self, writer, results, global_step):
    #     writer.add_scalar('train nll', results['train_nll'], global_step = global_step)
    #     writer.add_scalar('train_rmse', results['train_rmse'], global_step = global_step)
    #     writer.add_scalar('test nll', results['test_nll'], global_step = global_step)
    #     writer.add_scalar('test_rmse', results['test_rmse'], global_step = global_step)
    #     writer.add_scalar('mass min', results['mass_min'], global_step = global_step)
    #     writer.add_scalar('mass max', results['mass_max'], global_step = global_step)
    #     writer.add_scalar('mass std', results['mass_std'], global_step = global_step)
    #     self.test_rmses.append(results['test_rmse'])