from dataloader import myDataLoader
import torch
import common.utils as utils
import numpy as np
import os
import torch.nn.functional as F

class functions(myDataLoader):
    def __init__(self, opts) -> None:
        super().__init__(opts)
        self.test_error = []
        self.test_nll = []

    def pack_params(self, w1, b1, w2, b2):
        params = torch.cat([w1.flatten(), b1, w2.flatten(), b2])
        return params

    def unpack_params(self, params):
        w1 = params[:self.data_dim * self.n_hidden].view(self.data_dim, self.n_hidden)
        b1 = params[self.data_dim * self.n_hidden : (self.data_dim + 1) * self.n_hidden]
        temp = params[(self.data_dim + 1) * self.n_hidden:]
        w2, b2 = temp[:self.n_hidden * self.num_classes].view(self.n_hidden, self.num_classes), temp[-self.num_classes:]
        return w1, b1, w2, b2

    def init_params(self):
        w1 = 1.0 / np.sqrt(self.data_dim + 1) * torch.randn((self.data_dim, self.n_hidden), device = self.device)
        b1 = torch.zeros((self.n_hidden, ), device = self.device)
        w2 = 1.0 / np.sqrt(self.n_hidden + 1) * torch.randn((self.n_hidden, self.num_classes), device = self.device)
        b2 = torch.zeros((self.num_classes, ), device = self.device)
        return w1, b1, w2, b2

    @torch.no_grad()
    def init_net(self, shape):
        param_group = torch.zeros(shape, device = self.device)
        for i in range(len(param_group)):
            w1, b1, w2, b2 = self.init_params()
            param_group[i, :] = self.pack_params(w1, b1, w2, b2)
        return param_group

    # ------------------------------------------ likelihood and grad ------------------------------------------------------------------
    def prediction(self, x, features):
        num_particles = len(x)
        w1_s = x[:, :self.data_dim * self.n_hidden].view(num_particles, self.data_dim, self.n_hidden)
        b1_s = x[:, self.data_dim * self.n_hidden : (self.data_dim + 1) * self.n_hidden].unsqueeze(1)
        temp = x[:, (self.data_dim + 1) * self.n_hidden:]
        w2_s = temp[:, :self.n_hidden * self.num_classes].view(num_particles, self.n_hidden, self.num_classes)
        b2_s = temp[:, - self.num_classes : ].unsqueeze(1)
        outputs = torch.matmul(torch.sigmoid(torch.matmul(features, w1_s) + b1_s), w2_s) + b2_s
        return outputs, [w1_s, b1_s, w2_s, b2_s] # return the params for the following calc

    def log_posteriors_calc(self, x, features, labels):
        batch_size = len(features)
        outs, _ = self.prediction(x, features)
        out_log_softmax = F.log_softmax(outs, dim = 2)
        log_posteriors = torch.sum(out_log_softmax * labels.expand(self.opts.particle_num, batch_size, self.num_classes), dim = 2)
        log_posteriors = log_posteriors.mean(1) * self.train_num
        return log_posteriors

    def nl_grads_calc(self, x, features, labels): # use the auto grad of pytorch
        particles = x.detach()
        particles = particles.requires_grad_(True)
        loss = - self.log_posteriors_calc(particles, features, labels)
        grad_list = torch.autograd.grad(
            outputs = loss.sum(),
            inputs = particles
        )[0]
        return grad_list

    @torch.no_grad()
    def likelihoods_predicts_calc(self, particles, features, labels):
        batch_size = len(features)
        outs, _ = self.prediction(particles, features)
        out_softmax = F.softmax(outs, dim = 2)
        likelihoods = torch.sum(out_softmax * labels.expand(self.opts.particle_num, batch_size, self.num_classes), dim = 2)
        return likelihoods, out_softmax

    @torch.no_grad()
    def potential_calc(self, particles, features, labels):
        potential = - self.log_posteriors_calc(particles, features, labels)
        return potential

    
    @torch.no_grad()
    def evaluation(self, particles, mass):
        train_likeli, train_out_sf = self.likelihoods_predicts_calc(particles, self.train_features, self.train_labels)
        test_likeli, test_out_sf = self.likelihoods_predicts_calc(particles, self.test_features, self.test_labels)
        train_likeli_avg, test_likeli_avg = torch.matmul(mass, train_likeli), torch.matmul(mass, test_likeli)
        train_nll = ( - torch.log(train_likeli_avg)).mean() # instance average
        test_nll = ( - torch.log(test_likeli_avg)).mean() # instance average
        temp = mass.unsqueeze(1).unsqueeze(1)
        train_out_sf_avg, test_out_sf_avg = (temp * train_out_sf).sum(0), (temp * test_out_sf).sum(0)
        train_preds, test_preds = torch.max(train_out_sf_avg, dim = 1)[1], torch.max(test_out_sf_avg, dim = 1)[1]
        train_error = 1 - self.train_labels[torch.arange(0, self.train_num, dtype = int, device = self.device), train_preds].sum() / self.train_num
        test_error = 1 - self.test_labels[torch.arange(0, self.test_num, dtype = int, device = self.device), test_preds].sum() / self.test_num
        mass_max, mass_min, mass_std = mass.max() * self.opts.particle_num, mass.min() * self.opts.particle_num, mass.std() * self.opts.particle_num
        return {'train_nll':train_nll.item(), 'train_error':train_error.item(),\
            'test_nll':test_nll.item(), 'test_error':test_error.item(),\
            'mass_max':mass_max.item(), 'mass_min':mass_min.item(), 'mass_std':mass_std.item()}
    
    #------------------------------------ save the results -----------------------------------------------------

    def save_final_results(self, writer, save_folder, result_dict):
        utils.save_final_results(save_folder, result_dict)
        np.save(os.path.join(save_folder, 'test_error.npy'), np.array(self.test_error))
        np.save(os.path.join(save_folder, 'test_nll.npy'), np.array(self.test_nll))

    def save_eval_to_tensorboard(self, writer, results, global_step):
        writer.add_scalar('train nll', results['train_nll'], global_step = global_step)
        writer.add_scalar('train_error', results['train_error'], global_step = global_step)
        writer.add_scalar('test nll', results['test_nll'], global_step = global_step)
        writer.add_scalar('test_error', results['test_error'], global_step = global_step)
        writer.add_scalar('mass min', results['mass_min'], global_step = global_step)
        writer.add_scalar('mass max', results['mass_max'], global_step = global_step)
        writer.add_scalar('mass std', results['mass_std'], global_step = global_step)
        self.test_error.append(results['test_error'])
        self.test_nll.append(results['test_nll'])