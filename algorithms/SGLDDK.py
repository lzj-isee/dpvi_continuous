import torch
import numpy as np
from ._funcs import kernel_func, duplicate_kill_particles
from algorithms.GFSD import GFSD
from algorithms.GFSDCA import GFSDCA

class SGLDDK(GFSD):
    def __init__(self, opts, init_particles, init_mass) -> None:
        super().__init__(opts, init_particles, init_mass)

    def one_step_update(self, step_size = None, alpha = None, grad_fn = None, potential_fn = None, **kw):
        grads = grad_fn(self.particles) * kw['annealing'] if 'annealing' in kw.keys() else grad_fn(self.particles)
        potential = potential_fn(self.particles)
        kernel, _, _ = kernel_func(self.particles, self.knType, self.bwType, self.bwVal, bw_only = None)
        self.particles += step_size * grads + torch.randn_like(self.particles) * np.sqrt(2 * step_size)
        avg_first_variation = GFSDCA.get_avg_first_variation(self.mass, potential, kernel)
        prob_list = 1 - torch.exp( - avg_first_variation.abs() * alpha * step_size)
        self.particles = duplicate_kill_particles(prob_list, avg_first_variation > 0, self.particles, noise_amp = 0)

    def get_state(self):
        return self.particles, self.mass

























# import torch
# from dataloader import myDataLoader
# import importlib
# from tqdm import tqdm
# import numpy as np
# import time
# def run(opts): 
#     functions = importlib.import_module('tasks.{:}'.format(opts.task)).__getattribute__('functions')(opts)
#     print('algorithm and setting: \n', functions.save_name)
#     # init particles
#     particles = functions.init_net((opts.particle_num, functions.model_dim))
#     mass = torch.ones(opts.particle_num, device = functions.device) / opts.particle_num
#     curr_iter_count, results, coef_h = 0, None, 1 / np.log(opts.particle_num)
#     index_list = torch.linspace(0, opts.particle_num - 1, opts.particle_num, dtype = torch.int, device = functions.device)
#     # ---------------------------------------------- main part ---------------------------------------------------------------------
#     eval_count = 0
#     accu_time = 0
#     for epoch in tqdm(range(opts.epochs)):
#         if accu_time > opts.time: break
#         for iter, (train_features, train_labels) in enumerate(functions.train_loader):
#             start_time = time.time()
#             curr_iter_count += 1
#             grads = functions.nl_grads_calc(
#                 particles, features = train_features, labels = train_labels
#             )
#             # calc kernel
#             sq_distance = torch.cdist(particles, particles, p = 2) ** 2
#             # bandwidth and kernel
#             bandwidth_h = functions.bandwidth_calc(sq_distance, coef_h)
#             kernel = torch.exp( - sq_distance / bandwidth_h)
#             # update particles
#             particles = particles - opts.lr * grads + torch.randn_like(particles) * np.sqrt(2 * opts.lr)
#             # duplicate and kill particles
#             potential = functions.potential_calc(particles, train_features, train_labels)
#             beta = torch.log((mass * kernel).sum(1) + 1e-6) + potential
#             beta_bar = beta - (beta * mass).sum()
#             rand_number = torch.rand(opts.particle_num, device = functions.device)
#             prob_list = 1 - torch.exp( - beta_bar.abs() * opts.alpha * opts.lr)
#             rand_index = torch.randint(0, opts.particle_num - 1, (opts.particle_num,), device = functions.device)
#             for k in range(opts.particle_num):
#                 if beta_bar[k] > 0: # kill particle k
#                     if rand_number[k] < prob_list[k]:
#                         particles[k] = particles[index_list != k][rand_index[k]].clone()
#                 else: # duplicate particle k
#                     if rand_number[k] < prob_list[k]:
#                         particles[index_list != k][rand_index[k]] = particles[k].clone()
#             end_time = time.time()
#             accu_time = accu_time + end_time - start_time
#             # check the value of particles and mass
#             functions.check(particles, mass, curr_iter_count)
#             # evalutaion
#             if (int(accu_time) - eval_count) >= 1:
#                 eval_count += 1
#                 results = functions.evaluation(particles, mass)
#                 functions.save_eval_to_tensorboard(
#                     functions.writer, 
#                     results, 
#                     eval_count# epoch * len(functions.train_loader) + iter
#                 )
#     functions.save_final_results(functions.writer, functions.save_folder, results)
#     functions.writer.close()