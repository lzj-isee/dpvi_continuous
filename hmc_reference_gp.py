import torch
from dataloader import myDataLoader
import importlib
from tqdm import tqdm
import numpy as np
import argparse
import os
import random
import common.utils as utils
import matplotlib.pyplot as plt

def save_figure_tensorboard(functions, writer, particles, curr_iter_count):
    # calc mesh
    numx, numy = 51, 31
    x, y = torch.linspace(-5, 5, numx, device = functions.device), torch.linspace(-13, -7, numy, device = functions.device)
    grid_X, grid_Y = torch.meshgrid(x,y)
    loc = torch.cat([grid_X.unsqueeze(2), grid_Y.unsqueeze(2)], dim = 2)
    pdf = functions.log_pdf_calc(loc.view(-1, 2))
    pdf = pdf.view(numx, numy)
    fig = plt.figure(num = 1)
    plt.contour(grid_X.cpu().numpy(), grid_Y.cpu().numpy(), pdf.cpu().numpy(), levels = 30)
    # set particle size
    plt.scatter(particles[:, 0].cpu().numpy(), particles[:, 1].cpu().numpy(), alpha = 0.2, s = 5, c = 'r')
    # plt.scatter(particles[:, 0], particles[:, 1], alpha = 0.5, s = 15) # debug for testing ref particles
    plt.ylim([ - 13, -7])
    plt.xlim([ - 5, 5])
    plt.tight_layout()
    writer.add_figure(tag = 'samples', figure = fig, global_step = curr_iter_count)
    plt.close()

def run(opts):
    functions = importlib.import_module('tasks.{:}'.format(opts.task)).__getattribute__('functions')(opts)
    particles = functions.init_net((opts.particle_num, functions.model_dim))
    pars = []
    accu_accept_ratio = 0.0
    # ------------------------------------------------ GD find a good initialization --------------------------------------
    for i in tqdm(range(opts.burn_in + opts.outer_iter)):
        q = particles.clone()
        velocity = torch.randn_like(particles, device = functions.device)
        p = velocity.clone()
        grads = functions.nl_grads_calc(q, features = None, labels = None)
        p = p - 0.5 * opts.lr * grads
        for k in range(opts.inner_iter):
            q = q + opts.lr * p
            grads = functions.nl_grads_calc(q, features = None, labels = None)
            if k != (opts.inner_iter - 1): p = p - opts.lr * grads
        p = p - 0.5 * opts.lr * grads
        p = -p
        curr_u = functions.potential_calc(particles, None, None)
        curr_k = velocity.pow(2).sum(1) / 2
        prop_u = functions.potential_calc(q, None, None)
        prop_k = p.pow(2).sum(1) / 2
        accept_prob = torch.minimum(torch.exp(curr_u + curr_k - prop_u - prop_k), torch.ones(opts.particle_num, device = functions.device))
        accu_accept_ratio += accept_prob.mean()
        rand = torch.rand(opts.particle_num, device = functions.device)
        particles[rand < accept_prob] = q[rand < accept_prob].clone() # accept
        if i >= opts.burn_in: 
            pars.append(particles.clone())
            save_figure_tensorboard(functions, functions.writer, torch.cat(pars, dim = 0), i)
            #save_figure_tensorboard(functions, functions.writer, particles, i)
        functions.writer.add_scalar('min_acc_prob', accu_accept_ratio / (i+1), global_step = i)
    pars = torch.cat(pars, dim = 0)
    sq_dist = torch.cdist(pars, pars, p = 2)**2
    state = {'particles': pars, 'median_dist': sq_dist.median()}
    utils.create_dirs_if_not_exist('./hmc_reference')
    torch.save(state, './hmc_reference/{}.pth'.format(opts.dataset))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic setting
    parser.add_argument('--algorithm', type = str, default = 'HMC') 
    parser.add_argument('--task', type = str, default = 'gaussian_process')
    parser.add_argument('--dataset', type = str, default='lidar')
    parser.add_argument('--burn_in', type = int, default = 100)
    parser.add_argument('--outer_iter', type = int, default = 200)
    parser.add_argument('--inner_iter', type = int, default = 100)
    parser.add_argument('--batch_size', type = int, default = 1, help = 'invalid in gaussian and lr')
    parser.add_argument('--particle_num', type = int, default = 50)
    parser.add_argument('--lr', type = float, default = 0.1)
    parser.add_argument('--save_folder', type = str, default='results')
    parser.add_argument('--gpu', type = int, default = 3, help = 'gpu_id')
    parser.add_argument('--eval_interval', type = int, default = 1)
    parser.add_argument('--seed', type = int, default = 0, help = 'random seed for algorithm')
    parser.add_argument('--split_size', type = float, default = 0.1, help = 'split ratio for dataset')
    parser.add_argument('--split_seed', type = int, default = 19, help = 'random seed to split dataset')
    opts = parser.parse_args()
    # set the random seed
    os.environ['PYTHONHASHSEED'] = str(opts.seed)
    np.random.seed(opts.seed)
    random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.random.manual_seed(opts.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opts.seed) 
        torch.cuda.manual_seed_all(opts.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # import algorithm
    run(opts)