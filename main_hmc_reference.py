import pretty_errors
from functools import partial
import matplotlib.pyplot as plt
import argparse
import importlib
import torch
import os
import numpy as np
import utils
from dataloader import myDataLoader

def main(opts):
    utils.set_random_seed(opts.seed)
    writer, logger, save_folder = utils.get_logger(opts, name = opts.algorithm, save_folder = opts.save_folder)
    data_and_loader = myDataLoader(opts)
    task = importlib.import_module('tasks.%s'%opts.task).__getattribute__(opts.task)(opts, info_source = data_and_loader)
    algorithm = importlib.import_module('algorithms.%s'%opts.algorithm).__getattribute__(opts.algorithm)(
        opts, task.init_particles, torch.ones(opts.particle_num, device = opts.device) / opts.particle_num
    )
    loader_iter = iter(data_and_loader.get_train_loader())
    particle_list = []
    for step in range(opts.burn_in + opts.loop_iter):
        # get features and labels
        try: features, labels = next(loader_iter)
        except:
            loader_iter = iter(data_and_loader.get_train_loader())
            features, labels = next(loader_iter)
        # set step_size, alpha, annealing
        # annealing = (1 - opts.anneal) * np.tanh((1.3 * step / opts.max_iter * 2)**5) + opts.anneal if step < opts.max_iter / 2 and opts.anneal != 1.0 else 1.0
        # alpha = np.tanh((1.7 * step / opts.max_iter)**5) * opts.alpha if hasattr(opts, 'alpha') and annealing == 1.0 else 0
        # one step update
        algorithm.one_step_update(
            step_size = opts.lr, # common param
            leap_frog_num = opts.leap_iter, 
            grad_fn = partial(task.grad_logp, features = features, labels = labels), 
            potential_fn = partial(task.potential, features = features, labels = labels)
        )
        particles, mass = algorithm.get_state()
        logger.info('count: {}, average accept: {:.2f}'.format(step, algorithm.get_accept_ratio() * 100 / (step + 1)))
        writer.add_scalar('average accept', algorithm.get_accept_ratio() * 100 / (step + 1), global_step = step)
        utils.check(particles, mass, step, logger)
        if step >= opts.burn_in:
            particle_list.append(particles.cpu())
    particle_list = torch.cat(particle_list, dim = 0)
    torch.save(particle_list, os.path.join(save_folder, 'particles.pkl'))
    # plot the figure
    fig = task.plot_result(particle_list.cpu().numpy(), torch.ones(particle_list.shape[0]).cpu().numpy() / particle_list.shape[0], device = opts.device)        
    plt.savefig(os.path.join(save_folder, 'figure.png'), pad_inches = 0.0)
    writer.add_figure(tag = 'samples', figure = fig)
    plt.close()
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # basic setting
    parser.add_argument('--algorithm', type = str, default = 'HMC') 
    parser.add_argument('--task', type = str, default = 'gaussian_process')
    parser.add_argument('--dataset', type = str, default='lidar')
    parser.add_argument('--burn_in', type = int, default = 100)
    parser.add_argument('--loop_iter', type = int, default = 200)
    parser.add_argument('--leap_iter', type = int, default = 100)
    parser.add_argument('--batch_size', type = int, default = 1, help = 'invalid in gaussian and lr')
    parser.add_argument('--particle_num', type = int, default = 50)
    parser.add_argument('--lr', type = float, default = 0.1)
    parser.add_argument('--save_folder', type = str, default = 'hmc_reference/gaussian_process')
    parser.add_argument('--device', type = str, default = 'cuda:3')
    parser.add_argument('--seed', type = int, default = 9, help = 'random seed for algorithm')
    parser.add_argument('--split_size', type = float, default = 0.1, help = 'split ratio for dataset')
    parser.add_argument('--split_seed', type = int, default = 19, help = 'random seed to split dataset')
    opts = parser.parse_args()
    main(opts)