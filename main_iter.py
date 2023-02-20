import pretty_errors
from functools import partial
import argparse
import importlib
import torch
import os
import numpy as np
import utils
from dataloader import myDataLoader

def main(opts):
    utils.set_random_seed(opts.seed)
    writer, logger, save_folder = utils.get_logger(opts, name = opts.algorithm)
    data_and_loader = myDataLoader(opts)
    task = importlib.import_module('tasks.%s'%opts.task).__getattribute__(opts.task)(opts, info_source = data_and_loader)
    algorithm = importlib.import_module('algorithms.%s'%opts.algorithm).__getattribute__(opts.algorithm)(
        opts, task.init_particles, torch.ones(opts.particle_num, device = opts.device) / opts.particle_num
    )
    loader_iter = iter(data_and_loader.get_train_loader())
    for step in range(opts.max_iter):
        # get features and labels
        try: features, labels = next(loader_iter)
        except:
            loader_iter = iter(data_and_loader.get_train_loader())
            features, labels = next(loader_iter)
        # set step_size, alpha, annealing
        annealing = (1 - opts.anneal) * np.tanh((1.3 * step / opts.max_iter * 2)**5) + opts.anneal if step < opts.max_iter / 2 and opts.anneal != 1.0 else 1.0
        if hasattr(opts, 'alpha') and annealing == 1.0:
            alpha = np.tanh((2.0 * step / opts.max_iter)**5) * opts.alpha if opts.al_warmup else opts.alpha
        else:
            alpha = 0.0
        # one step update
        algorithm.one_step_update(
            step_size = opts.lr, # common param
            alpha = alpha, # valid in CA/DK type methods
            annealing = annealing, # common param
            grad_fn = partial(task.grad_logp, features = features, labels = labels), # common param 
            potential_fn = partial(task.potential, features = features, labels = labels), # valid in CA/DK type methods
            leap_frog_num = opts.leap_iter if hasattr(opts, 'leap_iter') else None # valid in HMC
        )
        particles, mass = algorithm.get_state()
        utils.check(particles, mass, step, logger)
        # evaluation
        if step % opts.eval_interval == 0 or (step + 1) == opts.max_iter:
            task.evaluation(particles, mass, writer = writer, logger = logger, count = step, save_folder = save_folder)
            if opts.algorithm == 'HMC': 
                writer.add_scalar('avg_accept', algorithm.get_accept_ratio() * 100 / (step + 1), global_step = step)
            # writer.add_scalar('anneal', annealing, global_step = step)
    task.final_process(
        particles, mass, writer = writer, logger = logger, save_folder = save_folder, 
        isSave = opts.save_particles if hasattr(opts, 'save_particles') else None)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # basic setting
    parser.add_argument('--algorithm', type = str, default = 'SGLD') 
    parser.add_argument('--task', type = str, default = 'bnn_regression')
    parser.add_argument('--dataset', type = str, default='kin8nm')
    parser.add_argument('--max_iter', type = int, default = 6000)
    parser.add_argument('--batch_size', type = int, default = 1, help = 'invalid in gaussian and lr')
    parser.add_argument('--eval_interval', type = int, default = 200)
    parser.add_argument('--save_folder', type = str, default='results')
    parser.add_argument('--device', type = str, default = 'cuda:3')
    parser.add_argument('--seed', type = int, default = 9, help = 'random seed for algorithm')
    parser.add_argument('--split_size', type = float, default = 0.1, help = 'split ratio for dataset')
    parser.add_argument('--split_seed', type = int, default = 19, help = 'random seed to split dataset')
    opts,_ = parser.parse_known_args()
    # algorithm setting
    if opts.algorithm in ['HMC', 'SGLD', 'SVGD', 'GFSD', 'GFSDCA', 'GFSDDK', 'BLOB', 'BLOBCA', 'BLOBDK', 'KSDD', 'KSDDCA', 'KSDDDK', 'SGLDDK']: 
        parser.add_argument('--particle_num', type = int, default = 128)
        parser.add_argument('--lr', type = float, default = 0.01)
        parser.add_argument('--anneal', type = float, default = 1.0)
        if opts.algorithm in ['HMC']:
            parser.add_argument('--leap_iter', type = int, default = 10)
        if not opts.algorithm in ['SGLD', 'HMC']:
            parser.add_argument('--bwType', type = str, default = 'fix', choices = ['med', 'heu', 'nei', 'fix'])
            parser.add_argument('--knType', type = str, default = 'rbf', choices = ['rbf', 'imq'], help = 'KSDD type methods only support rbf')
            parser.add_argument('--bwVal', type = float, default = 0.1)
    opts,_ = parser.parse_known_args()
    if opts.algorithm in ['SVGDCA', 'GFSDCA', 'GFSDDK', 'BLOBCA', 'BLOBDK', 'KSDDCA', 'KSDDDK', 'SGLDDK']:
        parser.add_argument('--alpha', type = float, default = 1.0)
        parser.add_argument('--al_warmup', action = 'store_true')
    # task setting
    if opts.task in ['single_gaussian', 'multi_gaussian', 'funnel', 'demo', 'gaussian_process']:
        parser.add_argument('--model_dim', type = int, default = 10)
        parser.add_argument('--save_particles', action = 'store_true')
    if opts.task in ['gaussian_process']:
        parser.add_argument('--reference_path', type = str, default = 'hmc_reference/gaussian_process/particles.pkl')
    opts = parser.parse_args()
    assert opts.anneal <= 1 and opts.anneal >= 0, 'annealing should be in [0, 1]'
    main(opts)