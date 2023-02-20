from tensorboard.backend.event_processing import event_accumulator
import argparse
from tqdm import tqdm
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pretty_errors
from matplotlib.pyplot import MultipleLocator
import torch
import scipy.io as scio
def create_dirs_if_not_exist(dir_list):
    if isinstance(dir_list, list):
        for dir in dir_list:
            if not os.path.exists(dir):
                os.makedirs(dir)
    else:
        if not os.path.exists(dir_list):
            os.makedirs(dir_list)

def log_posterior(particles, sq_c00_dist, train_labels):
    train_num = len(train_labels)
    sigma = 0.2
    param_1 = particles[:,1].view(-1,1).unsqueeze(1)
    param_0 = particles[:,0].view(-1,1).unsqueeze(1)
    K_f = torch.exp(param_0) * torch.exp( - torch.exp(param_1) * sq_c00_dist.unsqueeze(0))
    K_y = K_f + (sigma**2) * torch.eye(train_num) # M * N * N tensor
    log_ll = -0.5 * (train_labels.t()@(torch.inverse(K_y)@train_labels)).squeeze() - 0.5 * torch.logdet(K_y)
    # prior
    log_pr = - torch.log(particles.pow(2) + 1).sum(1)
    log_po = log_ll + log_pr
    return  log_po

if __name__ == "__main__":    
    raw_data = scio.loadmat('./datasets/lidar/lidar.mat')
    train_features = torch.from_numpy(raw_data['range'].astype(np.float32)) # N * 1 array
    train_labels = torch.from_numpy(raw_data['logratio'].astype(np.float32)) # N * 1 array
    sq_c00_dist = torch.cdist(train_features, train_features, p = 2).pow(2)
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', type = str, default = 'pdf', choices = ['png', 'eps', 'pdf'])
    parser.add_argument('--dpi', type = int, default = 300)
    parser.add_argument('--plot_size', type = float, default = 70)
    parser.add_argument('--min_ratio', type = float, default = 0.1)
    parser.add_argument('--max_ratio', type = float, default = 10)
    opts = parser.parse_args()
    f =  open('./load_results/plot_settings.yaml', 'r')
    plot_settings_common = yaml.load(f.read(), Loader=yaml.FullLoader)
    # --------------------------- load resutls for meg dataset ---------------------------------------------
    log_folder = './results_save/gaussian_process'
    log_folder_list = os.listdir(log_folder)
    data = {}
    for log in tqdm(log_folder_list):
        if 'S[0]' not in log: continue
        algo_name = log[0: log.find('_')]
        result_folder = os.path.join(log_folder, log)
        particles = np.load(os.path.join(result_folder, 'particles.npy'))
        mass = np.load(os.path.join(result_folder, 'mass.npy'))
        if not algo_name in data:
            data[algo_name] = [particles, mass]
    # plot particles
    save_folder = './figures/gaussian_process'
    create_dirs_if_not_exist(save_folder)
    for algorithm in tqdm(data.keys()):
        particle, mass = data[algorithm]
        particle, mass = particle[-1], mass[-1]
        particle_num = len(mass)
        plt.figure(figsize=(6.4, 4.8))
        x, y = torch.linspace(-5, 5, 51), torch.linspace(-13, -7, 31)
        grid_X, grid_Y = torch.meshgrid(x,y)
        loc = torch.cat([grid_X.unsqueeze(2), grid_Y.unsqueeze(2)], dim = 2)
        pdf = log_posterior(loc.view(-1, 2), sq_c00_dist, train_labels)
        pdf = pdf.view(51, 31)
        fig = plt.figure(num = 1)
        plt.contour(grid_X.cpu().numpy(), grid_Y.cpu().numpy(), pdf.cpu().numpy(), levels = 30)
        # set particle size
        mass = np.clip(mass * particle_num, opts.min_ratio, opts.max_ratio)
        size_list = (mass * opts.plot_size).astype(np.int)
        plt.scatter(particle[:, 0], particle[:, 1], alpha = 0.6, s = size_list, c = 'r', label = plot_settings_common['label'][algorithm])
        plt.legend(fontsize = 25, loc = 3)
        plt.ylim([ - 13, -7])
        plt.xlim([ - 5, 3])
        plt.xticks([])
        plt.yticks([])
        plt.tick_params(labelsize = 22)
        plt.tight_layout()
        if opts.suffix == 'eps':
            ax = plt.gca()
            ax.set_rasterized(True)
        plt.savefig(os.path.join(save_folder, '{}.{}'.format(algorithm, opts.suffix)), dpi = opts.dpi, bbox_inches = 'tight')
        plt.close()