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
def create_dirs_if_not_exist(dir_list):
    if isinstance(dir_list, list):
        for dir in dir_list:
            if not os.path.exists(dir):
                os.makedirs(dir)
    else:
        if not os.path.exists(dir_list):
            os.makedirs(dir_list)

def pdf_calc(particles):  # M * dim matrix
        mean = torch.zeros((1, 2))
        var = 6
        cov = 0.9 * 6
        cov_matrix = cov * torch.ones((2, 2))
        cov_matrix = cov_matrix - torch.diag(torch.diag(cov_matrix)) + torch.diag(var * torch.ones(2))
        in_cov_matrix = torch.inverse(cov_matrix)
        result = torch.exp( - (torch.matmul((particles-mean), in_cov_matrix) * (particles-mean)).sum(1) / 2)
        return result   # M array

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_folder', type = str, default = 'results_save/gaussian_figure/result_sg')
    parser.add_argument('--save_folder', type = str, default = './figures/2d_gaussian_appen/')
    parser.add_argument('--plot_size', type = float, default = 100)
    parser.add_argument('--min_ratio', type = float, default = 0.1)
    parser.add_argument('--max_ratio', type = float, default = 10)
    parser.add_argument('--alpha', type = float, default = 0.6)
    parser.add_argument('--suffix', type = str, default = 'pdf', choices = ['png', 'eps', 'pdf'])
    parser.add_argument('--dpi', type = int, default = 300)
    opts = parser.parse_args()
    sequence = ['results_5', 'results_10', 'results_20', 'results_50', 'results_100']
    x_axis = [5, 10, 20, 50, 100]
    f =  open('./load_results/plot_settings.yaml', 'r')
    plot_settings_common = yaml.load(f.read(), Loader=yaml.FullLoader)
    #plot_settings_this = plot_settings_common[opts.main_folder]
    log_folder_list = os.listdir(opts.main_folder)
    for name in log_folder_list:
        if os.path.isfile(os.path.join(opts.main_folder, name)):
            log_folder_list.remove(name)
    log_folder_list.sort()
    data = {}
    for k in sequence: data[k] = {}
    # load data
    for log_folder in log_folder_list:  # log_folder: different number of particles
        results_folder = os.listdir(os.path.join(opts.main_folder, log_folder))
        results_folder.sort()
        for result in results_folder:
            algo_name = result[0: result.find('_')]
            result_folder = os.path.join(opts.main_folder, log_folder, result)
            particles = np.load(os.path.join(result_folder, 'particles.npy'))
            mass = np.load(os.path.join(result_folder, 'mass.npy'))
            if not algo_name in data[log_folder]: 
                data[log_folder][algo_name] = [particles, mass]
            else:
                data[log_folder][algo_name].append([particles, mass])
    # plot particles
    for folder_name in sequence:
        save_folder_this = opts.save_folder + 'sg' +folder_name[folder_name.find('_'):]
        create_dirs_if_not_exist(save_folder_this)
        for algorithm in data[folder_name]:
            particle, mass = data[folder_name][algorithm]
            particle, mass = particle[-1], mass[-1]
            particle_num = len(mass)
            plt.figure(figsize=(6.4, 4.8), dpi = opts.dpi)
            x, y = torch.linspace(-6, 6, 121), torch.linspace(-6, 6, 121)
            grid_X, grid_Y = torch.meshgrid(x,y)
            loc = torch.cat([grid_X.unsqueeze(2), grid_Y.unsqueeze(2)], dim = 2)
            num = len(loc)
            pdf = pdf_calc(loc.view(-1, 2))
            pdf = pdf.view(num, num)
            # fig = plt.figure(num = 1)
            plt.contour(grid_X.cpu().numpy(), grid_Y.cpu().numpy(), pdf.cpu().numpy())
            # set particle size
            mass = np.clip(mass * particle_num, opts.min_ratio, opts.max_ratio)
            size_list = (mass * opts.plot_size).astype(np.int)
            plt.scatter(particle[:, 0], particle[:, 1], alpha = opts.alpha, s = size_list, c = 'r', label = plot_settings_common['label'][algorithm] + ', M = {}'.format(folder_name[folder_name.find('_')+ 1:]))
            plt.legend(fontsize = 25, loc = 2)
            plt.ylim([ - 6, 6])
            plt.xlim([ - 6, 6])
            plt.xticks([])
            plt.yticks([])
            plt.tick_params(labelsize = 22)
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.margins(0,0)
            if opts.suffix == 'eps':
                ax = plt.gca()
                ax.set_rasterized(True)
            plt.savefig(os.path.join(save_folder_this, '{}.{}'.format(algorithm, opts.suffix)), pad_inches = 0.0)
            plt.close()

