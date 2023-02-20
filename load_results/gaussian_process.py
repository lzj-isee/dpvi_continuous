import argparse
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
import importlib
import sys
sys.path.append("/home/lzj/hdd_files/code/parvi_bd")
from dataloader import myDataLoader
import torch
import pretty_errors
from tqdm import tqdm

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_folder', type = str, default = 'results_save')
    # parser.add_argument('--x_axis_max', type = int, default = 300, choices = [10000, 300])
    # parser.add_argument('--info', type = str, default = 'time', choices = ['iter, time'])
    parser.add_argument('--save_folder', type = str, default = './figures/gaussian_process')
    # parser.add_argument('--suffix', type = str, default = 'png', choices = ['png', 'eps', 'pdf'])
    parser.add_argument('--dpi', type = int, default = 300)
    # for plotting particles
    parser.add_argument('--task', type = str, default = 'gaussian_process')
    parser.add_argument('--dataset', type = str, default='lidar')
    parser.add_argument('--particle_num', type = int, default = 128)
    parser.add_argument('--batch_size', type = int, default = 1, help = 'invalid in gaussian and lr')
    parser.add_argument('--device', type = str, default = 'cpu')
    parser.add_argument('--seed', type = int, default = 9, help = 'random seed for algorithm')
    parser.add_argument('--split_size', type = float, default = 0.1, help = 'split ratio for dataset')
    parser.add_argument('--split_seed', type = int, default = 19, help = 'random seed to split dataset')
    opts = parser.parse_args()
    f =  open('./load_results/plot_settings.yaml', 'r')
    plot_settings = yaml.load(f.read(), Loader=yaml.FullLoader)
    # =================================================== iter ==========================================================================
    # loading results
    main_folder = os.path.join(opts.main_folder, 'results_gp_iter')
    data_iter = {}
    results_folder = os.listdir(main_folder)
    results_folder.sort()
    for folder in results_folder:
        method_name = folder[0: folder.find('_')]
        value = np.load(os.path.join(main_folder, folder, 'w2.npy'))
        if not method_name in data_iter:
            data_iter[method_name] = [value]
        else:
            data_iter[method_name].append(value)
    # process data
    stds_iter, means_iter, last_iter = {}, {}, {}
    for method_name in data_iter.keys():
        data_iter[method_name] = np.array(data_iter[method_name])
        stds_iter[method_name] = np.std(data_iter[method_name], axis = 0, ddof = 1)
        means_iter[method_name] = np.mean(data_iter[method_name], axis = 0)
        if not method_name in last_iter:
            last_iter[method_name] = [means_iter[method_name][-1]]
        else:
            last_iter[method_name].append(means_iter[method_name][-1])
    if not os.path.exists(opts.save_folder): os.makedirs(opts.save_folder)
    with open(os.path.join(opts.save_folder, 'gp_iter.txt'), mode = 'w') as f:
        for name in plot_settings['order']:
            if name not in last_iter.keys(): continue
            values = last_iter[name]
            f.write('{} & \t'.format(plot_settings['label'][name]) + '{:.3f}'.format(*values) + r'\\' + '\n')
    # plot_result
    plt.figure(figsize=(7.2, 4.8))
    for method_name in plot_settings['order']:
        if method_name not in last_iter.keys(): continue
        x_axis = np.linspace(0, 10000, len(means_iter[method_name]))
        plt.errorbar(
            x_axis, means_iter[method_name], yerr = stds_iter[method_name], capsize = 3, 
            color = plot_settings['color'][method_name], 
            linestyle = plot_settings['linestyle'][method_name], 
            label = plot_settings['label'][method_name], 
            alpha = 1.0, 
            linewidth = 1.0
        )
    plt.legend(fontsize = 10, ncol = 1, bbox_to_anchor=(1.25, 0.64), loc=5, borderaxespad = 0)
    plt.ylim(0.1, 0.5)
    plt.yscale('log')
    plt.xscale('linear')
    plt.xlabel('Iterations', {'size': 12})
    plt.ylabel('2-Wasserstein Distance', {'size': 12})
    plt.tick_params(labelsize = 12)
    plt.savefig(os.path.join(opts.save_folder, 'w2_iter.pdf'), dpi = opts.dpi, bbox_inches = 'tight')
    plt.close()
    # =================================================== time ==========================================================================
    # loading results
    main_folder = os.path.join(opts.main_folder, 'results_gp_time')
    data_time = {}
    results_folder = os.listdir(main_folder)
    results_folder.sort()
    for folder in results_folder:
        method_name = folder[0: folder.find('_')]
        value = np.load(os.path.join(main_folder, folder, 'w2.npy'))
        if not method_name in data_time:
            data_time[method_name] = [value]
        else:
            data_time[method_name].append(value)
    # process data
    stds_time, means_time, last_time = {}, {}, {}
    for method_name in data_iter.keys():
        data_time[method_name] = np.array(data_time[method_name])
        stds_time[method_name] = np.std(data_time[method_name], axis = 0, ddof = 1)
        means_time[method_name] = np.mean(data_time[method_name], axis = 0)
        if not method_name in last_time:
            last_time[method_name] = [means_time[method_name][-1]]
        else:
            last_time[method_name].append(means_time[method_name][-1])
    if not os.path.exists(opts.save_folder): os.makedirs(opts.save_folder)
    with open(os.path.join(opts.save_folder, 'gp_time.txt'), mode = 'w') as f:
        for name in plot_settings['order']:
            if name not in last_time.keys(): continue
            values = last_time[name]
            f.write('{} & \t'.format(plot_settings['label'][name]) + '{:.3f}'.format(*values) + r'\\' + '\n')
    # plot_result
    plt.figure(figsize=(7.2, 4.8))
    for method_name in plot_settings['order']:
        if method_name not in last_time.keys(): continue
        x_axis = np.linspace(0, 45, len(means_time[method_name]))
        plt.errorbar(
            x_axis, means_time[method_name], yerr = stds_time[method_name], capsize = 3, 
            color = plot_settings['color'][method_name], 
            linestyle = plot_settings['linestyle'][method_name], 
            label = plot_settings['label'][method_name], 
            alpha = 1.0, 
            linewidth = 1.0
        )
    plt.legend(fontsize = 10, ncol = 1, bbox_to_anchor=(1.25, 0.64), loc=5, borderaxespad = 0)
    plt.ylim(0.1, 0.5)
    plt.yscale('log')
    plt.xscale('linear')
    plt.xlabel('Time (s)', {'size': 12})
    plt.ylabel('2-Wasserstein Distance', {'size': 12})
    plt.tick_params(labelsize = 12)
    plt.savefig(os.path.join(opts.save_folder, 'w2_time.pdf'), dpi = opts.dpi, bbox_inches = 'tight')
    plt.close()
    # ================================================= particles =======================================================================
    # plot particles
    data_and_loader = myDataLoader(opts)
    task = importlib.import_module('tasks.%s'%opts.task).__getattribute__(opts.task)(opts, info_source = data_and_loader)
    results_folder = os.listdir(os.path.join(opts.main_folder, 'results_gp_iter'))
    results_folder.sort()
    data = {}
    for folder in results_folder:
        if 'S[0]' not in folder: continue
        method_name = folder[0: folder.find('_')]
        particles = np.load(os.path.join(opts.main_folder, 'results_gp_iter', folder, 'particles.npy'))
        mass = np.load(os.path.join(opts.main_folder, 'results_gp_iter', folder, 'mass.npy'))
        if not method_name in data:
            data[method_name] = [particles, mass]
    x, y = torch.linspace(-5, 5, 100, device = opts.device), torch.linspace(-13, -7, 100, device = opts.device)
    grid_X, grid_Y = torch.meshgrid(x, y, indexing = 'ij')
    loc = torch.cat([grid_X.unsqueeze(2), grid_Y.unsqueeze(2)], dim = 2)
    num = len(loc)
    log_pdf = - task.potential(loc.view(-1, 2)).view(num, num)
    for method_name in tqdm(data.keys()):
        particle, mass = data[method_name]
        particle, mass = particle[-1], mass[-1]
        mass = np.clip(mass * particle.shape[0], task.min_ratio, task.max_ratio)
        size_list = (mass * task.plot_size).astype(int)
        fig = plt.figure(figsize=(4.8, 2.4))
        plt.scatter(particle[:, 0], particle[:, 1], alpha = 0.5, s = size_list, c = 'r',  zorder = 2, label = plot_settings['label'][method_name])
        plt.contour(grid_X.cpu().numpy(), grid_Y.cpu().numpy(), log_pdf.cpu().numpy(), levels = 50,  zorder = 1)
        plt.legend(fontsize = 18, loc = 3)
        plt.ylim([ - 13, -8])
        plt.xlim([ - 4.5, 4.0])
        plt.xticks([])
        plt.yticks([])
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.savefig(os.path.join(opts.save_folder, '%s.pdf'%(plot_settings['label'][method_name])), dpi = opts.dpi)
        plt.close()
    # ===================================================== 2 in 1 ==================================================================
    # order = ['GFSD', 'SGLD', 'GFSDCA', 'SGLDDK', 'GFSDDK', 'HMC', 'BLOB', 'SVGD', 'BLOBCA', 'BLOBDK', 'KSDD', 'KSDDCA', 'KSDDDK']
    order = ['GFSD', 'GFSDCA', 'GFSDDK', 'BLOB', 'BLOBCA', 'BLOBDK', 'KSDD', 'KSDDCA', 'KSDDDK', 'SGLD', 'SGLDDK', 'HMC', 'SVGD']
    # plot_result_4in1
    plt.figure(figsize = (7.4 * 2, 4.0 * 1))
    plt.subplot(121)
    for method_name in order:
        if method_name not in last_iter.keys(): continue
        x_axis = np.linspace(0, 10000, len(means_iter[method_name]))
        plt.errorbar(
            x_axis, means_iter[method_name], yerr = stds_iter[method_name], capsize = 3, 
            color = plot_settings['color'][method_name], 
            linestyle = plot_settings['linestyle'][method_name], 
            label = plot_settings['label'][method_name], 
            alpha = 1.0, 
            linewidth = 1.0
        )
    plt.ylim(0.12, 0.4)
    plt.yscale('log')
    plt.xscale('linear')
    plt.xlabel('Iterations', {'size': 18})
    plt.ylabel('2-Wasserstein Distance', {'size': 18})
    plt.tick_params(labelsize = 18)
    plt.subplot(122)
    for method_name in order:
        if method_name not in last_time.keys(): continue
        x_axis = np.linspace(0, 300, len(means_time[method_name]))
        plt.errorbar(
            x_axis, means_time[method_name], yerr = stds_time[method_name], capsize = 3, 
            color = plot_settings['color'][method_name], 
            linestyle = plot_settings['linestyle'][method_name], 
            label = plot_settings['label'][method_name], 
            alpha = 1.0, 
            linewidth = 1.0
        )
    plt.ylim(0.12, 0.4)
    plt.yscale('log')
    plt.xscale('linear')
    plt.xlabel('Time (s)', {'size': 18})
    plt.ylabel('2-Wasserstein Distance', {'size': 18})
    plt.tick_params(labelsize = 18)
    plt.legend(fontsize = 18, ncol = 5, bbox_to_anchor=(-0.13, 1.4), loc=9, borderaxespad = 0)
    plt.savefig(os.path.join(opts.save_folder, 'gp_2in1.pdf'), dpi = opts.dpi, bbox_inches = 'tight')
    plt.close()