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

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', type = str, default = 'png', choices = ['png', 'eps', 'pdf'])
    parser.add_argument('--dpi', type = int, default = 300)
    parser.add_argument('--sparse', type = float, default = 1.0)
    parser.add_argument('--x_max', type = int, default = 6000)
    opts = parser.parse_args()
    f =  open('./load_results/plot_settings_gp_fm.yaml', 'r')
    plot_settings_common = yaml.load(f.read(), Loader=yaml.FullLoader)
    # --------------------------- load resutls for meg dataset ---------------------------------------------
    log_folder = './results_save/gaussian_process_fm'
    log_folder_list = os.listdir(log_folder)
    data = {}
    for log in tqdm(log_folder_list):
        algo_name = log[0: log.find('_')]
        result_folder = os.path.join(log_folder, log)
        w2_value = np.load(os.path.join(result_folder, 'w2_values.npy'))
        ksd_value = np.load(os.path.join(result_folder, 'ksd_values.npy'))
        if not algo_name in data:
            data[algo_name] = {'w2': [w2_value], 'ksd': [ksd_value]}
        else:
            data[algo_name]['w2'].append(w2_value)
            data[algo_name]['ksd'].append(ksd_value)
    # process data
    mean_w2, mean_ksd = {}, {}
    for name in data.keys():
        data[name]['w2'] = np.array(data[name]['w2'])
        data[name]['ksd'] = np.array(data[name]['ksd'])
        mean_w2[name] = np.mean(data[name]['w2'], axis = 0)
        mean_ksd[name] = np.mean(data[name]['ksd'], axis = 0)
        point_num = int(len(np.mean(data[name]['w2'], axis = 0)))
        index = np.linspace(0, point_num - 1, int(point_num * opts.sparse), dtype = int)
        mean_w2[name] = mean_w2[name][0:point_num][index]
        mean_ksd[name] = mean_ksd[name][0:point_num][index]
        x_axis = np.linspace(0, opts.x_max, len(mean_w2[name]))
    # plot w2
    plt.figure(figsize=(7.68, 4.8))
    for name in plot_settings_common['order']:
        if name not in data.keys(): continue
        plt.plot(x_axis, mean_w2[name], 
            color = plot_settings_common['color'][name],
            linestyle = plot_settings_common['linestyle'][name],
            label = plot_settings_common['label'][name],
            alpha = 1.0, 
            linewidth = 1.0)
    # figure setting
    plt.legend(fontsize = 14, ncol = 4, bbox_to_anchor=(-0.15, 1.02), loc=3, borderaxespad = 0)
    plt.ylim(0.11, 0.6)
    plt.yscale('log')
    plt.xscale('linear')
    plt.xlabel('Iteration', {'size': 18})
    plt.ylabel('Log 2-Wasserstein Distance', {'size': 18})
    plt.tick_params(labelsize = 18)
    if opts.suffix == 'eps':
        ax = plt.gca()
        ax.set_rasterized(True)
    plt.savefig('./figures/gp_fm_w2.{}'.format(opts.suffix), dpi = opts.dpi, bbox_inches = 'tight')
    plt.close()
    # plot ksd
    plt.figure(figsize=(7.68, 4.8))
    for name in plot_settings_common['order']:
        if name not in data.keys(): continue
        plt.plot(x_axis, mean_ksd[name], 
            color = plot_settings_common['color'][name],
            linestyle = plot_settings_common['linestyle'][name],
            label = plot_settings_common['label'][name],
            alpha = 1.0, 
            linewidth = 1.0)
    # figure setting
    plt.legend(fontsize = 14, ncol = 4, bbox_to_anchor=(-0.15, 1.02), loc=3, borderaxespad = 0)
    plt.ylim(4e-4, 0.30)
    plt.yscale('log')
    plt.xscale('linear')
    plt.xlabel('Iteration', {'size': 18})
    plt.ylabel('Log KSD', {'size': 18})
    plt.tick_params(labelsize = 18)
    if opts.suffix == 'eps':
        ax = plt.gca()
        ax.set_rasterized(True)
    plt.savefig('./figures/gp_fm_ksd.{}'.format(opts.suffix), dpi = opts.dpi, bbox_inches = 'tight')
    plt.close()