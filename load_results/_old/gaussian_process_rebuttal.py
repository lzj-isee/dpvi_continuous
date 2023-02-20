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
    parser.add_argument('--suffix', type = str, default = 'pdf', choices = ['png', 'eps', 'pdf'])
    parser.add_argument('--dpi', type = int, default = 300)
    parser.add_argument('--sparse_iter', type = float, default = 0.5)
    parser.add_argument('--x_max_iter', type = int, default = 6000)
    parser.add_argument('--sparse_time', type = float, default = 0.3)
    parser.add_argument('--x_max_time', type = int, default = 60)
    opts = parser.parse_args()
    plt.figure(figsize=(7.20, 7.20))
    # ------------------------------------------------ plot iter ----------------------------------------------------------
    f =  open('./load_results/plot_settings.yaml', 'r')
    plot_settings_common = yaml.load(f.read(), Loader=yaml.FullLoader)
    log_folder = './results_save/gaussian_process'
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
    std_w2, std_ksd = {}, {}
    for name in data.keys():
        data[name]['w2'] = np.array(data[name]['w2'])
        data[name]['ksd'] = np.array(data[name]['ksd'])
        mean_w2[name] = np.mean(data[name]['w2'], axis = 0)
        mean_ksd[name] = np.mean(data[name]['ksd'], axis = 0)
        std_w2[name] = np.std(data[name]['w2'], axis = 0, ddof = 1)
        std_ksd[name] = np.std(data[name]['ksd'], axis = 0, ddof = 1)
        point_num = int(len(np.mean(data[name]['w2'], axis = 0)))
        index = np.linspace(0, point_num - 1, int(point_num * opts.sparse_iter), dtype = int)
        mean_w2[name] = mean_w2[name][0:point_num][index]
        mean_ksd[name] = mean_ksd[name][0:point_num][index]
        std_w2[name] = std_w2[name][0:point_num][index]
        std_ksd[name] = std_ksd[name][0:point_num][index]
        x_axis = np.linspace(0, opts.x_max_iter, len(mean_w2[name]))
    plt.subplot(2,1,1)
    for name in plot_settings_common['order']:
        if name not in data.keys(): continue
        plt.errorbar(x_axis, mean_w2[name], yerr = std_w2[name], capsize = 3, 
            color = plot_settings_common['color'][name],
            linestyle = plot_settings_common['linestyle'][name],
            label = plot_settings_common['label'][name],
            alpha = 1.0, 
            linewidth = 1.0)
    # figure setting
    plt.legend(fontsize = 10, ncol = 1, bbox_to_anchor=(1.25, 0.5), loc=5, borderaxespad = 0)
    plt.ylim(0.11, 0.4)
    plt.yscale('log')
    plt.xscale('linear')
    plt.xlabel('Iteration', {'size': 12})
    plt.ylabel('2-Wasserstein Distance', {'size': 12})
    plt.tick_params(labelsize = 8)
    # --------------------------------------------------- plot time -----------------------------------------------------
    f =  open('./load_results/plot_settings.yaml', 'r')
    plot_settings_common = yaml.load(f.read(), Loader=yaml.FullLoader)
    log_folder = './results_save/gaussian_process_time'
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
    std_w2, std_ksd = {}, {}
    point_num, index = {}, {}
    x_axis = {}
    for name in data.keys():
        data[name]['w2'] = np.array(data[name]['w2'])
        data[name]['ksd'] = np.array(data[name]['ksd'])
        mean_w2[name] = np.mean(data[name]['w2'], axis = 0)
        mean_ksd[name] = np.mean(data[name]['ksd'], axis = 0)
        std_w2[name] = np.std(data[name]['w2'], axis = 0, ddof = 1)
        std_ksd[name] = np.std(data[name]['ksd'], axis = 0, ddof = 1)
        point_num[name] = int(len(np.mean(data[name]['w2'], axis = 0)))
        index[name] = np.linspace(0, point_num[name] - 1, int(point_num[name] * opts.sparse_time), dtype = int)
        mean_w2[name] = mean_w2[name][0:point_num[name]][index[name]]
        mean_ksd[name] = mean_ksd[name][0:point_num[name]][index[name]]
        std_w2[name] = std_w2[name][0:point_num[name]][index[name]]
        std_ksd[name] = std_ksd[name][0:point_num[name]][index[name]]
        x_axis[name] = np.linspace(0, opts.x_max_time, len(mean_w2[name]))
    plt.subplot(2,1,2)
    for name in plot_settings_common['order']:
        if name not in data.keys(): continue
        plt.errorbar(x_axis[name], mean_w2[name], yerr = std_w2[name], capsize = 3, 
            color = plot_settings_common['color'][name],
            linestyle = plot_settings_common['linestyle'][name],
            label = plot_settings_common['label'][name],
            alpha = 1.0, 
            linewidth = 1.0)
    # figure setting
    # plt.legend(fontsize = 14, ncol = 4, bbox_to_anchor=(-0.15, 1.05), loc=3, borderaxespad = 0)
    plt.ylim(0.11, 0.4)
    plt.yscale('log')
    plt.xscale('linear')
    plt.xlabel('Time (s)', {'size': 12})
    plt.ylabel('2-Wasserstein Distance', {'size': 12})
    plt.tick_params(labelsize = 8)
    # ------------------------------------------------ save ------------------------------------------------------------------
    if opts.suffix == 'eps':
        ax = plt.gca()
        ax.set_rasterized(True)
    plt.savefig('./figures/gp_w2_rebuttal.{}'.format(opts.suffix), dpi = opts.dpi, bbox_inches = 'tight')
    plt.close()