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
    #parser.add_argument('--main_folder', type = str, default = 'results_save/result_electrical')
    #parser.add_argument('--term', type = str, default = 'test_rmse')
    parser.add_argument('--suffix', type = str, default = 'png', choices = ['png', 'eps', 'pdf'])
    parser.add_argument('--dpi', type = int, default = 600)
    parser.add_argument('--sparse_meg', type = float, default = 1.0)
    parser.add_argument('--x_max_meg', type = int, default = 10000)
    opts = parser.parse_args()
    f =  open('./load_results/plot_settings.yaml', 'r')
    plot_settings_common = yaml.load(f.read(), Loader=yaml.FullLoader)
    # --------------------------- load resutls for meg dataset ---------------------------------------------
    log_folder = './results_save/ica/meg'
    log_folder_list = os.listdir(log_folder)
    data = {}
    for log in tqdm(log_folder_list):
        algo_name = log[0: log.find('_')]
        result_folder = os.path.join(log_folder, log)
        value = np.load(os.path.join(result_folder, 'test_nll.npy'))
        if not algo_name in data:
            data[algo_name] = [value]
        else:
            data[algo_name].append(value)
    # process data
    stds, means = {}, {}
    for name in data.keys():
        data[name] = np.array(data[name])
        stds[name] = np.std(data[name], axis = 0, ddof = 1)
        means[name] = np.mean(data[name], axis = 0)
        point_num = int(len(np.mean(data[name], axis = 0)))
        index = np.linspace(0, point_num - 1, int(point_num * opts.sparse_meg), dtype = int)
        means[name] = means[name][0:point_num][index]
        x_axis = np.linspace(0, opts.x_max_meg, len(means[name]))
    plt.figure(figsize=(7.68, 4.8))
    for name in plot_settings_common['order']:
        if name not in data.keys(): continue
        if name in 'KSDDBD': continue
        plt.plot(x_axis, means[name], 
            color = plot_settings_common['color'][name],
            linestyle = plot_settings_common['linestyle'][name],
            label = plot_settings_common['label'][name],
            alpha = 0.9, 
            linewidth = 1.0)
    # figure setting
    plt.legend(fontsize = 18)
    plt.ylim(11, 18)
    plt.yscale('linear')
    plt.xscale('linear')
    plt.xlabel('Iteration', {'size': 18})
    plt.ylabel('Test Negative log likelihood', {'size': 18})
    plt.tick_params(labelsize = 18)
    if opts.suffix == 'eps':
        ax = plt.gca()
        ax.set_rasterized(True)
    plt.savefig('./figures/ica_meg_1.{}'.format(opts.suffix), dpi = opts.dpi, bbox_inches = 'tight')
    plt.close()
    # plot KSDD and KSDDBD
    plt.figure(figsize=(7.68, 4.8))
    for name in plot_settings_common['order']:
        if name not in data.keys(): continue
        if not name in 'KSDDBD': continue
        plt.plot(x_axis, means[name], 
            color = plot_settings_common['color'][name],
            linestyle = plot_settings_common['linestyle'][name],
            label = plot_settings_common['label'][name],
            alpha = 0.9, 
            linewidth = 1.0)
    # figure setting
    plt.legend(fontsize = 18)
    plt.ylim(35, 37)
    plt.yscale('linear')
    plt.xscale('linear')
    plt.xlabel('Iteration', {'size': 18})
    plt.ylabel('Test Negative log likelihood', {'size': 18})
    plt.tick_params(labelsize = 18)
    if opts.suffix == 'eps':
        ax = plt.gca()
        ax.set_rasterized(True)
    plt.savefig('./figures/ica_meg_2.{}'.format(opts.suffix), dpi = opts.dpi, bbox_inches = 'tight')
    plt.close()
    
    # --------------------------- load results for artificial dataset --------------------------------------
    log_folder = './results_save/ica/art'

    