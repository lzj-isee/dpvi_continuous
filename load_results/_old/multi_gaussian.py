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
    parser.add_argument('--main_folder', type = str, default = 'results_save/result_mg')
    parser.add_argument('--term', type = str, default = 'W2')
    parser.add_argument('--suffix', type = str, default = 'pdf', choices = ['png', 'eps', 'pdf'])
    parser.add_argument('--dpi', type = int, default = 300)
    opts = parser.parse_args()
    sequence = ['results_5', 'results_10', 'results_20', 'results_50', 'results_100']
    x_axis = [5, 10, 20, 50, 100]
    f =  open('./load_results/plot_settings.yaml', 'r')
    plot_settings_common = yaml.load(f.read(), Loader=yaml.FullLoader)
    plot_settings_this = plot_settings_common[opts.main_folder]
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
            value = np.load(os.path.join(result_folder, 'w2_values.npy'))
            if not algo_name in data[log_folder]: 
                data[log_folder][algo_name] = [value]
            else:
                data[log_folder][algo_name].append(value)
    # process data
    stds, means, last_iter = {}, {}, {}
    for k in sequence: stds[k], means[k] = {}, {}
    for setting in data.keys():
        for name in data[setting].keys():
            data[setting][name] = np.array(data[setting][name])
            stds[setting][name] = np.std(data[setting][name], axis = 0, ddof = 1)
            means[setting][name] = np.mean(data[setting][name], axis = 0)
    for setting in data.keys():
        for name in data[setting].keys():
            if not name in last_iter:
                last_iter[name] = [means[setting][name][-1]]
            else:
                last_iter[name].append(means[setting][name][-1])
    with open('./figures/multi_gaussian.md', mode = 'w') as f:
        for name in plot_settings_common['order']:
            if name not in last_iter.keys(): continue
            values = last_iter[name]
            f.write('{}: \t'.format(plot_settings_common['label'][name]) + '{:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f}'.format(*values) + '\n')
    debug = 1
    # start plotting
    # plt.figure(figsize=(7.68, 4.8))
    # for name in plot_settings_common['order']:
    #     if name not in last_iter.keys(): continue
    #     plt.plot(x_axis, last_iter[name], 
    #         color = plot_settings_common['color'][name],
    #         linestyle = plot_settings_common['linestyle'][name],
    #         label = plot_settings_common['label'][name],
    #         alpha = 1.0, 
    #         marker='o', markersize=2, 
    #         linewidth = 1.0)
    # # figure setting
    # plt.xticks(ticks = x_axis, labels = x_axis)
    # plt.grid()
    # plt.legend(fontsize = 14, ncol = 4, bbox_to_anchor=(-0.15, 1.02), loc=3, borderaxespad = 0)
    # plt.ylim(plot_settings_this['y_min'][opts.term],plot_settings_this['y_max'][opts.term])
    # plt.yscale(plot_settings_this['y_scale'][opts.term])
    # plt.xlabel(plot_settings_this['x_label'], {'size': 18})
    # plt.ylabel(plot_settings_this['y_label'][opts.term], {'size': 18})
    # if plot_settings_this['use_tick'] and plot_settings_this['y_scale'][opts.term] != 'log':
    #     ax = plt.gca()
    #     ax.ticklabel_format(style = 'sci', axis = 'y', scilimits = (-2,2))
    # #plt.tight_layout()
    # plt.tick_params(labelsize = 18)
    # if opts.suffix == 'eps':
    #     ax = plt.gca()
    #     ax.set_rasterized(True)
    # plt.savefig('./figures/{}.{}'.format(
    #     plot_settings_this['output'][opts.term], opts.suffix), dpi = opts.dpi, bbox_inches = 'tight')
    # plt.close()


