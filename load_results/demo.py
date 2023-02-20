import argparse
from tqdm import tqdm
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pretty_errors

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_folder', type = str, default = './results_demo')
    parser.add_argument('--suffix', type = str, default = 'png', choices = ['png', 'eps', 'pdf'])
    parser.add_argument('--dpi', type = int, default = 200)
    parser.add_argument('--sparse', type = float, default = 1.0)
    parser.add_argument('--x_max', type = int, default = 15000)
    opts = parser.parse_args()
    # f =  open('./load_results/plot_settings.yaml', 'r')
    # plot_settings_common = yaml.load(f.read(), Loader=yaml.FullLoader)
    # --------------------------- load resutls for meg dataset ---------------------------------------------
    subfolder_list = os.listdir(opts.result_folder)
    subfolder_list.sort()
    data = {}
    for subfoler in subfolder_list:
        method_name = subfoler[0: subfoler.find('_')]
        if subfoler.find('an[0.1]') > 0:
            method_name += ' + anneal'
        w2_list = np.load(os.path.join(opts.result_folder, subfoler, 'w2.npy'))
        if not method_name in data: 
            data[method_name] = [w2_list]
        else:
            data[method_name].append(w2_list)
    # process data
    mean, std, x_axis = {}, {}, {}
    for method_name in data.keys():
        data[method_name] = np.array(data[method_name])
        mean[method_name] = np.mean(data[method_name], axis = 0)
        std[method_name] = np.std(data[method_name], axis = 0, ddof = 1)
        point_num = int(len(np.mean(data[method_name], axis = 0))) # maximum iter
        index = np.linspace(0, point_num - 1, int(point_num * opts.sparse), dtype = int)
        mean[method_name] = mean[method_name][0:point_num][index]
        std[method_name] = std[method_name][0:point_num][index]
        x_axis[method_name] = np.linspace(0, opts.x_max, len(mean[method_name]))
    with open('./figures/demo.txt', mode = 'w') as f:
        for method_name in data.keys():
            if method_name not in mean.keys(): continue
            f.write('{}: \t\t'.format(method_name) + '{:.3e} & {:.3e}'.format(mean[method_name][-1], std[method_name][-1]) + '\n')
    plt.figure(figsize=(7.68, 4.8))
    for method_name in data.keys():
        plt.errorbar(x_axis[method_name], mean[method_name], yerr = std[method_name], capsize = 3, 
            # color = plot_settings_common['color'][name],
            # linestyle = plot_settings_common['linestyle'][name],
            label = method_name,
            alpha = 1.0, 
            linewidth = 1.0)
    # figure setting
    plt.legend(fontsize = 14, ncol = 4, bbox_to_anchor=(-0.15, 1.05), loc=3, borderaxespad = 0)
    # plt.ylim(0.11, 0.6)
    plt.yscale('log')
    plt.xscale('linear')
    plt.xlabel('Iteration', {'size': 18})
    plt.ylabel('2-Wasserstein Distance', {'size': 18})
    plt.tick_params(labelsize = 18)
    if opts.suffix == 'eps':
        ax = plt.gca()
        ax.set_rasterized(True)
    plt.savefig('./figures/demo_w2.{}'.format(opts.suffix), dpi = opts.dpi, bbox_inches = 'tight')
    plt.close()
    # # plot ksd
    # plt.figure(figsize=(7.68, 4.8))
    # for name in plot_settings_common['order']:
    #     if name not in data.keys(): continue
    #     plt.plot(x_axis, mean_ksd[name], 
    #         color = plot_settings_common['color'][name],
    #         linestyle = plot_settings_common['linestyle'][name],
    #         label = plot_settings_common['label'][name],
    #         alpha = 1.0, 
    #         linewidth = 1.0)
    # # figure setting
    # plt.legend(fontsize = 14, ncol = 4, bbox_to_anchor=(-0.15, 1.02), loc=3, borderaxespad = 0)
    # plt.ylim(4e-4, 0.30)
    # plt.yscale('log')
    # plt.xscale('linear')
    # plt.xlabel('Iteration', {'size': 18})
    # plt.ylabel('KSD', {'size': 18})
    # plt.tick_params(labelsize = 18)
    # if opts.suffix == 'eps':
    #     ax = plt.gca()
    #     ax.set_rasterized(True)
    # plt.savefig('./figures/gp_ksd.{}'.format(opts.suffix), dpi = opts.dpi, bbox_inches = 'tight')
    # plt.close()