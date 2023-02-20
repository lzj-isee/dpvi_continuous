import argparse
from tqdm import tqdm
import yaml
import os
import numpy as np
import pretty_errors

def _loading(opts, term: str, plot_settings):
    log_folder_list = os.listdir(opts.main_folder)
    for name in log_folder_list:
        if os.path.isfile(os.path.join(opts.main_folder, name)):
            log_folder_list.remove(name)
    log_folder_list.sort()
    data = {}
    # load data
    for log_folder in tqdm(log_folder_list):
        algo_name = log_folder[0:log_folder.find('_')]
        result_folder = os.path.join(opts.main_folder, log_folder)
        value = np.load(os.path.join(result_folder, '%s.npy'%(term)))
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
    if not os.path.exists(opts.save_folder): os.makedirs(opts.save_folder)
    with open(os.path.join(opts.save_folder, opts.save_name + '_%s'%(term) + '.txt'), mode = 'w') as f:
        for name in plot_settings['order']:
            if name not in data.keys(): continue
            f.write('{}: '.format(plot_settings['label'][name]) + '{:.3e} +/- {:.3e}'.format(means[name][-1], stds[name][-1])+ '\n')

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_folder', type = str, default = 'results_save/results_winered')
    parser.add_argument('--save_name', type = str, default = 'winered')
    parser.add_argument('--save_folder', type = str, default = './figures/bnn_regression')
    parser.add_argument('--suffix', type = str, default = 'png', choices = ['png', 'eps', 'pdf'])
    opts = parser.parse_args()
    f =  open('./load_results/plot_settings.yaml', 'r')
    plot_settings = yaml.load(f.read(), Loader=yaml.FullLoader)
    _loading(opts, 'test_rmse', plot_settings)
    _loading(opts, 'test_nll', plot_settings)
    _loading(opts, 'test_nll_con', plot_settings)
    




    # # start plotting
    # plt.figure(figsize=(7.68, 4.8))
    # for name in plot_settings_common['order']:
    #     if name not in data.keys(): continue
    #     plt.plot(x_axis, means[name], 
    #         color = plot_settings_common['color'][name],
    #         linestyle = plot_settings_common['linestyle'][name],
    #         label = plot_settings_common['label'][name],
    #         alpha = 0.9, 
    #         linewidth = 1.0)
    # # figure setting
    # plt.legend(fontsize = 18)
    # plt.ylim(plot_settings_this['y_min'][opts.term],plot_settings_this['y_max'][opts.term])
    # plt.yscale(plot_settings_this['y_scale'][opts.term])
    # plt.xscale(plot_settings_this['x_scale'][opts.term])
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