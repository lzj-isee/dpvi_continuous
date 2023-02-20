import argparse
import yaml
import os, torch
import numpy as np
import matplotlib.pyplot as plt
import pretty_errors
from tqdm import tqdm
import importlib
import sys
sys.path.append("/home/lzj/parvi_bd")
from dataloader import myDataLoader

if __name__ == "__main__":    
    edge = -1.1837590174581716
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_folder', type = str, default = 'results_save')
    parser.add_argument('--save_folder', type = str, default = './figures/multi_gaussian')
    # parser.add_argument('--dpi', type = int, default = 300)
    # for plotting particles
    # parser.add_argument('--task', type = str, default = 'multi_gaussian')
    # parser.add_argument('--dataset', type = str, default='--')
    # parser.add_argument('--particle_num', type = int, default = 128)
    # parser.add_argument('--model_dim', type = int, default = 10)
    # parser.add_argument('--batch_size', type = int, default = 1, help = 'invalid in gaussian and lr')
    # parser.add_argument('--device', type = str, default = 'cpu')
    # parser.add_argument('--seed', type = int, default = 9, help = 'random seed for algorithm')
    # parser.add_argument('--split_size', type = float, default = 0.1, help = 'split ratio for dataset')
    # parser.add_argument('--split_seed', type = int, default = 19, help = 'random seed to split dataset')
    opts = parser.parse_args()
    particle_num = ['num32', 'num64', 'num128', 'num256', 'num512']
    # scale = {'num32':(2.5, 4.5), 'num64':(2.3, 4.2), 'num128':(2.25, 4.0), 'num256':(2.0, 4.0), 'num512':(2.0, 4.0)}
    f =  open('./load_results/plot_settings.yaml', 'r')
    plot_settings = yaml.load(f.read(), Loader=yaml.FullLoader)
    # ================================================== calculate the mass ==========================================================
    save_folder = os.path.join(opts.save_folder, 'particles')
    # data_and_loader = myDataLoader(opts)
    # task = importlib.import_module('tasks.%s'%opts.task).__getattribute__(opts.task)(opts, info_source = data_and_loader)
    results_folder = os.listdir(os.path.join(opts.main_folder, 'results_mg_iter'))
    results_folder.sort()
    data = {}
    for k  in particle_num: data[k] = {}
    for num in data.keys():
        results_folder = os.listdir(os.path.join(opts.main_folder, 'results_mg_iter', num))
        results_folder.sort()
        for folder in results_folder:
            if 'S[0]' not in folder: continue
            method_name = folder[0: folder.find('_')]
            particles = np.load(os.path.join(opts.main_folder, 'results_mg_iter', num, folder, 'particles.npy'))
            mass = np.load(os.path.join(opts.main_folder, 'results_mg_iter', num, folder, 'mass.npy'))
            if not method_name in data:
                data[num][method_name] = [particles, mass]
    num_stat, mass_stat = {}, {}
    init_particles, init_mass = {}, {}
    for num in data.keys():
        init_particles, init_mass = data[num]['SVGD']
        init_particles, init_mass = init_particles[0], init_mass[0]
        left = init_particles[:, 0:2].sum(1) < edge
        right = np.bitwise_not(left)
        if not 'init' in num_stat:
            num_stat['init'] = [np.sum(left) / len(left)]
            mass_stat['init'] = [init_mass[left].sum()]
        else:
            num_stat['init'].append(np.sum(left) / len(left))
            mass_stat['init'].append(init_mass[left].sum())

        for method_name in data[num].keys():
            particles, mass = data[num][method_name]
            particles, mass = particles[-1], mass[-1]
            left = particles[:, 0:2].sum(1) < edge
            right = np.bitwise_not(left)
            if not method_name in num_stat:
                num_stat[method_name] = [np.sum(left) / len(left)]
                mass_stat[method_name] = [mass[left].sum()]
            else:
                num_stat[method_name].append(np.sum(left) / len(left))
                mass_stat[method_name].append(mass[left].sum())
        
    with open(os.path.join(opts.save_folder, 'multi_gaussian_num_mass.txt'), mode = 'w') as f:
        values = []
        for term in num_stat['init']:
            values.append(term)
            values.append(1 - term)
        f.write('num: {} & \t'.format('init') + '{:.3f} / {:.3f} & {:.3f} / {:.3f} & {:.3f} / {:.3f} & {:.3f} / {:.3f} & {:.3f} / {:.3f}'.format(*values) + r'\\' + '\n')
        values = []
        for term in mass_stat['init']:
            values.append(term)
            values.append(1 - term)
        f.write('mass: {} & \t'.format('init') + '{:.3f} / {:.3f} & {:.3f} / {:.3f} & {:.3f} / {:.3f} & {:.3f} / {:.3f} & {:.3f} / {:.3f}'.format(*values) + r'\\' + '\n')
        for name in plot_settings['order']:
            if name not in num_stat.keys(): continue
            values = []
            for term in num_stat[name]:
                values.append(term)
                values.append(1 - term)
            f.write('num: {} & \t'.format(plot_settings['label'][name]) + '{:.3f} / {:.3f} & {:.3f} / {:.3f} & {:.3f} / {:.3f} & {:.3f} / {:.3f} & {:.3f} / {:.3f}'.format(*values) + r'\\' + '\n')
            values = []
            for term in mass_stat[name]:
                values.append(term)
                values.append(1 - term)
            f.write('mass: {} & \t'.format(plot_settings['label'][name]) + '{:.3f} / {:.3f} & {:.3f} / {:.3f} & {:.3f} / {:.3f} & {:.3f} / {:.3f} & {:.3f} / {:.3f}'.format(*values) + r'\\' + '\n')
    

    # ========================================== iter ==================================================================================
    # # loading_results_iteration
    # main_folder = os.path.join(opts.main_folder, 'results_mg_iter')
    # data_iter = {}
    # for k  in particle_num: data_iter[k] = {}
    # for num in os.listdir(main_folder):
    #     results_folder = os.listdir(os.path.join(main_folder, num))
    #     results_folder.sort()
    #     for folder in results_folder:
    #         method_name = folder[0: folder.find('_')]
    #         value = np.load(os.path.join(main_folder, num, folder, 'w2.npy'))
    #         if not method_name in data_iter[num]:
    #             data_iter[num][method_name] = [value]
    #         else:
    #             data_iter[num][method_name].append(value)
    # # process_data
    # stds_iter, means_iter, last_iter = {}, {}, {}
    # for k in particle_num: stds_iter[k], means_iter[k] = {}, {}
    # for num in data_iter.keys():
    #     for method_name in data_iter[num].keys():
    #         data_iter[num][method_name] = np.array(data_iter[num][method_name])
    #         stds_iter[num][method_name] = np.std(data_iter[num][method_name], axis = 0, ddof = 1)
    #         means_iter[num][method_name] = np.mean(data_iter[num][method_name], axis = 0)
    #         if not method_name in last_iter:
    #             last_iter[method_name] = [means_iter[num][method_name][-1]]
    #         else:
    #             last_iter[method_name].append(means_iter[num][method_name][-1])
    # if not os.path.exists(opts.save_folder): os.makedirs(opts.save_folder)
    # with open(os.path.join(opts.save_folder, 'multi_gaussian_iter.txt'), mode = 'w') as f:
    #     for name in plot_settings['order']:
    #         if name not in last_iter.keys(): continue
    #         values = last_iter[name]
    #         f.write('{} & \t'.format(plot_settings['label'][name]) + '{:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f}'.format(*values) + r'\\' + '\n')
    # # plot_result_iter_seperate
    # for num in particle_num:
    #     plt.figure(figsize=(7.2, 4.8))
    #     for method_name in plot_settings['order']:
    #         if method_name not in last_iter.keys(): continue
    #         x_axis = np.linspace(0, 10000, len(means_iter[num][method_name]))
    #         plt.errorbar(
    #             x_axis, means_iter[num][method_name], yerr = stds_iter[num][method_name] / 5, capsize = 3, 
    #             color = plot_settings['color'][method_name], 
    #             linestyle = plot_settings['linestyle'][method_name], 
    #             label = plot_settings['label'][method_name], 
    #             alpha = 1.0, 
    #             linewidth = 1.0
    #         )
    #     plt.legend(fontsize = 10, ncol = 1, bbox_to_anchor=(1.25, 0.64), loc=5, borderaxespad = 0)
    #     plt.ylim(scale[num])
    #     plt.yscale('linear')
    #     plt.xscale('linear')
    #     plt.xlabel('Iterations, M = %s'%(num[3:]), {'size': 12})
    #     plt.ylabel('2-Wasserstein Distance', {'size': 12})
    #     plt.tick_params(labelsize = 12)
    #     plt.savefig(os.path.join(opts.save_folder, 'mg_%s_iter.pdf'%(num)), dpi = opts.dpi, bbox_inches = 'tight')
    #     plt.close()
    # # ========================================== time ==================================================================================
    # # loading_results_time
    # main_folder = os.path.join(opts.main_folder, 'results_mg_time')
    # data_time = {}
    # for k  in particle_num: data_time[k] = {}
    # for num in os.listdir(main_folder):
    #     results_folder = os.listdir(os.path.join(main_folder, num))
    #     results_folder.sort()
    #     for folder in results_folder:
    #         method_name = folder[0: folder.find('_')]
    #         value = np.load(os.path.join(main_folder, num, folder, 'w2.npy'))
    #         if not method_name in data_time[num]:
    #             data_time[num][method_name] = [value]
    #         else:
    #             data_time[num][method_name].append(value)
    # # process_data
    # stds_time, means_time, last_time = {}, {}, {}
    # for k in particle_num: stds_time[k], means_time[k] = {}, {}
    # for num in data_time.keys():
    #     for method_name in data_time[num].keys():
    #         data_time[num][method_name] = np.array(data_time[num][method_name])
    #         stds_time[num][method_name] = np.std(data_time[num][method_name], axis = 0, ddof = 1)
    #         means_time[num][method_name] = np.mean(data_time[num][method_name], axis = 0)
    #         if not method_name in last_time:
    #             last_time[method_name] = [means_time[num][method_name][-1]]
    #         else:
    #             last_time[method_name].append(means_time[num][method_name][-1])
    # if not os.path.exists(opts.save_folder): os.makedirs(opts.save_folder)
    # with open(os.path.join(opts.save_folder, 'multi_gaussian_time.txt'), mode = 'w') as f:
    #     for name in plot_settings['order']:
    #         if name not in last_time.keys(): continue
    #         values = last_time[name]
    #         f.write('{} & \t'.format(plot_settings['label'][name]) + '{:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f}'.format(*values) + r'\\' + '\n')
    # # plot_result_time_seperate
    # for num in particle_num:
    #     plt.figure(figsize=(7.2, 4.8))
    #     for method_name in plot_settings['order']:
    #         if method_name not in last_time.keys(): continue
    #         x_axis = np.linspace(0, 45, len(means_time[num][method_name]))
    #         plt.errorbar(
    #             x_axis, means_time[num][method_name], yerr = stds_time[num][method_name] / 5, capsize = 3, 
    #             color = plot_settings['color'][method_name], 
    #             linestyle = plot_settings['linestyle'][method_name], 
    #             label = plot_settings['label'][method_name], 
    #             alpha = 1.0, 
    #             linewidth = 1.0
    #         )
    #     plt.legend(fontsize = 10, ncol = 1, bbox_to_anchor=(1.25, 0.64), loc=5, borderaxespad = 0)
    #     plt.ylim(scale[num])
    #     plt.yscale('linear')
    #     plt.xscale('linear')
    #     plt.xlabel('Time (s), M = %s'%(num[3:]), {'size': 12})
    #     plt.ylabel('2-Wasserstein Distance', {'size': 12})
    #     plt.tick_params(labelsize = 12)
    #     plt.savefig(os.path.join(opts.save_folder, 'mg_%s_time.pdf'%(num)), dpi = opts.dpi, bbox_inches = 'tight')
    #     plt.close()
    # # ============================================= particle ===================================================================
    # # plot particles
    # save_folder = os.path.join(opts.save_folder, 'particles')
    # data_and_loader = myDataLoader(opts)
    # task = importlib.import_module('tasks.%s'%opts.task).__getattribute__(opts.task)(opts, info_source = data_and_loader)
    # results_folder = os.listdir(os.path.join(opts.main_folder, 'results_mg_iter'))
    # results_folder.sort()
    # data = {}
    # for k  in particle_num: data[k] = {}
    # for num in data.keys():
    #     results_folder = os.listdir(os.path.join(opts.main_folder, 'results_mg_iter', num))
    #     results_folder.sort()
    #     for folder in results_folder:
    #         if 'S[0]' not in folder: continue
    #         method_name = folder[0: folder.find('_')]
    #         particles = np.load(os.path.join(opts.main_folder, 'results_mg_iter', num, folder, 'particles.npy'))
    #         mass = np.load(os.path.join(opts.main_folder, 'results_mg_iter', num, folder, 'mass.npy'))
    #         if not method_name in data:
    #             data[num][method_name] = [particles, mass]
    # x, y = torch.linspace(-4, 4, 101, device = opts.device), torch.linspace(-4, 4, 101, device = opts.device)
    # grid_X, grid_Y = torch.meshgrid(x, y, indexing = 'ij')
    # loc = torch.cat([grid_X.unsqueeze(2), grid_Y.unsqueeze(2)], dim = 2)
    # num = len(loc)
    # log_pdf = - task.potential(loc.view(-1, 2)).view(num, num)
    # for num in particle_num:
    #     this_save_folder = os.path.join(save_folder, num)
    #     if not os.path.exists(this_save_folder): os.makedirs(this_save_folder)
    #     for method_name in tqdm(data[num].keys()):
    #         particle, mass = data[num][method_name]
    #         particle, mass = particle[-1], mass[-1]
    #         mass = np.clip(mass * particle.shape[0], task.min_ratio, task.max_ratio)
    #         size_list = (mass * task.plot_size).astype(int)
    #         fig = plt.figure(figsize=(4.8, 4.8))
    #         plt.scatter(particle[:, 0], particle[:, 1], alpha = 0.5, s = size_list, c = 'r',  zorder = 2, label = plot_settings['label'][method_name])
    #         plt.contour(grid_X.cpu().numpy(), grid_Y.cpu().numpy(), log_pdf.cpu().numpy(), levels = 30,  zorder = 1)
    #         plt.ylim([-4, 4])
    #         plt.xlim([-4, 4])
    #         plt.legend(fontsize = 16, loc = 3)
    #         plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    #         plt.savefig(os.path.join(this_save_folder, '%s.pdf'%(plot_settings['label'][method_name])), dpi = opts.dpi)
    #         plt.close()
    # # ================================================== 4 in 1 ====================================================================
    # order = ['GFSD', 'SGLD', 'GFSDCA', 'SGLDDK', 'GFSDDK', 'HMC', 'BLOB', 'SVGD', 'BLOBCA', 'BLOBDK', 'KSDD', 'KSDDCA', 'KSDDDK']
    # # plot_result_4in1
    # plt.figure(figsize = (7.2 * 2, 4.8 * 2))
    # plt.subplot(221)
    # for method_name in order:
    #     if method_name not in last_iter.keys(): continue
    #     x_axis = np.linspace(0, 10000, len(means_iter['num32'][method_name]))
    #     plt.errorbar(
    #         x_axis, means_iter['num32'][method_name], yerr = stds_iter['num32'][method_name] / 5, capsize = 3, 
    #         color = plot_settings['color'][method_name], 
    #         linestyle = plot_settings['linestyle'][method_name], 
    #         label = plot_settings['label'][method_name], 
    #         alpha = 1.0, 
    #         linewidth = 1.0
    #     )
    # plt.ylim(scale['num32'])
    # plt.yscale('linear')
    # plt.xscale('linear')
    # plt.xlabel('Iterations , M = 32', {'size': 12})
    # plt.ylabel('2-Wasserstein Distance', {'size': 12})
    # plt.tick_params(labelsize = 12)
    # plt.subplot(222)
    # for method_name in order:
    #     if method_name not in last_time.keys(): continue
    #     x_axis = np.linspace(0, 45, len(means_time['num32'][method_name]))
    #     plt.errorbar(
    #         x_axis, means_time['num32'][method_name], yerr = stds_time['num32'][method_name] / 5, capsize = 3, 
    #         color = plot_settings['color'][method_name], 
    #         linestyle = plot_settings['linestyle'][method_name], 
    #         label = plot_settings['label'][method_name], 
    #         alpha = 1.0, 
    #         linewidth = 1.0
    #     )
    # plt.ylim(scale['num32'])
    # plt.yscale('linear')
    # plt.xscale('linear')
    # plt.xlabel('Time (s) , M = 32', {'size': 12})
    # plt.ylabel('2-Wasserstein Distance', {'size': 12})
    # plt.tick_params(labelsize = 12)
    # plt.subplot(223)
    # for method_name in order:
    #     if method_name not in last_iter.keys(): continue
    #     x_axis = np.linspace(0, 10000, len(means_iter['num512'][method_name]))
    #     plt.errorbar(
    #         x_axis, means_iter['num512'][method_name], yerr = stds_iter['num512'][method_name] / 5, capsize = 3, 
    #         color = plot_settings['color'][method_name], 
    #         linestyle = plot_settings['linestyle'][method_name], 
    #         label = plot_settings['label'][method_name], 
    #         alpha = 1.0, 
    #         linewidth = 1.0
    #     )
    # plt.ylim(scale['num512'])
    # plt.yscale('linear')
    # plt.xscale('linear')
    # plt.xlabel('Iterations , M = 512', {'size': 12})
    # plt.ylabel('2-Wasserstein Distance', {'size': 12})
    # plt.tick_params(labelsize = 12)
    # plt.subplot(224)
    # for method_name in order:
    #     if method_name not in last_iter.keys(): continue
    #     x_axis = np.linspace(0, 45, len(means_iter['num512'][method_name]))
    #     plt.errorbar(
    #         x_axis, means_iter['num512'][method_name], yerr = stds_iter['num512'][method_name] / 5, capsize = 3, 
    #         color = plot_settings['color'][method_name], 
    #         linestyle = plot_settings['linestyle'][method_name], 
    #         label = plot_settings['label'][method_name], 
    #         alpha = 1.0, 
    #         linewidth = 1.0
    #     )
    # plt.ylim(scale['num512'])
    # plt.yscale('linear')
    # plt.xscale('linear')
    # plt.xlabel('Time (s) , M = 512', {'size': 12})
    # plt.ylabel('2-Wasserstein Distance', {'size': 12})
    # plt.tick_params(labelsize = 12)
    # plt.legend(fontsize = 10, ncol = 9, bbox_to_anchor=(-0.15, 2.4), loc=9, borderaxespad = 0)
    # plt.savefig(os.path.join(opts.save_folder, 'mg_4in1.pdf'), dpi = opts.dpi, bbox_inches = 'tight')
    # plt.close()