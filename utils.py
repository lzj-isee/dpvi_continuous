import torch, os, random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import logging
import sys

@torch.no_grad()
def check(particles, mass, curr_iter_count, logger = None):
    if particles.max() != particles.max(): 
        message = 'particle value Nan at iter %d'%curr_iter_count
        if logger: logger.error(message)
        raise ValueError(message)
    if mass.min() <= 0: 
        message = 'non-positive mass at iter %d'%curr_iter_count
        if logger: logger.error(message)
        raise ValueError(message)


def set_algorithm_save_name(opts):
    # create information
    if opts.algorithm in ['SGLD']:
        save_name = opts.algorithm + '_' + 'lr[{:.1e}]an[{:.1f}]S[{}]'.format(opts.lr, opts.anneal, opts.seed)
    elif opts.algorithm in ['HMC']:
        save_name = opts.algorithm + '_' + 'lr[{:.1e}]lf[{}]an[{:.1f}]S[{}]'.format(opts.lr, opts.leap_iter, opts.anneal, opts.seed)
    elif opts.algorithm in ['SVGD', 'BLOB', 'GFSD', 'KSDD']: 
        if not opts.bwType == 'fix':
            save_name = opts.algorithm + '_' + '{}_lr[{:.1e}]an[{:.1f}]S[{}]'.format(opts.bwType, opts.lr, opts.anneal, opts.seed)
        else:
            save_name = opts.algorithm + '_' + '{:.1e}_lr[{:.1e}]an[{:.1f}]S[{}]'.format(opts.bwVal, opts.lr, opts.anneal, opts.seed)
    elif opts.algorithm in ['BLOBCA', 'BLOBDK', 'GFSDCA', 'GFSDDK', 'KSDDCA', 'KSDDDK', 'SGLDDK']:
        if not opts.bwType == 'fix':
            save_name = opts.algorithm + '_' + '{}_lr[{:.1e}]al[{:.1e}]an[{:.1f}]S[{}]'.format(
                opts.bwType, opts.lr, opts.alpha, opts.anneal, opts.seed)
        else:
            save_name = opts.algorithm + '_' + '{:.1e}_lr[{:.1e}]al[{:.1e}]an[{:.1f}]S[{}]'.format(
                opts.bwVal, opts.lr, opts.alpha, opts.anneal, opts.seed)
    else: 
        raise NotImplementedError
    return save_name


def get_logger(opts, name, save_folder = None):
    if save_folder is None: # set the save_folder according to the algorithm
        save_folder = os.path.join(opts.save_folder, set_algorithm_save_name(opts))
    # creat log folder
    create_dirs_if_not_exist(save_folder)
    # clear previous log files
    clear_log(save_folder)
    # creat tensorboard SummaryWriter
    writer = SummaryWriter(log_dir = save_folder)
    # creat logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
    ch = logging.StreamHandler(stream = sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(save_folder, 'log.txt'), mode = 'w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # save the parameters
    save_settings(logger, vars(opts))
    return writer, logger, save_folder

def set_random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) 
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def save_settings(logger, settings):
    string = '\n'
    for key in settings: 
        string += ' - ' + key + ': ' + '{}'.format(settings[key])+' \n'
    logger.info(string)

def create_dirs_if_not_exist(dir_list):
    if isinstance(dir_list, list):
        for dir in dir_list:
            if not os.path.exists(dir):
                os.makedirs(dir)
    else:
        if not os.path.exists(dir_list):
            os.makedirs(dir_list)

def save_final_results(save_folder, result_dict):
    # save results
    with open(os.path.join(save_folder, 'results.txt'), mode='w') as f:
        for key in result_dict:
            f.write(key + ': ' + '{}'.format(result_dict[key])+ '\n')

def clear_log(save_folder):
    # clear log files
    if os.path.exists(save_folder):
        names = os.listdir(save_folder)
        for name in names:
            os.remove(save_folder+'/'+name)
        print('\n clear files in {}'.format(save_folder))
    else:
        pass