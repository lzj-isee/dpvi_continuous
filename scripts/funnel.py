import os 

# extra tunning is required for KSDD-type methods
# how many reference should we use? 5000? 10000? 20000? 

common_settings = ' --task funnel --model_dim 10 --max_iter 6000 --eval_interval 300 --particle_num 512 --save_folder results'
for i in range(0, 1):
    os.system('python3 main_iter.py --algorithm SVGD --lr 0.1 --bwType med --seed {}'.format(i) + common_settings)
    os.system('python3 main_iter.py --algorithm SGLD --lr 0.001 --seed {}'.format(i) + common_settings)
    # os.system('python3 main_iter.py --algorithm SGLDDK --lr 0.001 --bwType fix --bwVal 0.1 --alpha 0.2 --seed {}'.format(i) + common_settings)
    # os.system('python3 main_iter.py --algorithm GFSD --lr 0.001 --bwType nei --seed {}'.format(i) + common_settings)
    # os.system('python3 main_iter.py --algorithm GFSDCA --lr 0.001 --bwType nei --alpha 1.5 --seed {}'.format(i) + common_settings)
    # os.system('python3 main_iter.py --algorithm GFSDDK --lr 0.001 --bwType nei --alpha 1.0 --seed {}'.format(i) + common_settings)
    os.system('python3 main_iter.py --algorithm BLOB --lr 0.003 --bwType nei --seed {}'.format(i) + common_settings)
    # os.system('python3 main_iter.py --algorithm BLOBCA --lr 0.003 --bwType nei --alpha 0.5 --seed {}'.format(i) + common_settings)
    # os.system('python3 main_iter.py --algorithm BLOBDK --lr 0.001 --bwType nei --alpha 1.0 --seed {}'.format(i) + common_settings)
    # os.system('python3 main_iter.py --algorithm KSDD --lr 0.1 --bwType fix --bwVal 1.0 --annealing 0.2 --seed {}'.format(i) + common_settings)
    # os.system('python3 main_iter.py --algorithm KSDDCA --lr 0.1 --bwType fix --bwVal 1.0 --alpha 1.0 --annealing 0.2 --seed {}'.format(i) + common_settings)
    # os.system('python3 main_iter.py --algorithm KSDDDK --lr 0.1 --bwType fix --bwVal 1.0 --alpha 0.2 --annealing 0.2 --seed {}'.format(i) + common_settings)