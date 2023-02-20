import os 

common_settings = ' ' + ' --save_particles ' + '--task single_gaussian --model_dim 10 --max_iter 10000 --eval_interval 500 --particle_num 32 --save_folder results_save/results_sg_iter/num32 --device cuda:0'
for i in range(0, 10):
    os.system('python main_iter.py --algorithm SVGD --lr 0.3 --bwType med --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm HMC --lr 0.01 --leap_iter 10 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm SGLD --lr 0.01 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm SGLDDK --lr 0.01 --bwType fix --bwVal 1 --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm GFSD --lr 0.015 --bwType nei --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm GFSDCA --lr 0.015 --bwType nei --alpha 0.3 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm GFSDDK --lr 0.015 --bwType nei --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm BLOB --lr 0.015 --bwType nei --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm BLOBCA --lr 0.015 --bwType nei --alpha 0.3 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm BLOBDK --lr 0.015 --bwType nei --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm KSDD --lr 0.1 --bwType fix --bwVal 8.0 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm KSDDCA --lr 0.1 --bwType fix --bwVal 8.0 --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm KSDDDK --lr 0.1 --bwType fix --bwVal 8.0 --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)

common_settings = ' ' + ' --save_particles ' + '--task single_gaussian --model_dim 10 --max_iter 10000 --eval_interval 500 --particle_num 64 --save_folder results_save/results_sg_iter/num64 --device cuda:0'
for i in range(0, 10):
    os.system('python main_iter.py --algorithm SVGD --lr 0.3 --bwType med --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm HMC --lr 0.01 --leap_iter 10 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm SGLD --lr 0.01 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm SGLDDK --lr 0.01 --bwType fix --bwVal 1 --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm GFSD --lr 0.015 --bwType nei --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm GFSDCA --lr 0.015 --bwType nei --alpha 0.3 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm GFSDDK --lr 0.015 --bwType nei --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm BLOB --lr 0.015 --bwType nei --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm BLOBCA --lr 0.015 --bwType nei --alpha 0.3 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm BLOBDK --lr 0.015 --bwType nei --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm KSDD --lr 0.1 --bwType fix --bwVal 6.0 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm KSDDCA --lr 0.1 --bwType fix --bwVal 6.0 --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm KSDDDK --lr 0.1 --bwType fix --bwVal 6.0 --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)

common_settings = ' ' + ' --save_particles ' + '--task single_gaussian --model_dim 10 --max_iter 10000 --eval_interval 500 --particle_num 128 --save_folder results_save/results_sg_iter/num128 --device cuda:0'
for i in range(0, 10):
    os.system('python main_iter.py --algorithm SVGD --lr 0.3 --bwType med --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm HMC --lr 0.01 --leap_iter 10 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm SGLD --lr 0.01 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm SGLDDK --lr 0.01 --bwType fix --bwVal 1 --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm GFSD --lr 0.015 --bwType nei --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm GFSDCA --lr 0.015 --bwType nei --alpha 0.3 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm GFSDDK --lr 0.015 --bwType nei --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm BLOB --lr 0.015 --bwType nei --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm BLOBCA --lr 0.015 --bwType nei --alpha 0.3 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm BLOBDK --lr 0.015 --bwType nei --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm KSDD --lr 0.2 --bwType fix --bwVal 4.0 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm KSDDCA --lr 0.2 --bwType fix --bwVal 4.0 --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm KSDDDK --lr 0.2 --bwType fix --bwVal 4.0 --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)

common_settings = ' ' + ' --save_particles ' + '--task single_gaussian --model_dim 10 --max_iter 10000 --eval_interval 500 --particle_num 256 --save_folder results_save/results_sg_iter/num256 --device cuda:0'
for i in range(0, 10):
    os.system('python main_iter.py --algorithm SVGD --lr 0.3 --bwType med --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm HMC --lr 0.01 --leap_iter 10 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm SGLD --lr 0.01 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm SGLDDK --lr 0.01 --bwType fix --bwVal 1 --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm GFSD --lr 0.015 --bwType nei --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm GFSDCA --lr 0.015 --bwType nei --alpha 0.3 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm GFSDDK --lr 0.015 --bwType nei --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm BLOB --lr 0.015 --bwType nei --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm BLOBCA --lr 0.015 --bwType nei --alpha 0.3 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm BLOBDK --lr 0.015 --bwType nei --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm KSDD --lr 0.4 --bwType fix --bwVal 4.0 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm KSDDCA --lr 0.4 --bwType fix --bwVal 4.0 --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm KSDDDK --lr 0.4 --bwType fix --bwVal 4.0 --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)

common_settings = ' ' + ' --save_particles ' + '--task single_gaussian --model_dim 10 --max_iter 10000 --eval_interval 500 --particle_num 512 --save_folder results_save/results_sg_iter/num512 --device cuda:0'
for i in range(0, 10):
    os.system('python main_iter.py --algorithm SVGD --lr 0.3 --bwType med --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm HMC --lr 0.01 --leap_iter 10 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm SGLD --lr 0.01 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm SGLDDK --lr 0.01 --bwType fix --bwVal 1 --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm GFSD --lr 0.015 --bwType nei --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm GFSDCA --lr 0.015 --bwType nei --alpha 0.3 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm GFSDDK --lr 0.015 --bwType nei --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm BLOB --lr 0.015 --bwType nei --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm BLOBCA --lr 0.015 --bwType nei --alpha 0.3 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm BLOBDK --lr 0.015 --bwType nei --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm KSDD --lr 0.4 --bwType fix --bwVal 4.0 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm KSDDCA --lr 0.4 --bwType fix --bwVal 4.0 --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm KSDDDK --lr 0.4 --bwType fix --bwVal 4.0 --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)