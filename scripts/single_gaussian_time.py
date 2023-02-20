import os 

common_settings = ' ' + '--task single_gaussian --model_dim 10 --max_time 45 --eval_interval 3 --particle_num 32 --save_folder results_save/results_sg_time/num32 --device cuda:0'
for i in range(0, 10):
    os.system('python main_time.py --algorithm SVGD --lr 0.3 --bwType med --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm HMC --lr 0.01 --leap_iter 10 --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm SGLD --lr 0.01 --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm SGLDDK --lr 0.01 --bwType fix --bwVal 1 --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm GFSD --lr 0.015 --bwType nei --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm GFSDCA --lr 0.015 --bwType nei --alpha 0.3 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm GFSDDK --lr 0.015 --bwType nei --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm BLOB --lr 0.015 --bwType nei --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm BLOBCA --lr 0.015 --bwType nei --alpha 0.3 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm BLOBDK --lr 0.015 --bwType nei --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm KSDD --lr 0.1 --bwType fix --bwVal 8.0 --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm KSDDCA --lr 0.1 --bwType fix --bwVal 8.0 --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm KSDDDK --lr 0.1 --bwType fix --bwVal 8.0 --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)

common_settings = ' ' + '--task single_gaussian --model_dim 10 --max_time 45 --eval_interval 3 --particle_num 64 --save_folder results_save/results_sg_time/num64 --device cuda:0'
for i in range(0, 10):
    os.system('python main_time.py --algorithm SVGD --lr 0.3 --bwType med --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm HMC --lr 0.01 --leap_iter 10 --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm SGLD --lr 0.01 --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm SGLDDK --lr 0.01 --bwType fix --bwVal 1 --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm GFSD --lr 0.015 --bwType nei --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm GFSDCA --lr 0.015 --bwType nei --alpha 0.3 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm GFSDDK --lr 0.015 --bwType nei --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm BLOB --lr 0.015 --bwType nei --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm BLOBCA --lr 0.015 --bwType nei --alpha 0.3 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm BLOBDK --lr 0.015 --bwType nei --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm KSDD --lr 0.1 --bwType fix --bwVal 6.0 --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm KSDDCA --lr 0.1 --bwType fix --bwVal 6.0 --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm KSDDDK --lr 0.1 --bwType fix --bwVal 6.0 --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)

common_settings = ' ' + '--task single_gaussian --model_dim 10 --max_time 45 --eval_interval 3 --particle_num 128 --save_folder results_save/results_sg_time/num128 --device cuda:0'
for i in range(0, 10):
    os.system('python main_time.py --algorithm SVGD --lr 0.3 --bwType med --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm HMC --lr 0.01 --leap_iter 10 --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm SGLD --lr 0.01 --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm SGLDDK --lr 0.01 --bwType fix --bwVal 1 --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm GFSD --lr 0.015 --bwType nei --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm GFSDCA --lr 0.015 --bwType nei --alpha 0.3 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm GFSDDK --lr 0.015 --bwType nei --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm BLOB --lr 0.015 --bwType nei --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm BLOBCA --lr 0.015 --bwType nei --alpha 0.3 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm BLOBDK --lr 0.015 --bwType nei --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm KSDD --lr 0.2 --bwType fix --bwVal 4.0 --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm KSDDCA --lr 0.2 --bwType fix --bwVal 4.0 --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm KSDDDK --lr 0.2 --bwType fix --bwVal 4.0 --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)

common_settings = ' ' + '--task single_gaussian --model_dim 10 --max_time 45 --eval_interval 3 --particle_num 256 --save_folder results_save/results_sg_time/num256 --device cuda:0'
for i in range(0, 10):
    os.system('python main_time.py --algorithm SVGD --lr 0.3 --bwType med --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm HMC --lr 0.01 --leap_iter 10 --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm SGLD --lr 0.01 --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm SGLDDK --lr 0.01 --bwType fix --bwVal 1 --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm GFSD --lr 0.015 --bwType nei --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm GFSDCA --lr 0.015 --bwType nei --alpha 0.3 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm GFSDDK --lr 0.015 --bwType nei --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm BLOB --lr 0.015 --bwType nei --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm BLOBCA --lr 0.015 --bwType nei --alpha 0.3 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm BLOBDK --lr 0.015 --bwType nei --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm KSDD --lr 0.4 --bwType fix --bwVal 4.0 --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm KSDDCA --lr 0.4 --bwType fix --bwVal 4.0 --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm KSDDDK --lr 0.4 --bwType fix --bwVal 4.0 --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)

common_settings = ' ' + '--task single_gaussian --model_dim 10 --max_time 45 --eval_interval 3 --particle_num 512 --save_folder results_save/results_sg_time/num512 --device cuda:0'
for i in range(0, 10):
    os.system('python main_time.py --algorithm SVGD --lr 0.3 --bwType med --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm HMC --lr 0.01 --leap_iter 10 --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm SGLD --lr 0.01 --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm SGLDDK --lr 0.01 --bwType fix --bwVal 1 --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm GFSD --lr 0.015 --bwType nei --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm GFSDCA --lr 0.015 --bwType nei --alpha 0.3 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm GFSDDK --lr 0.015 --bwType nei --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm BLOB --lr 0.015 --bwType nei --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm BLOBCA --lr 0.015 --bwType nei --alpha 0.3 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm BLOBDK --lr 0.015 --bwType nei --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm KSDD --lr 0.4 --bwType fix --bwVal 4.0 --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm KSDDCA --lr 0.4 --bwType fix --bwVal 4.0 --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm KSDDDK --lr 0.4 --bwType fix --bwVal 4.0 --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)