import os

# common_settings = ' ' + '--task gaussian_process --dataset lidar --max_iter 10000 --particle_num 128 --eval_interval 500 --device cuda:0 --save_folder results_gp --reference_path hmc_reference/gaussian_process/particles.pkl'
# for i in range(0, 10):
#     os.system('python main_iter.py --algorithm SVGD --lr 8.0e-2 --bwType med --seed {}'.format(i) + common_settings)

#     os.system('python main_iter.py --algorithm HMC --lr 1.0e-2 --leap_iter 5 --seed {}'.format(i) + common_settings)
#     os.system('python main_iter.py --algorithm SGLD --lr 2.0e-3 --seed {}'.format(i) + common_settings)
#     os.system('python main_iter.py --algorithm SGLDDK --lr 2.0e-3 --bwType fix --bwVal 1.0 --alpha 0.3 --al_warmup --seed {}'.format(i) + common_settings)

#     os.system('python main_iter.py --algorithm GFSD --lr 1.0e-2 --bwType nei --seed {}'.format(i) + common_settings)
#     os.system('python main_iter.py --algorithm GFSDCA --lr 1.0e-2 --bwType nei --alpha 0.3 --al_warmup --seed {}'.format(i) + common_settings)
#     os.system('python main_iter.py --algorithm GFSDDK --lr 1.0e-2 --bwType nei --alpha 0.3 --al_warmup --seed {}'.format(i) + common_settings)

#     os.system('python main_iter.py --algorithm BLOB --lr 1.0e-2 --bwType nei --seed {}'.format(i) + common_settings)
#     os.system('python main_iter.py --algorithm BLOBCA --lr 1.0e-2 --bwType nei --alpha 0.3 --al_warmup --seed {}'.format(i) + common_settings)
#     os.system('python main_iter.py --algorithm BLOBDK --lr 1.0e-2 --bwType nei --alpha 0.3 --al_warmup --seed {}'.format(i) + common_settings)

#     os.system('python main_iter.py --algorithm KSDD --lr 2.0e-2 --bwType fix --bwVal 0.1 --seed {}'.format(i) + common_settings)
#     os.system('python main_iter.py --algorithm KSDDCA --lr 2.0e-2 --bwType fix --bwVal 0.1 --alpha 0.3 --al_warmup --seed {}'.format(i) + common_settings)
#     os.system('python main_iter.py --algorithm KSDDDK --lr 2.0e-2 --bwType fix --bwVal 0.1 --alpha 0.3 --al_warmup --seed {}'.format(i) + common_settings)

common_settings = ' ' + '--task gaussian_process --dataset lidar --max_time 300 --particle_num 128 --eval_interval 15 --device cuda:0 --save_folder results_gp_time --reference_path hmc_reference/gaussian_process/particles.pkl'
for i in range(0, 10):
    os.system('python main_time.py --algorithm SVGD --lr 8.0e-2 --bwType med --seed {}'.format(i) + common_settings)

    os.system('python main_time.py --algorithm HMC --lr 1.0e-2 --leap_iter 5 --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm SGLD --lr 2.0e-3 --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm SGLDDK --lr 2.0e-3 --bwType fix --bwVal 1.0 --alpha 0.3 --al_warmup --seed {}'.format(i) + common_settings)

    os.system('python main_time.py --algorithm GFSD --lr 1.0e-2 --bwType nei --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm GFSDCA --lr 1.0e-2 --bwType nei --alpha 0.3 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm GFSDDK --lr 1.0e-2 --bwType nei --alpha 0.3 --al_warmup --seed {}'.format(i) + common_settings)

    os.system('python main_time.py --algorithm BLOB --lr 1.0e-2 --bwType nei --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm BLOBCA --lr 1.0e-2 --bwType nei --alpha 0.3 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm BLOBDK --lr 1.0e-2 --bwType nei --alpha 0.3 --al_warmup --seed {}'.format(i) + common_settings)

    os.system('python main_time.py --algorithm KSDD --lr 2.0e-2 --bwType fix --bwVal 0.1 --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm KSDDCA --lr 2.0e-2 --bwType fix --bwVal 0.1 --alpha 0.3 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_time.py --algorithm KSDDDK --lr 2.0e-2 --bwType fix --bwVal 0.1 --alpha 0.3 --al_warmup --seed {}'.format(i) + common_settings)


# common_settings = ' ' + '--task gaussian_process --save_particles --dataset lidar --max_iter 10000 --particle_num 128 --eval_interval 500 --device cuda:0 --save_folder results_gp --reference_path hmc_reference/gaussian_process/particles.pkl'
# for i in range(0, 1):
#     os.system('python main_iter.py --algorithm SVGD --lr 8.0e-2 --bwType med --seed {}'.format(i) + common_settings)

#     os.system('python main_iter.py --algorithm HMC --lr 1.0e-2 --leap_iter 5 --seed {}'.format(i) + common_settings)
#     os.system('python main_iter.py --algorithm SGLD --lr 2.0e-3 --seed {}'.format(i) + common_settings)
#     os.system('python main_iter.py --algorithm SGLDDK --lr 2.0e-3 --bwType fix --bwVal 1.0 --alpha 0.3 --al_warmup --seed {}'.format(i) + common_settings)

#     os.system('python main_iter.py --algorithm GFSD --lr 1.0e-2 --bwType nei --seed {}'.format(i) + common_settings)
#     os.system('python main_iter.py --algorithm GFSDCA --lr 1.0e-2 --bwType nei --alpha 0.3 --al_warmup --seed {}'.format(i) + common_settings)
#     os.system('python main_iter.py --algorithm GFSDDK --lr 1.0e-2 --bwType nei --alpha 0.3 --al_warmup --seed {}'.format(i) + common_settings)

#     os.system('python main_iter.py --algorithm BLOB --lr 1.0e-2 --bwType nei --seed {}'.format(i) + common_settings)
#     os.system('python main_iter.py --algorithm BLOBCA --lr 1.0e-2 --bwType nei --alpha 0.3 --al_warmup --seed {}'.format(i) + common_settings)
#     os.system('python main_iter.py --algorithm BLOBDK --lr 1.0e-2 --bwType nei --alpha 0.3 --al_warmup --seed {}'.format(i) + common_settings)

#     os.system('python main_iter.py --algorithm KSDD --lr 2.0e-2 --bwType fix --bwVal 0.1 --seed {}'.format(i) + common_settings)
#     os.system('python main_iter.py --algorithm KSDDCA --lr 2.0e-2 --bwType fix --bwVal 0.1 --alpha 0.3 --al_warmup --seed {}'.format(i) + common_settings)
#     os.system('python main_iter.py --algorithm KSDDDK --lr 2.0e-2 --bwType fix --bwVal 0.1 --alpha 0.3 --al_warmup --seed {}'.format(i) + common_settings)