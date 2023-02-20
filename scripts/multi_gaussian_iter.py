import os 

common_settings = ' ' + '--task multi_gaussian --model_dim 10 --max_iter 10000 --eval_interval 500 --particle_num 32 --save_folder results_save/results_mg_iter/num32 --device cuda:1'
for i in range(0, 10):
    # if i == 0: common_settings += ' --save_particles '
    os.system('python main_iter.py --algorithm SVGD --lr 0.5 --bwType med --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm HMC --lr 0.03 --leap_iter 10 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm SGLD --lr 0.003 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm SGLDDK --lr 0.003 --bwType fix --bwVal 1.0 --alpha 0.02 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm GFSD --lr 0.01 --bwType nei --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm GFSDCA --lr 0.01 --bwType nei --alpha 0.2 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm GFSDDK --lr 0.01 --bwType nei --alpha 0.2 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm BLOB --lr 0.01 --bwType nei --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm BLOBCA --lr 0.01 --bwType nei --alpha 0.2 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm BLOBDK --lr 0.01 --bwType nei --alpha 0.2 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm KSDD --lr 0.05 --bwType fix --bwVal 16.0 --anneal 0.3 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm KSDDCA --lr 0.05 --bwType fix --bwVal 16.0 --alpha 0.1 --al_warmup --anneal 0.3 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm KSDDDK --lr 0.05 --bwType fix --bwVal 16.0 --alpha 0.1 --al_warmup --anneal 0.3 --seed {}'.format(i) + common_settings)

common_settings = ' ' + '--task multi_gaussian --model_dim 10 --max_iter 10000 --eval_interval 500 --particle_num 64 --save_folder results_save/results_mg_iter/num64 --device cuda:1'
for i in range(0, 10):
    # if i == 0: common_settings += ' --save_particles '
    os.system('python main_iter.py --algorithm SVGD --lr 0.5 --bwType med --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm HMC --lr 0.03 --leap_iter 10 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm SGLD --lr 0.003 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm SGLDDK --lr 0.003 --bwType fix --bwVal 1.0 --alpha 0.02 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm GFSD --lr 0.01 --bwType nei --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm GFSDCA --lr 0.01 --bwType nei --alpha 0.2 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm GFSDDK --lr 0.01 --bwType nei --alpha 0.2 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm BLOB --lr 0.01 --bwType nei --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm BLOBCA --lr 0.01 --bwType nei --alpha 0.2 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm BLOBDK --lr 0.01 --bwType nei --alpha 0.2 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm KSDD --lr 0.05 --bwType fix --bwVal 12.0 --anneal 0.3 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm KSDDCA --lr 0.05 --bwType fix --bwVal 12.0 --alpha 0.1 --al_warmup --anneal 0.3 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm KSDDDK --lr 0.05 --bwType fix --bwVal 12.0 --alpha 0.1 --al_warmup --anneal 0.3 --seed {}'.format(i) + common_settings)

common_settings = ' ' + '--task multi_gaussian --model_dim 10 --max_iter 10000 --eval_interval 500 --particle_num 128 --save_folder results_save/results_mg_iter/num128 --device cuda:1'
for i in range(0, 10):
    # if i == 0: common_settings += ' --save_particles '
    os.system('python main_iter.py --algorithm SVGD --lr 0.5 --bwType med --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm HMC --lr 0.03 --leap_iter 10 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm SGLD --lr 0.003 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm SGLDDK --lr 0.003 --bwType fix --bwVal 1.0 --alpha 0.02 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm GFSD --lr 0.01 --bwType nei --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm GFSDCA --lr 0.01 --bwType nei --alpha 0.2 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm GFSDDK --lr 0.01 --bwType nei --alpha 0.2 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm BLOB --lr 0.01 --bwType nei --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm BLOBCA --lr 0.01 --bwType nei --alpha 0.2 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm BLOBDK --lr 0.01 --bwType nei --alpha 0.2 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm KSDD --lr 0.05 --bwType fix --bwVal 8.0 --anneal 0.3 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm KSDDCA --lr 0.05 --bwType fix --bwVal 8.0 --alpha 0.1 --al_warmup --anneal 0.3 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm KSDDDK --lr 0.05 --bwType fix --bwVal 8.0 --alpha 0.1 --al_warmup --anneal 0.3 --seed {}'.format(i) + common_settings)

common_settings = ' ' + '--task multi_gaussian --model_dim 10 --max_iter 10000 --eval_interval 500 --particle_num 256 --save_folder results_save/results_mg_iter/num256 --device cuda:1'
for i in range(0, 10):
    # if i == 0: common_settings += ' --save_particles '
    os.system('python main_iter.py --algorithm SVGD --lr 0.5 --bwType med --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm HMC --lr 0.03 --leap_iter 10 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm SGLD --lr 0.003 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm SGLDDK --lr 0.003 --bwType fix --bwVal 0.2 --alpha 0.02 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm GFSD --lr 0.01 --bwType nei --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm GFSDCA --lr 0.01 --bwType nei --alpha 0.2 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm GFSDDK --lr 0.01 --bwType nei --alpha 0.2 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm BLOB --lr 0.01 --bwType nei --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm BLOBCA --lr 0.01 --bwType nei --alpha 0.2 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm BLOBDK --lr 0.01 --bwType nei --alpha 0.2 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm KSDD --lr 0.1 --bwType fix --bwVal 4.0 --anneal 0.3 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm KSDDCA --lr 0.1 --bwType fix --bwVal 4.0 --alpha 0.1 --al_warmup --anneal 0.3 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm KSDDDK --lr 0.1 --bwType fix --bwVal 4.0 --alpha 0.1 --al_warmup --anneal 0.3 --seed {}'.format(i) + common_settings)

common_settings = ' ' + '--task multi_gaussian --model_dim 10 --max_iter 10000 --eval_interval 500 --particle_num 512 --save_folder results_save/results_mg_iter/num512 --device cuda:1'
for i in range(0, 10):
    # if i == 0: common_settings += ' --save_particles '
    os.system('python main_iter.py --algorithm SVGD --lr 0.5 --bwType med --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm HMC --lr 0.03 --leap_iter 10 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm SGLD --lr 0.003 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm SGLDDK --lr 0.003 --bwType fix --bwVal 0.2 --alpha 0.02 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm GFSD --lr 0.01 --bwType nei --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm GFSDCA --lr 0.01 --bwType nei --alpha 0.2 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm GFSDDK --lr 0.01 --bwType nei --alpha 0.2 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm BLOB --lr 0.01 --bwType nei --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm BLOBCA --lr 0.01 --bwType nei --alpha 0.2 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm BLOBDK --lr 0.01 --bwType nei --alpha 0.2 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm KSDD --lr 0.15 --bwType fix --bwVal 2.0 --anneal 0.3 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm KSDDCA --lr 0.15 --bwType fix --bwVal 2.0 --alpha 0.1 --al_warmup --anneal 0.3 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm KSDDDK --lr 0.15 --bwType fix --bwVal 2.0 --alpha 0.1 --al_warmup --anneal 0.3 --seed {}'.format(i) + common_settings)









































# extra tunning is required for KSDD-type methods
# common_settings = ' ' + '--task multi_gaussian --model_dim 10 --max_iter 10000 --eval_interval 300 --particle_num 64 --save_folder results'
# lr_list = [0.01, 0.02, 0.05, 0.1, 0.2]
# bwVal_list = [2.0, 1.0, 0.5, 0.2, 0.1]
# anneal_list = [0.1, 0.2, 0.5]
# alpha_list = [0.1, 0.2, 0.5]
# for lr in lr_list:
#     for bwVal in bwVal_list:
#         for anneal in anneal_list:
#             os.system('python3 main_iter.py --algorithm KSDD --lr {} --bwType fix --bwVal {} --anneal {} --seed {}'.format(lr, bwVal, anneal, 0) + common_settings)
            # for alpha in anneal_list:
            #     os.system('python3 main_iter.py --algorithm KSDDCA --lr {} --bwType fix --bwVal {} --alpha {} --anneal {} --seed {}'.format(lr, bwVal, alpha, anneal, 0) + common_settings)
            #     os.system('python3 main_iter.py --algorithm KSDDDK --lr {} --bwType fix --bwVal {} --alpha {} --anneal {} --seed {}'.format(lr, bwVal, alpha, anneal, 0) + common_settings)

# common_settings = ' ' + '--task multi_gaussian --model_dim 10 --max_iter 10000 --eval_interval 500 --particle_num 64 --save_folder results --device cuda:3'
# for i in range(0, 1):
#     os.system('python3 main_iter.py --algorithm SVGD --lr 0.1 --bwType med --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm SVGD --lr 0.1 --bwType med --anneal 0.1 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main_iter.py --algorithm SGLD --lr 0.001 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main_iter.py --algorithm SGLDDK --lr 0.001 --bwType fix --bwVal 5 --alpha 0.2 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main_iter.py --algorithm GFSD --lr 0.001 --bwType nei --seed {}'.format(i) + common_settings)
#     # os.system('python3 main_iter.py --algorithm GFSDCA --lr 0.001 --bwType nei --alpha 0.5 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main_iter.py --algorithm GFSDDK --lr 0.001 --bwType nei --alpha 0.5 --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm BLOB --lr 0.001 --bwType nei --seed {}'.format(i) + common_settings)
#     # os.system('python3 main_iter.py --algorithm BLOBCA --lr 0.001 --bwType nei --alpha 0.5 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main_iter.py --algorithm BLOBDK --lr 0.001 --bwType nei --alpha 0.5 --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm BLOB --lr 0.001 --bwType nei --anneal 0.1 --seed {}'.format(i) + common_settings)
    # os.system('python3 main_iter.py --algorithm BLOBCA --lr 0.001 --bwType nei --alpha 0.5 --anneal 0.1 --seed {}'.format(i) + common_settings)
    # os.system('python3 main_iter.py --algorithm BLOBDK --lr 0.001 --bwType nei --alpha 0.5 --anneal 0.1 --seed {}'.format(i) + common_settings)
    # os.system('python3 main_iter.py --algorithm KSDD --lr 0.1 --bwType fix --bwVal 0.5 --anneal 0.1 --seed {}'.format(i) + common_settings)
    # os.system('python3 main_iter.py --algorithm KSDDCA --lr 0.1 --bwType fix --bwVal 0.5 --alpha 0.5 --anneal 0.1 --seed {}'.format(i) + common_settings)
    # os.system('python3 main_iter.py --algorithm KSDDDK --lr 0.1 --bwType fix --bwVal 0.5 --alpha 0.5 --anneal 0.1 --seed {}'.format(i) + common_settings)

# common_settings = ' --task multi_gaussian --max_iter 6000 --eval_interval 300 --particle_num 512 --save_folder results'
# for i in range(0, 1):
#     os.system('python3 main_iter.py --algorithm SVGD --lr 0.1 --bwType med --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm SGLD --lr 0.001 --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm SGLDDK --lr 0.001 --bwType fix --bwVal 0.1 --alpha 0.2 --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm GFSD --lr 0.001 --bwType nei --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm GFSDCA --lr 0.001 --bwType nei --alpha 1.5 --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm GFSDDK --lr 0.001 --bwType nei --alpha 1.0 --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm BLOB --lr 0.001 --bwType nei --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm BLOBCA --lr 0.001 --bwType nei --alpha 1.5 --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm BLOBDK --lr 0.001 --bwType nei --alpha 1.0 --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm KSDD --lr 0.1 --bwType fix --bwVal 1.0 --annealing 0.2 --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm KSDDCA --lr 0.1 --bwType fix --bwVal 1.0 --alpha 1.0 --annealing 0.2 --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm KSDDDK --lr 0.1 --bwType fix --bwVal 1.0 --alpha 0.2 --annealing 0.2 --seed {}'.format(i) + common_settings)

# common_settings = ' --task multi_gaussian --max_iter 6000 --eval_interval 300 --particle_num 256 --save_folder results'
# for i in range(0, 10):
#     os.system('python3 main_iter.py --algorithm SVGD --lr 0.1 --bwType med --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm SGLD --lr 0.001 --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm SGLDDK --lr 0.001 --bwType fix --bwVal 0.1 --alpha 0.2 --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm GFSD --lr 0.001 --bwType nei --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm GFSDCA --lr 0.001 --bwType nei --alpha 1.5 --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm GFSDDK --lr 0.001 --bwType nei --alpha 1.0 --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm BLOB --lr 0.001 --bwType nei --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm BLOBCA --lr 0.001 --bwType nei --alpha 1.5 --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm BLOBDK --lr 0.001 --bwType nei --alpha 1.0 --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm KSDD --lr 0.1 --bwType fix --bwVal 1.0 --annealing 0.2 --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm KSDDCA --lr 0.1 --bwType fix --bwVal 1.0 --alpha 1.0 --annealing 0.2 --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm KSDDDK --lr 0.1 --bwType fix --bwVal 1.0 --alpha 0.2 --annealing 0.2 --seed {}'.format(i) + common_settings)


# common_settings = ' --task multi_gaussian --max_iter 6000 --eval_interval 300 --particle_num 32 --save_folder results'
# for i in range(0, 1):
#     os.system('python3 main_iter.py --algorithm SVGD --lr 0.1 --bwType med --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm SGLD --lr 0.001 --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm SGLDDK --lr 0.001 --bwType fix --bwVal 5 --alpha 0.2 --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm GFSD --lr 0.001 --bwType nei --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm GFSDCA --lr 0.001 --bwType nei --alpha 0.5 --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm GFSDDK --lr 0.001 --bwType nei --alpha 0.5 --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm BLOB --lr 0.001 --bwType nei --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm BLOBCA --lr 0.001 --bwType nei --alpha 0.5 --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm BLOBDK --lr 0.001 --bwType nei --alpha 0.5 --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm KSDD --lr 0.1 --bwType fix --bwVal 1.0 --annealing 0.2 --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm KSDDCA --lr 0.1 --bwType fix --bwVal 1.0 --alpha 1.0 --annealing 0.2 --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm KSDDDK --lr 0.1 --bwType fix --bwVal 1.0 --alpha 0.2 --annealing 0.2 --seed {}'.format(i) + common_settings)

    



