import os

# common_settings = ' ' + '--task bnn_regression --dataset electrical --max_iter 10000 --batch_size 128 --eval_interval 500 --particle_num 128 --save_folder results_save/results_electrical --device cuda:3'
# for i in range(0, 10):
#     os.system('python3 main_iter.py --algorithm SVGD --lr 1.0e-4 --bwType med --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm SGLD --lr 0.85e-6 --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm SGLDDK --lr 0.85e-6 --bwType fix --bwVal 1.0 --alpha 1.0 --al_warmup --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm GFSD --lr 0.85e-6 --bwType nei --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm GFSDCA --lr 0.85e-6 --bwType nei --alpha 1.0 --al_warmup --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm GFSDDK --lr 0.85e-6 --bwType nei --alpha 1.0 --al_warmup --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm BLOB --lr 0.85e-6 --bwType nei --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm BLOBCA --lr 0.85e-6 --bwType nei --alpha 1.0 --al_warmup --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm BLOBDK --lr 0.85e-6 --bwType nei --alpha 1.0 --al_warmup --seed {}'.format(i) + common_settings)

# common_settings = ' ' + '--task bnn_regression --dataset concrete --max_iter 30000 --batch_size 128 --eval_interval 1500 --particle_num 128 --save_folder results_save/results_concrete --device cuda:3'
# for i in range(0, 10):
#     os.system('python3 main_iter.py --algorithm SVGD --lr 5.0e-4 --bwType med --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm SGLD --lr 4.0e-6 --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm SGLDDK --lr 4.0e-6 --bwType fix --bwVal 1.0 --alpha 0.2 --al_warmup --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm GFSD --lr 4.0e-6 --bwType nei --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm GFSDCA --lr 4.0e-6 --bwType nei --alpha 0.5 --al_warmup --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm GFSDDK --lr 4.0e-6 --bwType nei --alpha 0.5 --al_warmup --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm BLOB --lr 4.0e-6 --bwType nei --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm BLOBCA --lr 4.0e-6 --bwType nei --alpha 0.5 --al_warmup --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm BLOBDK --lr 4.0e-6 --bwType nei --alpha 0.5 --al_warmup --seed {}'.format(i) + common_settings)

# common_settings = ' ' + '--task bnn_regression --dataset kin8nm --max_iter 30000 --batch_size 128 --eval_interval 1500 --particle_num 128 --save_folder results_save/results_kin8nm --device cuda:3'
# for i in range(0, 10):
#     os.system('python3 main_iter.py --algorithm SVGD --lr 1.2e-4 --bwType med --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm SGLD --lr 1.0e-6 --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm SGLDDK --lr 1.0e-6 --bwType fix --bwVal 1.0 --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm GFSD --lr 1.0e-6 --bwType nei --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm GFSDCA --lr 1.0e-6 --bwType nei --alpha 0.3 --al_warmup --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm GFSDDK --lr 1.0e-6 --bwType nei --alpha 0.3 --al_warmup --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm BLOB --lr 1.0e-6 --bwType nei --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm BLOBCA --lr 1.0e-6 --bwType nei --alpha 0.3 --al_warmup --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm BLOBDK --lr 1.0e-6 --bwType nei --alpha 0.3 --al_warmup --seed {}'.format(i) + common_settings)

# common_settings = ' ' + '--task bnn_regression --dataset space --max_iter 30000 --batch_size 128 --eval_interval 1500 --particle_num 128 --save_folder results_save/results_space --device cuda:3'
# for i in range(0, 1):
#     os.system('python3 main_iter.py --algorithm SVGD --lr 3.0e-4 --bwType med --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm SGLD --lr 3.0e-6 --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm SGLDDK --lr 3.0e-6 --bwType fix --bwVal 1.0 --alpha 0.2 --al_warmup --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm GFSD --lr 3.0e-6 --bwType nei --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm GFSDCA --lr 3.0e-6 --bwType nei --alpha 1.0 --al_warmup --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm GFSDDK --lr 3.0e-6 --bwType nei --alpha 1.0 --al_warmup --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm BLOB --lr 3.0e-6 --bwType nei --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm BLOBCA --lr 3.0e-6 --bwType nei --alpha 1.0 --al_warmup --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm BLOBDK --lr 3.0e-6 --bwType nei --alpha 1.0 --al_warmup --seed {}'.format(i) + common_settings)
    
    
# common_settings = ' ' + '--task bnn_regression --dataset WineRed --max_iter 30000 --batch_size 128 --eval_interval 1500 --particle_num 128 --save_folder results_save/results_winered --device cuda:3'
# for i in range(0, 10):
#     os.system('python3 main_iter.py --algorithm SVGD --lr 4.0e-4 --bwType med --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm SGLD --lr 4.0e-6 --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm SGLDDK --lr 4.0e-6 --bwType fix --bwVal 1.0 --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm GFSD --lr 3.4e-6 --bwType nei --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm GFSDCA --lr 3.4e-6 --bwType nei --alpha 0.5 --al_warmup --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm GFSDDK --lr 3.4e-6 --bwType nei --alpha 0.5 --al_warmup --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm BLOB --lr 3.4e-6 --bwType nei --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm BLOBCA --lr 3.4e-6 --bwType nei --alpha 0.5 --al_warmup --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm BLOBDK --lr 3.4e-6 --bwType nei --alpha 0.5 --al_warmup --seed {}'.format(i) + common_settings)

# common_settings = ' ' + '--task bnn_regression --dataset cpusmall --max_iter 20000 --batch_size 128 --eval_interval 500 --particle_num 128 --save_folder results --device cuda:3'
# for i in range(0, 1):
#     os.system('python3 main_iter.py --algorithm SVGD --lr 5e-5 --bwType med --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm SGLD --lr 5e-7 --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm SGLDDK --lr 5e-7 --bwType fix --bwVal 1.0 --alpha 0.1 --al_warmup --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm GFSD --lr 5e-7 --bwType nei --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm GFSDCA --lr 5e-7 --bwType nei --alpha 1.0 --al_warmup --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm GFSDDK --lr 5e-7 --bwType nei --alpha 1.0 --al_warmup --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm BLOB --lr 5e-7 --bwType nei --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm BLOBCA --lr 5e-7 --bwType nei --alpha 1.0 --al_warmup --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm BLOBDK --lr 5e-7 --bwType nei --alpha 1.0 --al_warmup --seed {}'.format(i) + common_settings)
    
    
    