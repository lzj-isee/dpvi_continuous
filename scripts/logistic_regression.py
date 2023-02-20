import os 

common_settings = ' ' + '--task logistic_regression --dataset phishing --split_size 0.95 --max_iter 20000 --batch_size 1 --eval_interval 500 --particle_num 128 --save_folder results --device cuda:3'
for i in range(0, 1):
    # os.system('python3 main_iter.py --algorithm SVGD --lr 3e-3 --bwType med --seed {}'.format(i) + common_settings)
    # os.system('python3 main_iter.py --algorithm SGLD --lr 1e-5 --seed {}'.format(i) + common_settings)
    # os.system('python3 main_iter.py --algorithm SGLDDK --lr 1e-5 --bwType fix --bwVal 1.0 --alpha 1.0 --al_warmup --seed {}'.format(i) + common_settings)
    os.system('python3 main_iter.py --algorithm GFSD --lr 1e-4 --bwType nei --seed {}'.format(i) + common_settings)
    os.system('python3 main_iter.py --algorithm GFSDCA --lr 1e-4 --bwType nei --alpha 1.0 --al_warmup --seed {}'.format(i) + common_settings)
    # os.system('python3 main_iter.py --algorithm GFSDDK --lr 0.85e-6 --bwType nei --alpha 1.0 --al_warmup --seed {}'.format(i) + common_settings)
    # os.system('python3 main_iter.py --algorithm BLOB --lr 0.85e-6 --bwType nei --seed {}'.format(i) + common_settings)
    # os.system('python3 main_iter.py --algorithm BLOBCA --lr 0.85e-6 --bwType nei --alpha 1.0 --al_warmup --seed {}'.format(i) + common_settings)
    # os.system('python3 main_iter.py --algorithm BLOBDK --lr 0.85e-6 --bwType nei --alpha 1.0 --al_warmup --seed {}'.format(i) + common_settings)


# common_settings = ' ' + '--task logistic_regression --dataset pima --split_size 0.95 --max_iter 20000 --batch_size 1 --eval_interval 500 --particle_num 128 --save_folder results --device cuda:3'
# for i in range(0, 1):
#     # os.system('python3 main_iter.py --algorithm SVGD --lr 3e-3 --bwType med --seed {}'.format(i) + common_settings)
#     # os.system('python3 main_iter.py --algorithm SGLD --lr 1e-5 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main_iter.py --algorithm SGLDDK --lr 1e-5 --bwType fix --bwVal 1.0 --alpha 1.0 --al_warmup --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm GFSD --lr 3e-3 --bwType nei --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm GFSDCA --lr 3e-3 --bwType nei --alpha 1.0 --al_warmup --seed {}'.format(i) + common_settings)
#     # os.system('python3 main_iter.py --algorithm GFSDDK --lr 0.85e-6 --bwType nei --alpha 1.0 --al_warmup --seed {}'.format(i) + common_settings)
#     # os.system('python3 main_iter.py --algorithm BLOB --lr 0.85e-6 --bwType nei --seed {}'.format(i) + common_settings)
#     # os.system('python3 main_iter.py --algorithm BLOBCA --lr 0.85e-6 --bwType nei --alpha 1.0 --al_warmup --seed {}'.format(i) + common_settings)
#     # os.system('python3 main_iter.py --algorithm BLOBDK --lr 0.85e-6 --bwType nei --alpha 1.0 --al_warmup --seed {}'.format(i) + common_settings)


# common_settings = ' ' + '--task logistic_regression --dataset pima --max_iter 10000 --batch_size 1 --eval_interval 500 --particle_num 128 --save_folder results --device cuda:3'
# for i in range(0, 1): # no differnece between GFSD and GFSD-CA
#     # os.system('python3 main_iter.py --algorithm SVGD --lr 3e-3 --bwType med --seed {}'.format(i) + common_settings)
#     # os.system('python3 main_iter.py --algorithm SGLD --lr 1e-5 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main_iter.py --algorithm SGLDDK --lr 1e-5 --bwType fix --bwVal 1.0 --alpha 1.0 --al_warmup --seed {}'.format(i) + common_settings)
#     # os.system('python3 main_iter.py --algorithm GFSD --lr 4e-5 --bwType nei --seed {}'.format(i) + common_settings)
#     os.system('python3 main_iter.py --algorithm GFSDCA --lr 4e-5 --bwType nei --alpha 1.0 --al_warmup --seed {}'.format(i) + common_settings)
#     # os.system('python3 main_iter.py --algorithm GFSDDK --lr 0.85e-6 --bwType nei --alpha 1.0 --al_warmup --seed {}'.format(i) + common_settings)
#     # os.system('python3 main_iter.py --algorithm BLOB --lr 0.85e-6 --bwType nei --seed {}'.format(i) + common_settings)
#     # os.system('python3 main_iter.py --algorithm BLOBCA --lr 0.85e-6 --bwType nei --alpha 1.0 --al_warmup --seed {}'.format(i) + common_settings)
#     # os.system('python3 main_iter.py --algorithm BLOBDK --lr 0.85e-6 --bwType nei --alpha 1.0 --al_warmup --seed {}'.format(i) + common_settings)

'''
common_settings = ' --task logistic_regression --dataset codrna --epochs 10001 --eval_interval 100 --gpu 3 --particle_num 1024 --save_folder results'
for i in range(0, repeat):
    # os.system('python3 main.py --algorithm SVGD --lr 4e-4 --bwType med --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm SVGDBD --lr 4e-4 --bwType med --alpha 0.01 --seed {}'.format(i) + common_settings)
    os.system('python3 main.py --algorithm GFSD --lr 1e-5 --bwType nei --seed {}'.format(i) + common_settings)
    os.system('python3 main.py --algorithm GFSDBD --lr 1e-5 --bwType nei --alpha 0.1 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm BLOB --lr 3e-5 --bwType nei --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm BLOBBD --lr 3e-5 --bwType nei --alpha 10 --seed {}'.format(i) + common_settings)
'''
'''
common_settings = ' --task logistic_regression --dataset pima --epochs 10001 --eval_interval 10 --gpu 3 --particle_num 256 --save_folder results'
for i in range(0, repeat):
    os.system('python3 main.py --algorithm SVGD --lr 3e-3 --bwType med --seed {}'.format(i) + common_settings)
    os.system('python3 main.py --algorithm SVGDBD --lr 3e-3 --bwType med --seed {}'.format(i) + common_settings)
    os.system('python3 main.py --algorithm GFSD --lr 3e-4 --bwType nei --seed {}'.format(i) + common_settings)
    os.system('python3 main.py --algorithm GFSDBD --lr 3e-4 --bwType nei --seed {}'.format(i) + common_settings)
    os.system('python3 main.py --algorithm BLOB --lr 3e-4 --bwType nei --seed {}'.format(i) + common_settings)
    os.system('python3 main.py --algorithm BLOBBD --lr 3e-4 --bwType nei --seed {}'.format(i) + common_settings)
'''
'''
common_settings = ' --task logistic_regression --dataset a9a --epochs 10001 --eval_interval 10 --gpu 3 --particle_num 256 --save_folder results'
for i in range(0, repeat):
    # os.system('python3 main.py --algorithm SVGD --lr 6e-5 --bwType med --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm SVGDBD --lr 4e-5 --bwType med --alpha 1e-3 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm GFSD --lr 1e-6 --bwType nei --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm GFSDBD --lr 1e-6 --bwType nei --alpha 30 --seed {}'.format(i) + common_settings)
    os.system('python3 main.py --algorithm BLOB --lr 1e-6 --bwType nei --seed {}'.format(i) + common_settings)
    os.system('python3 main.py --algorithm BLOBBD --lr 1e-6 --bwType nei --alpha 30 --seed {}'.format(i) + common_settings)
'''
'''
common_settings = ' --task logistic_regression --dataset ijcnn --epochs 10001 --eval_interval 10 --gpu 3 --particle_num 16 --save_folder results'
for i in range(0, repeat):
    # os.system('python3 main.py --algorithm SVGD --lr 6e-5 --bwType med --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm SVGDBD --lr 4e-5 --bwType med --alpha 1e-3 --seed {}'.format(i) + common_settings)
    os.system('python3 main.py --algorithm GFSD --lr 1e-4 --bwType nei --seed {}'.format(i) + common_settings)
    os.system('python3 main.py --algorithm GFSDBD --lr 1e-4 --bwType nei --alpha 20 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm BLOB --lr 1e-5 --bwType nei --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm BLOBBD --lr 1e-5 --bwType nei --seed {}'.format(i) + common_settings)
'''
'''
common_settings = ' --task logistic_regression --dataset ijcnn --epochs 10001 --eval_interval 10 --gpu 3 --particle_num 16 --save_folder results'
for i in range(0, repeat):
    # os.system('python3 main.py --algorithm SVGD --lr 6e-5 --bwType med --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm SVGDBD --lr 4e-5 --bwType med --alpha 1e-3 --seed {}'.format(i) + common_settings)
    os.system('python3 main.py --algorithm GFSD --lr 1e-4 --bwType nei --seed {}'.format(i) + common_settings)
    os.system('python3 main.py --algorithm GFSDBD --lr 1e-4 --bwType nei --alpha 1e-1 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm BLOB --lr 1e-6 --bwType nei --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm BLOBBD --lr 1e-6 --bwType nei --alpha 1e-1 --seed {}'.format(i) + common_settings)
'''