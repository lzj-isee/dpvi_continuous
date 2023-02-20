import os 

repeat = 1


common_settings = ' --lr_hmc 1e-5 --lr_gd_init 1e-6'
common_settings += ' --task logistic_regression --dataset codrna --epochs 30001 --eval_interval 200 --gpu 3 --particle_num 128 --save_folder results'
for i in range(0, repeat):
    #os.system('python3 main.py --algorithm SVGD --lr 6e-1 --bwType med --seed {}'.format(i) + common_settings)
    os.system('python3 main.py --algorithm GFSD --lr 4.0e-4 --bwType nei --seed {}'.format(i) + common_settings)
    #os.system('python3 main.py --algorithm GFSDBD --lr 4.0e-4 --bwType nei --alpha 0.1 --seed {}'.format(i) + common_settings)
    #os.system('python3 main.py --algorithm BLOB --lr 2e-4 --bwType nei --seed {}'.format(i) + common_settings)
    #os.system('python3 main.py --algorithm BLOBBD --lr 2e-4 --bwType nei --alpha 0.5 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm KSDD --lr 10 --bwType fix --bwVal 10 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm KSDDBD --lr 10 --bwType fix --bwVal 10 --alpha 0.1 --seed {}'.format(i) + common_settings)

# common_settings = ' --lr_hmc 0.03 --lr_gd_init 1e-5'
# common_settings += ' --task logistic_regression --dataset heart --epochs 5001 --eval_interval 100 --gpu 3 --particle_num 128 --save_folder results'
# for i in range(0, repeat):
#     # os.system('python3 main.py --algorithm SVGD --lr 4e-4 --bwType med --seed {}'.format(i) + common_settings)
#     #os.system('python3 main.py --algorithm GFSD --lr 1e-4 --bwType nei --seed {}'.format(i) + common_settings)
#     os.system('python3 main.py --algorithm GFSDBD --lr 1e-4 --bwType nei --alpha 0.1 --seed {}'.format(i) + common_settings)
#     #os.system('python3 main.py --algorithm BLOB --lr 1e-4 --bwType nei --seed {}'.format(i) + common_settings)
#     #os.system('python3 main.py --algorithm BLOBBD --lr 1e-4 --bwType nei --alpha 0.6 --seed {}'.format(i) + common_settings)

# common_settings = ' --lr_hmc 0.03 --lr_gd_init 1e-5'
# common_settings += ' --task logistic_regression --dataset fourclass --epochs 2001 --eval_interval 100 --gpu 3 --particle_num 1024 --save_folder results'
# for i in range(0, repeat):
#     #os.system('python3 main.py --algorithm SVGD --lr 3e-2 --bwType med --seed {}'.format(i) + common_settings)
#     #os.system('python3 main.py --algorithm GFSD --lr 3e-4 --bwType nei --seed {}'.for# common_settings = ' --lr_hmc 0.01 --lr_gd_init 1e-5'
# common_settings += ' --task logistic_regression --dataset pima --epochs 10001 --eval_interval 200 --gpu 3 --particle_num 128 --save_folder results'
# for i in range(0, repeat):
#     #os.system('python3 main.py --algorithm SVGD --lr 6e-1 --bwType med --seed {}'.format(i) + common_settings)
#     os.system('python3 main.py --algorithm GFSD --lr 1.0e-5 --bwType nei --seed {}'.format(i) + common_settings)
#     os.system('python3 main.py --algorithm GFSDBD --lr 1.0e-5 --bwType nei --alpha 1.0 --seed {}'.format(i) + common_settings)
#     os.system('python3 main.py --algorithm BLOB --lr 1.0e-5 --bwType nei --seed {}'.format(i) + common_settings)
#     os.system('python3 main.py --algorithm BLOBBD --lr 1.0e-5 --bwType nei --alpha 1.0 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm KSDD --lr 10 --bwType fix --bwVal 10 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm KSDDBD --lr 10 --bwType fix --bwVal 10 --alpha 0.1 --seed {}'.format(i) + common_settings)mat(i) + common_settings)
#     #os.system('python3 main.py --algorithm GFSDBD --lr 3e-4 --bwType nei --alpha 0.6 --seed {}'.format(i) + common_settings)
#     #os.system('python3 main.py --algorithm BLOB --lr 3e-4 --bwType nei --seed {}'.format(i) + common_settings)
#     #os.system('python3 main.py --algorithm BLOBBD --lr 3e-4 --bwType nei --alpha 0.6 --seed {}'.format(i) + common_settings)
#     #os.system('python3 main.py --algorithm KSDD --lr 1.2e-4 --bwType fix --bwVal 0.05 --seed {}'.format(i) + common_settings)
#     #os.system('python3 main.py --algorithm KSDDBD --lr 1.2e-4 --bwType fix --bwVal 0.05 --alpha 0.1 --seed {}'.format(i) + common_settings)
#     #os.system('python3 main.py --algorithm SGLD --lr 3e-4 --seed {}'.format(i) + common_settings)
#     #os.system('python3 main.py --algorithm SGLDDK --lr 3e-4 --bwType fix --bwVal 1.0 --alpha 0.1 --seed {}'.format(i) + common_settings)
#     #os.system('python3 main.py --algorithm BLOBDK --lr 3e-4 --bwType nei --alpha 0.6 --seed {}'.format(i) + common_settings)




















# common_settings = ' --task logistic_regression --temper 100 --ksd_h 100 --dataset a3a --epochs 10001 --eval_interval 200 --gpu 3 --particle_num 1024 --save_folder results'
# for i in range(0, repeat):
#     os.system('python3 main.py --algorithm SVGD --lr 1e-1 --bwType med --seed {}'.format(i) + common_settings)
#     os.system('python3 main.py --algorithm GFSD --lr 2e-3 --bwType nei --seed {}'.format(i) + common_settings)
#     os.system('python3 main.py --algorithm GFSDBD --lr 2e-3 --bwType nei --alpha 0.1 --seed {}'.format(i) + common_settings)
#     os.system('python3 main.py --algorithm BLOB --lr 2e-3 --bwType nei --seed {}'.format(i) + common_settings)
#     os.system('python3 main.py --algorithm BLOBBD --lr 2e-3 --bwType nei --alpha 0.1 --seed {}'.format(i) + common_settings)
#     os.system('python3 main.py --algorithm KSDD --lr 1e-2 --bwType fix --bwVal 200 --seed {}'.format(i) + common_settings)
#     os.system('python3 main.py --algorithm KSDDBD --lr 1e-2 --bwType fix --bwVal 200 --alpha 0.1 --seed {}'.format(i) + common_settings)



# common_settings = ' --task logistic_regression --temper 10 --ksd_h 1 --dataset heart --epochs 100001 --eval_interval 100 --gpu 3 --particle_num 1024 --save_folder results'
# for i in range(0, repeat):
#     # os.system('python3 main.py --algorithm SVGD --lr 4e-4 --bwType med --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm SVGDBD --lr 4e-4 --bwType med --alpha 0.01 --seed {}'.format(i) + common_settings)
#     os.system('python3 main.py --algorithm GFSD --lr 4e-4 --bwType nei --seed {}'.format(i) + common_settings)
#     os.system('python3 main.py --algorithm GFSDBD --lr 4e-4 --bwType nei --alpha 0.1 --seed {}'.format(i) + common_settings)
#     os.system('python3 main.py --algorithm BLOB --lr 4e-4 --bwType nei --seed {}'.format(i) + common_settings)
#     os.system('python3 main.py --algorithm BLOBBD --lr 4e-4 --bwType nei --alpha 0.1 --seed {}'.format(i) + common_settings)

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