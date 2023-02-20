import os

repeat = 20


# common_settings = ' --task bnn_regression --dataset boston --epochs 3000 --batch_size 128 --eval_interval 100 --gpu 0 --particle_num 128 --save_folder results20/result_boston'
# for i in range(0, repeat):
#     # os.system('python3 main.py --algorithm SVGD --lr 1e-3 --bwType med --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm GFSD --lr 1.0e-5 --bwType nei --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm GFSDBD --lr 1.0e-5 --bwType nei --alpha 1.0 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm BLOB --lr 1.0e-5 --bwType nei --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm BLOBBD --lr 1.0e-5 --bwType nei --alpha 1.0 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm SGLD --lr 1.0e-5 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm SGLDDK --lr 1.0e-5 --bwType fix --bwVal 1.0 --alpha 0.1 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm BLOBDK --lr 1.0e-5 --bwType nei --alpha 0.5 --seed {}'.format(i) + common_settings)
#     os.system('python3 main.py --algorithm GFSDDK --lr 1.0e-5 --bwType nei --alpha 0.5 --seed {}'.format(i) + common_settings)

# common_settings = ' --task bnn_regression --dataset WineRed --epochs 2000 --batch_size 128 --eval_interval 100 --gpu 0 --particle_num 128 --save_folder results20/result_winered'
# for i in range(0, repeat):
#     # os.system('python3 main.py --algorithm SVGD --lr 4.0e-4 --bwType med --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm GFSD --lr 3.4e-6 --bwType nei --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm GFSDBD --lr 3.4e-6 --bwType nei --alpha 0.5 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm BLOB --lr 3.4e-6 --bwType nei --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm BLOBBD --lr 3.4e-6 --bwType nei --alpha 0.5 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm SGLD --lr 4.0e-6 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm SGLDDK --lr 4.0e-6 --bwType fix --bwVal 1.0 --alpha 0.1 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm BLOBDK --lr 3.4e-6 --bwType nei --alpha 0.2 --seed {}'.format(i) + common_settings)
#     os.system('python3 main.py --algorithm GFSDDK --lr 3.4e-6 --bwType nei --alpha 0.2 --seed {}'.format(i) + common_settings)

# common_settings = ' --task bnn_regression --dataset concrete --epochs 2500 --batch_size 128 --eval_interval 100 --gpu 0 --particle_num 128 --save_folder results20/result_concrete'
# for i in range(0, repeat):
#     # os.system('python3 main.py --algorithm SVGD --lr 5.0e-4 --bwType med --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm GFSD --lr 4.0e-6 --bwType nei --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm GFSDBD --lr 4.0e-6 --bwType nei --alpha 0.3 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm BLOB --lr 4.0e-6 --bwType nei --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm BLOBBD --lr 4.0e-6 --bwType nei --alpha 0.3 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm SGLD --lr 4.0e-6 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm SGLDDK --lr 4.0e-6 --bwType fix --bwVal 1.0 --alpha 0.1 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm BLOBDK --lr 4.0e-6 --bwType nei --alpha 0.1 --seed {}'.format(i) + common_settings)
#     os.system('python3 main.py --algorithm GFSDDK --lr 4.0e-6 --bwType nei --alpha 0.1 --seed {}'.format(i) + common_settings)


# common_settings = ' --task bnn_regression --dataset space --epochs 1500 --batch_size 128 --eval_interval 100 --gpu 0 --particle_num 128 --save_folder results20/result_space'
# for i in range(0, repeat):
#     # os.system('python3 main.py --algorithm SVGD --lr 2.4e-4 --bwType med --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm GFSD --lr 2.0e-6 --bwType nei --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm GFSDBD --lr 2.0e-6 --bwType nei --alpha 1.0 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm BLOB --lr 2.0e-6 --bwType nei --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm BLOBBD --lr 2.0e-6 --bwType nei --alpha 1.0 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm SGLD --lr 2.0e-6 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm SGLDDK --lr 2.0e-6 --bwType fix --bwVal 1.0 --alpha 0.1 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm BLOBDK --lr 2.0e-6 --bwType nei --alpha 0.5 --seed {}'.format(i) + common_settings)
#     os.system('python3 main.py --algorithm GFSDDK --lr 2.0e-6 --bwType nei --alpha 0.5 --seed {}'.format(i) + common_settings)


# common_settings = ' --task bnn_regression --dataset energy --epochs 300 --batch_size 128 --eval_interval 100 --gpu 0 --particle_num 128 --save_folder results20/result_energy'
# for i in range(0, repeat):
#     # os.system('python3 main.py --algorithm SVGD --lr 7.0e-5 --bwType med --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm GFSD --lr 4.8e-7 --bwType nei --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm GFSDBD --lr 4.8e-7 --bwType nei --alpha 1.0 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm BLOB --lr 4.8e-7 --bwType nei --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm BLOBBD --lr 4.8e-7 --bwType nei --alpha 1.0 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm SGLD --lr 4.8e-7 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm SGLDDK --lr 4.8e-7 --bwType fix --bwVal 1.0 --alpha 0.1 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm BLOBDK --lr 4.8e-7 --bwType nei --alpha 0.5 --seed {}'.format(i) + common_settings)
#     os.system('python3 main.py --algorithm GFSDDK --lr 4.8e-7 --bwType nei --alpha 0.5 --seed {}'.format(i) + common_settings)


# common_settings = ' --task bnn_regression --dataset kin8nm --epochs 500 --batch_size 128 --eval_interval 100 --gpu 1 --particle_num 128 --save_folder results20/result_kin8nm'
# for i in range(0, repeat):
#     # os.system('python3 main.py --algorithm SVGD --lr 1.0e-4 --bwType med --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm GFSD --lr 1.0e-6 --bwType nei --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm GFSDBD --lr 1.0e-6 --bwType nei --alpha 0.3 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm BLOB --lr 1.0e-6 --bwType nei --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm BLOBBD --lr 1.0e-6 --bwType nei --alpha 0.3 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm SGLD --lr 1.0e-6 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm SGLDDK --lr 1.0e-6 --bwType fix --bwVal 1.0 --alpha 0.1 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm BLOBDK --lr 1.0e-6 --bwType nei --alpha 0.1 --seed {}'.format(i) + common_settings)
#     os.system('python3 main.py --algorithm GFSDDK --lr 1.0e-6 --bwType nei --alpha 0.1 --seed {}'.format(i) + common_settings)


# common_settings = ' --task bnn_regression --dataset electrical --epochs 100 --batch_size 128 --eval_interval 100 --gpu 1 --particle_num 128 --save_folder results20/result_electrical'
# for i in range(0, repeat):
#     # os.system('python3 main.py --algorithm SVGD --lr 1.0e-4 --bwType med --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm GFSD --lr 0.85e-6 --bwType nei --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm GFSDBD --lr 0.85e-6 --bwType nei --alpha 1.0 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm BLOB --lr 0.85e-6 --bwType nei --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm BLOBBD --lr 0.85e-6 --bwType nei --alpha 1.0 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm SGLD --lr 0.85e-6 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm SGLDDK --lr 0.85e-6 --bwType fix --bwVal 1.0 --alpha 0.1 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm BLOBDK --lr 0.85e-6 --bwType nei --alpha 0.5 --seed {}'.format(i) + common_settings)
#     os.system('python3 main.py --algorithm GFSDDK --lr 0.85e-6 --bwType nei --alpha 0.5 --seed {}'.format(i) + common_settings)


# common_settings = ' --task bnn_regression --dataset casp --epochs 200 --batch_size 128 --eval_interval 100 --gpu 1 --particle_num 128 --save_folder results20/result_casp'
# for i in range(0, repeat):
#     # os.system('python3 main.py --algorithm SVGD --lr 3.0e-5 --bwType med --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm GFSD --lr 1.8e-7 --bwType nei --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm GFSDBD --lr 1.8e-7 --bwType nei --alpha 0.5 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm BLOB --lr 1.8e-7 --bwType nei --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm BLOBBD --lr 1.8e-7 --bwType nei --alpha 0.4 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm SGLD --lr 1.8e-7 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm SGLDDK --lr 1.8e-7 --bwType fix --bwVal 1.0 --alpha 0.1 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm BLOBDK --lr 1.8e-7 --bwType nei --alpha 0.2 --seed {}'.format(i) + common_settings)
#     os.system('python3 main.py --algorithm GFSDDK --lr 1.8e-7 --bwType nei --alpha 0.2 --seed {}'.format(i) + common_settings)


# common_settings = ' --task bnn_regression --dataset slice --epochs 50 --batch_size 128 --eval_interval 100 --gpu 1 --particle_num 128 --save_folder results20/result_slice'
# for i in range(0, repeat):
#     # os.system('python3 main.py --algorithm SVGD --lr 1.5e-6 --bwType med --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm GFSD --lr 1.2e-8 --bwType nei --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm GFSDBD --lr 1.2e-8 --bwType nei --alpha 1.0 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm BLOB --lr 1.2e-8 --bwType nei --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm BLOBBD --lr 1.2e-8 --bwType nei --alpha 1.0 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm SGLD --lr 1.2e-8 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm SGLDDK --lr 1.2e-8 --bwType fix --bwVal 1.0 --alpha 0.1 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm BLOBDK --lr 1.2e-8 --bwType nei --alpha 0.5 --seed {}'.format(i) + common_settings)
#     os.system('python3 main.py --algorithm GFSDDK --lr 1.2e-8 --bwType nei --alpha 0.5 --seed {}'.format(i) + common_settings)


common_settings = ' --task bnn_regression --dataset superconduct --epochs 200 --batch_size 128 --eval_interval 100 --gpu 1 --particle_num 128 --save_folder results20/result_superconduct'
for i in range(19, repeat):
    # os.system('python3 main.py --algorithm SVGD --lr 1.5e-5 --bwType med --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm GFSD --lr 1.2e-7 --bwType nei --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm GFSDBD --lr 1.2e-7 --bwType nei --alpha 1.0 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm BLOB --lr 1.2e-7 --bwType nei --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm BLOBBD --lr 1.2e-7 --bwType nei --alpha 1.0 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm SGLD --lr 1.2e-7 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm SGLDDK --lr 1.2e-7 --bwType fix --bwVal 1.0 --alpha 0.1 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm BLOBDK --lr 1.2e-7 --bwType nei --alpha 0.5 --seed {}'.format(i) + common_settings)
    os.system('python3 main.py --algorithm GFSDDK --lr 1.2e-7 --bwType nei --alpha 0.5 --seed {}'.format(i) + common_settings)


# common_settings = ' --task bnn_regression --dataset cpusmall --epochs 50 --batch_size 128 --eval_interval 100 --gpu 1 --particle_num 128 --save_folder results20/result_cpusmall'
# for i in range(0, repeat):
#     # os.system('python3 main.py --algorithm SVGD --lr 5e-5 --bwType med --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm GFSD --lr 5e-7 --bwType nei --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm GFSDBD --lr 5e-7 --bwType nei --alpha 1.0 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm BLOB --lr 5e-7 --bwType nei --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm BLOBBD --lr 5e-7 --bwType nei --alpha 1.0 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm SGLD --lr 5e-7 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm SGLDDK --lr 5e-7 --bwType fix --bwVal 1.0 --alpha 0.1 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main.py --algorithm BLOBDK --lr 5e-7 --bwType nei --alpha 0.5 --seed {}'.format(i) + common_settings)
#     os.system('python3 main.py --algorithm GFSDDK --lr 5e-7 --bwType nei --alpha 0.5 --seed {}'.format(i) + common_settings)
