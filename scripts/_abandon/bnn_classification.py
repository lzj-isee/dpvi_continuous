import os 

repeat = 1

common_settings = ' --task bnn_classification --dataset usps --epochs 100 --batch_size 32 --eval_interval 10 --gpu 3 --particle_num 256 --save_folder results'
for i in range(0, repeat):
    # os.system('python3 main.py --algorithm SVGD --lr 2e-4 --bwType med --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm SVGDBD --lr 2e-4 --bwType med --alpha 2e-3 --seed {}'.format(i) + common_settings)
    os.system('python3 main.py --algorithm GFSD --lr 2e-6 --bwType nei --seed {}'.format(i) + common_settings)
    os.system('python3 main.py --algorithm GFSDBD --lr 2e-6 --bwType nei --alpha 0.1 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm BLOB --lr 3e-6 --bwType nei --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm BLOBBD --lr 3e-6 --bwType nei --alpha 0.3 --seed {}'.format(i) + common_settings)
