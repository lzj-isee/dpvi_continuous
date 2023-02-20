import os

repeat = 5
common_settings = ' --task ica --train_num 1000 --dim 2 --ksd_h 1.0 --train_num 1000 --epochs 10000 --batch_size 1 --eval_interval 50 --gpu 3 --particle_num 1024 --save_folder results'
for i in range(0, repeat):
    # os.system('python3 main.py --algorithm SVGD --lr 1e-3 --bwType med --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm GFSD --lr 2e-6 --bwType nei --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm GFSDBD --lr 2e-6 --bwType nei --alpha 1.0 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm BLOB --lr 2e-6 --bwType nei --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm BLOBBD --lr 2e-6 --bwType nei --alpha 1.0 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm KSDD --lr 2e-5 --bwType fix --bwVal 2 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm KSDDBD --lr 2e-5  --bwType fix --bwVal 2 --alpha 1.0 --seed {}'.format(i) + common_settings)