import os

repeat = 1
common_settings = ' --task ica_meg --dataset meg --dim 10 --train_num 1000 --epochs 10000 --batch_size 1 --eval_interval 100 --gpu 0 --particle_num 128 --save_folder results'
for i in range(0, repeat):
    os.system('python3 main.py --algorithm SVGD --lr 0.5 --bwType med --seed {}'.format(i) + common_settings)
    os.system('python3 main.py --algorithm GFSD --lr 5e-3 --bwType nei --seed {}'.format(i) + common_settings)
    os.system('python3 main.py --algorithm GFSDBD --lr 5e-3 --bwType nei --alpha 0.5 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm GFSD --lr 1e-3 --bwType nei --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm GFSDBD --lr 1e-3 --bwType nei --alpha 0.5 --seed {}'.format(i) + common_settings)
    os.system('python3 main.py --algorithm BLOB --lr 5e-3 --bwType nei --seed {}'.format(i) + common_settings)
    os.system('python3 main.py --algorithm BLOBBD --lr 5e-3 --bwType nei --alpha 0.1 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm KSDD --lr 1e-3 --bwType fix --bwVal 200 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm KSDDBD --lr 1e-3  --bwType fix --bwVal 200 --alpha 0.2 --seed {}'.format(i) + common_settings)
    #os.system('python3 main.py --algorithm SGLD --lr 1e-4 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm SGLDDK --lr 1e-4 --bwType fix --bwVal 1.0 --alpha 0.5 --seed {}'.format(i) + common_settings)
    #os.system('python3 main.py --algorithm BLOBDK --lr 5e-3 --bwType nei --alpha 0.1 --seed {}'.format(i) + common_settings)
