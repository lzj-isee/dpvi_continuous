import os

repeat = 20

common_settings = ' --far_mode --save_particles --task gaussian_process --dataset lidar --epochs 6200 --batch_size 1 --eval_interval 200 --gpu 3 --particle_num 128 --save_folder result_gp_fm'
for i in range(0, repeat):
    # os.system('python3 main.py --algorithm SVGD --lr 8.0e-2 --bwType med --seed {}'.format(i) + common_settings)


    os.system('python3 main.py --algorithm GFSD --lr 4.0e-3 --bwType nei --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm GFSDBD --lr 4.0e-3 --bwType nei --alpha 0.3 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm GFSDDK --lr 4.0e-3 --bwType nei --alpha 0.3 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm GFSDBDDK --lr 4.0e-3 --bwType nei --alpha 0.3 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm BLOB --lr 4.0e-3 --bwType nei --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm BLOBBD --lr 4.0e-3 --bwType nei --alpha 0.2 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm BLOBDK --lr 4.0e-3 --bwType nei --alpha 0.2 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm BLOBBDDK --lr 4.0e-3 --bwType nei --alpha 0.2 --seed {}'.format(i) + common_settings)


    # os.system('python3 main.py --algorithm KSDD --lr 2e-2 --bwType fix --bwVal  0.1 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm KSDDBD --lr 2e-2 --bwType fix --bwVal 0.1 --alpha 0.2 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm SGLD --lr 2.0e-3 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm SGLDDK --lr 2.0e-3 --bwType fix --bwVal 1.0 --alpha 0.2 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm KSDDDK --lr 2e-2 --bwType fix --bwVal 0.1 --alpha 0.2 --seed {}'.format(i) + common_settings)