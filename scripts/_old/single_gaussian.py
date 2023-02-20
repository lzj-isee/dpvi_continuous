import os 

repeat = 10

common_settings = ' --save_particles --task single_gaussian_2d --epochs 2001 --eval_interval 10 --gpu -1 --particle_num 5 --save_folder result_sg/results_5'
for i in range(0, repeat):
    # os.system('python3 main.py --algorithm SVGD --lr 1.0 --bwType med --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm GFSD --lr 0.05 --bwType nei --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm GFSDBD --lr 0.05 --bwType nei --alpha 0.1 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm GFSDDK --lr 0.05 --bwType nei --alpha 0.1 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm BLOB --lr 0.05 --bwType nei --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm BLOBBD --lr 0.05 --bwType nei --alpha 0.1 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm KSDD --lr 0.8 --bwType fix --bwVal 14 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm KSDDBD --lr 0.8 --bwType fix --bwVal 14 --alpha 0.1 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm KSDDDK --lr 0.8 --bwType fix --bwVal 14 --alpha 0.1 --seed {}'.format(i) + common_settings)
    os.system('python3 main.py --algorithm SGLD --lr 0.005 --seed {}'.format(i) + common_settings)
    os.system('python3 main.py --algorithm SGLDDK --lr 0.005 --bwType fix --bwVal 1.0 --alpha 0.1 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm BLOBDK --lr 0.05 --bwType nei --alpha 0.1 --seed {}'.format(i) + common_settings)    
    

common_settings = ' --save_particles --task single_gaussian_2d --epochs 2001 --eval_interval 10 --gpu -1 --particle_num 10 --save_folder result_sg/results_10'
for i in range(0, repeat):
    # os.system('python3 main.py --algorithm SVGD --lr 1.0 --bwType med --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm GFSD --lr 0.1 --bwType nei --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm GFSDBD --lr 0.1 --bwType nei --alpha 0.1 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm GFSDDK --lr 0.1 --bwType nei --alpha 0.1 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm BLOB --lr 0.05 --bwType nei --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm BLOBBD --lr 0.05 --bwType nei --alpha 0.1 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm KSDD --lr 1.0 --bwType fix --bwVal 7.0 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm KSDDBD --lr 1.0 --bwType fix --bwVal 7.0 --alpha 0.05 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm KSDDDK --lr 1.0 --bwType fix --bwVal 7.0 --alpha 0.05 --seed {}'.format(i) + common_settings)
    os.system('python3 main.py --algorithm SGLD --lr 0.005 --seed {}'.format(i) + common_settings)
    os.system('python3 main.py --algorithm SGLDDK --lr 0.005 --bwType fix --bwVal 1.0 --alpha 0.1 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm BLOBDK --lr 0.05 --bwType nei --alpha 0.1 --seed {}'.format(i) + common_settings)


common_settings = ' --save_particles --task single_gaussian_2d --epochs 2001 --eval_interval 10 --gpu -1 --particle_num 20 --save_folder result_sg/results_20'
for i in range(0, repeat):
    # os.system('python3 main.py --algorithm SVGD --lr 1.0 --bwType med --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm GFSD --lr 0.1 --bwType nei --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm GFSDBD --lr 0.1 --bwType nei --alpha 0.1 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm GFSDDK --lr 0.1 --bwType nei --alpha 0.1 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm BLOB --lr 0.05 --bwType nei --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm BLOBBD --lr 0.05 --bwType nei --alpha 0.1 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm KSDD --lr 1.0 --bwType fix --bwVal 5.0 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm KSDDBD --lr 1.0 --bwType fix --bwVal 5.0 --alpha 0.2 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm KSDDDK --lr 1.0 --bwType fix --bwVal 5.0 --alpha 0.2 --seed {}'.format(i) + common_settings)
    os.system('python3 main.py --algorithm SGLD --lr 0.005 --seed {}'.format(i) + common_settings)
    os.system('python3 main.py --algorithm SGLDDK --lr 0.005 --bwType fix --bwVal 1.0 --alpha 0.1 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm BLOBDK --lr 0.05 --bwType nei --alpha 0.1 --seed {}'.format(i) + common_settings)


common_settings = ' --save_particles --task single_gaussian_2d --epochs 2001 --eval_interval 10 --gpu -1 --particle_num 50 --save_folder result_sg/results_50'
for i in range(0, repeat):
    # os.system('python3 main.py --algorithm SVGD --lr 1.0 --bwType med --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm GFSD --lr 0.1 --bwType nei --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm GFSDBD --lr 0.1 --bwType nei --alpha 0.1 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm GFSDDK --lr 0.1 --bwType nei --alpha 0.1 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm BLOB --lr 0.05 --bwType nei --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm BLOBBD --lr 0.05 --bwType nei --alpha 0.1 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm KSDD --lr 1.0 --bwType fix --bwVal 2.0 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm KSDDBD --lr 1.0 --bwType fix --bwVal 2.0 --alpha 0.1 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm KSDDDK --lr 1.0 --bwType fix --bwVal 2.0 --alpha 0.1 --seed {}'.format(i) + common_settings)
    os.system('python3 main.py --algorithm SGLD --lr 0.005 --seed {}'.format(i) + common_settings)
    os.system('python3 main.py --algorithm SGLDDK --lr 0.005 --bwType fix --bwVal 1.0 --alpha 0.1 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm BLOBDK --lr 0.05 --bwType nei --alpha 0.1 --seed {}'.format(i) + common_settings)
    

common_settings = ' --save_particles --task single_gaussian_2d --epochs 2001 --eval_interval 10 --gpu -1 --particle_num 100 --save_folder result_sg/results_100'
for i in range(0, repeat):
    # os.system('python3 main.py --algorithm SVGD --lr 1.0 --bwType med --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm GFSD --lr 0.1 --bwType nei --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm GFSDBD --lr 0.1 --bwType nei --alpha 0.1 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm GFSDDK --lr 0.1 --bwType nei --alpha 0.1 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm BLOB --lr 0.05 --bwType nei --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm BLOBBD --lr 0.05 --bwType nei --alpha 0.1 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm KSDD --lr 1.0 --bwType fix --bwVal 1.0 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm KSDDBD --lr 1.0 --bwType fix --bwVal 1.0 --alpha 0.05 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm KSDDDK --lr 1.0 --bwType fix --bwVal 1.0 --alpha 0.05 --seed {}'.format(i) + common_settings)
    os.system('python3 main.py --algorithm SGLD --lr 0.005 --seed {}'.format(i) + common_settings)
    os.system('python3 main.py --algorithm SGLDDK --lr 0.005 --bwType fix --bwVal 1.0 --alpha 0.1 --seed {}'.format(i) + common_settings)
    # os.system('python3 main.py --algorithm BLOBDK --lr 0.05 --bwType nei --alpha 0.1 --seed {}'.format(i) + common_settings)

