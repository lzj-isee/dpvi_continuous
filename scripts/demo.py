import os 

common_settings = ' ' + '--task demo --max_iter 15000 --eval_interval 500 --particle_num 128 --save_folder results_demo --device cuda:0'
for i in range(0, 10):
    os.system('python main_iter.py --algorithm SVGD --lr 0.1 --bwType med --anneal 1.0 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm SVGD --lr 0.1 --bwType med --anneal 0.1 --seed {}'.format(i) + common_settings)

    os.system('python main_iter.py --algorithm SGLD --lr 0.1 --anneal 1.0 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm SGLD --lr 0.1 --anneal 0.1 --seed {}'.format(i) + common_settings)

    os.system('python main_iter.py --algorithm SGLDDK --lr 0.02 --bwType fix --bwVal 0.5 --alpha 0.1 --anneal 1.0 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm SGLDDK --lr 0.02 --bwType fix --bwVal 0.5 --alpha 0.1 --anneal 0.1 --seed {}'.format(i) + common_settings)
    
    os.system('python main_iter.py --algorithm GFSD --lr 0.01 --bwType nei --anneal 1.0 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm GFSD --lr 0.01 --bwType nei --anneal 0.1 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm GFSDCA --lr 0.01 --bwType nei --alpha 0.3 --anneal 1.0 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm GFSDCA --lr 0.01 --bwType nei --alpha 0.3 --anneal 0.1 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm GFSDDK --lr 0.01 --bwType nei --alpha 0.3 --anneal 1.0 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm GFSDDK --lr 0.01 --bwType nei --alpha 0.3 --anneal 0.1 --seed {}'.format(i) + common_settings)
    
    os.system('python main_iter.py --algorithm BLOB --lr 0.01 --bwType nei --anneal 1.0 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm BLOB --lr 0.01 --bwType nei --anneal 0.1 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm BLOBCA --lr 0.01 --bwType nei --alpha 0.3 --anneal 1.0 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm BLOBCA --lr 0.01 --bwType nei --alpha 0.3 --anneal 0.1 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm BLOBDK --lr 0.01 --bwType nei --alpha 0.3 --anneal 1.0 --seed {}'.format(i) + common_settings)
    os.system('python main_iter.py --algorithm BLOBDK --lr 0.01 --bwType nei --alpha 0.3 --anneal 0.1 --seed {}'.format(i) + common_settings)