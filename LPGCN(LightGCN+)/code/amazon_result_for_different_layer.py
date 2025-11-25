import sys
import multiprocessing
import os
from datetime import datetime

# Ensure stdout is line-buffered or unbuffered
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

# dataset = ["gowalla", "yelp2018", "amazon-book"]
dataset = "amazon-book"
seed = [0, 1, 2, 2020, 2021, 2022, 2023, 2025, 2026]
# seed=2024
topks = "[20]"
recdim = 64
batch_size = 2048
testbatch = 200
# group_size=25
# user_batch=100
lr=0.001
layer=[1,2,3]
# decay=[1e-2,1e-3,1e-4,1e-5,1e-6]
layer1_decay=1e-3
layer2_decay=1e-3
layer3_decay=1e-3

# gowalla_layer = 4
# yelp2018_layer = 3
# amazon_book_layer = 4

# gowalla_decay = 1e-4
# yelp2018_decay = 1e-3
# amazon_book_decay = 1e-4


def exec_command(arg):
    os.system(arg)


# params coarse tuning function
def coarse_tune():
    commands = []
    for l in layer:
        for s in seed:
            if l==1:
                d=layer1_decay
            elif l==2:
                d=layer2_decay
            else:
                d=layer3_decay
            cmd = (f"CUDA_VISIBLE_DEVICES=3 python main.py --dataset {dataset} "
                f"--decay={d} --lr={lr} --layer={l} --seed={s} --topks={topks} --recdim={recdim} "
                f"--batch_size={batch_size} --testbatch={testbatch}")
        
            print(cmd, flush=True)
            commands.append(cmd)
    
    print('\n', flush=True)
    pool = multiprocessing.Pool(processes=1)
    for cmd in commands:
        pool.apply_async(exec_command, (cmd,))
    pool.close()
    pool.join()

if __name__ == '__main__':
    coarse_tune()
