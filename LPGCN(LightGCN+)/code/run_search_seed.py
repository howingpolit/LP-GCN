import sys
import multiprocessing
import os
from datetime import datetime

# Ensure stdout is line-buffered or unbuffered
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

dataset = ["gowalla", "yelp2018", "amazon-book"]
seed = [0, 1, 2, 2020, 2021, 2022, 2023, 2025, 2026]
# seed=2024
topks = "[20]"
recdim = 64
batch_size = 2048
testbatch = 200
# group_size=25
# user_batch=100
lr=0.001
# layer=[1,2,3,4]
# decay=[1e-2,1e-3,1e-4,1e-5,1e-6]

gowalla_layer = 4
yelp2018_layer = 3
amazon_book_layer = 4

gowalla_decay = 1e-4
yelp2018_decay = 1e-3
amazon_book_decay = 1e-4


def exec_command(arg):
    os.system(arg)


# params coarse tuning function
def coarse_tune():
    commands = []
    for ds in dataset:
        for s in seed:
            if ds=="gowalla":
                d=gowalla_decay
                l=gowalla_layer
            elif ds =="yelp2018":
                d=yelp2018_decay
                l=yelp2018_layer
            else:
                d=amazon_book_decay
                l=amazon_book_layer
            cmd = (f"CUDA_VISIBLE_DEVICES=1 python main.py --dataset {ds} "
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
