'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''

import os
from os.path import join
import torch
from enum import Enum
from parse import parse_args
import multiprocessing
import time

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()

ROOT_PATH = "../"
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
RUNS_PATH = join(CODE_PATH, 'runs')
RESULT_PATH = join(RUNS_PATH, args.dataset + f"_layer={args.layer}_decay={args.decay}_" + time.strftime("%m-%d-%Hh%Mm%Ss") + "_" + args.comment)
ARGS_PATH = join(RESULT_PATH, 'args.txt')
RES_PATH = join(RESULT_PATH, 'res.txt')
TEST_PATH = join(RESULT_PATH, 'tensorboard_TEST')
VAL_PATH = join(RESULT_PATH, 'tensorboard_VAL')
WEIGHT_PATH = join(RESULT_PATH, 'checkpoints')

if not os.path.exists(RUNS_PATH):
    os.makedirs(RUNS_PATH, exist_ok=True)
os.makedirs(RESULT_PATH, exist_ok=True)
os.makedirs(TEST_PATH, exist_ok=True)
os.makedirs(VAL_PATH, exist_ok=True)
os.makedirs(WEIGHT_PATH, exist_ok=True)
with open(ARGS_PATH, "a", newline='') as f:
    for key,val in vars(args).items():
        f.write(f'{{{key}}}: {val}\n')


config = {}
all_dataset = ['lastfm', 'gowalla', 'yelp2018', 'amazon-book']
all_models  = ['mf', 'lgn']
# config['batch_size'] = 4096
config['bpr_batch_size'] = args.batch_size
config['latent_dim_rec'] = args.recdim
config['lightGCN_n_layers']= args.layer
config['dropout'] = args.dropout
config['keep_prob']  = args.keepprob
config['A_n_fold'] = args.a_fold
config['test_u_batch_size'] = args.testbatch
config['multicore'] = args.multicore
config['lr'] = args.lr
config['decay'] = args.decay
config['pretrain'] = args.pretrain
config['A_split'] = False
config['bigdata'] = False

GPU = torch.cuda.is_available()
device = torch.device('cuda:0' if GPU else "cpu")
CORES = multiprocessing.cpu_count() // 2
seed = args.seed

dataset = args.dataset
model_name = args.model
if dataset not in all_dataset:
    raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")
if model_name not in all_models:
    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")




TRAIN_epochs = args.epochs
LOAD = args.load
LOAD_PATH = args.loadpath
topks = eval(args.topks)
tensorboard = args.tensorboard
comment = args.comment
# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)



def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")

logo = r"""
██╗      ██████╗ ███╗   ██╗
██║     ██╔════╝ ████╗  ██║
██║     ██║  ███╗██╔██╗ ██║
██║     ██║   ██║██║╚██╗██║
███████╗╚██████╔╝██║ ╚████║
╚══════╝ ╚═════╝ ╚═╝  ╚═══╝
"""
# font: ANSI Shadow
# refer to http://patorjk.com/software/taag/#p=display&f=ANSI%20Shadow&t=Sampling
# print(logo)
