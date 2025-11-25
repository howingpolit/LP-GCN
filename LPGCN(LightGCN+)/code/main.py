import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
import csv
import warnings
warnings.filterwarnings('ignore')

# ==============================
utils.set_torch_seed(world.seed)
utils.set_numpy_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset


with open(world.train_loss_process_record_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['epoch','train_loss'])

with open(world.evolution_process_record_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['epoch', 'recall@20','NDCG@20'])


Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)
stopping_step = 0
cur_best_pre = 0.
best_res = {}
should_stop = False
early_stopping_step = 50
Neg_k = 1


if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(world.LOAD_PATH,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {world.LOAD_PATH}")
    except FileNotFoundError:
        print(f"{world.LOAD_PATH} not exists, start from beginning")


# init tensorboard
if world.tensorboard:
    wTest : SummaryWriter = SummaryWriter(world.TEST_PATH)
    wVal : SummaryWriter = SummaryWriter(world.VAL_PATH)
else:
    wTest = None
    wVal = None
    world.cprint("not enable tensorflowboard")

try:
    for epoch in range(1, world.TRAIN_epochs+1):
        # start = time.time()
        output_information,aver_loss = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=wTest)
        print(f'EPOCH[{epoch}/{world.TRAIN_epochs}] {output_information}')
        # print('Train time(s):', int(time.time()-start))
        with open(world.train_loss_process_record_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, aver_loss])

        if epoch % 5 == 0:
            # start = time.time()
            cprint("[VALIDATION]")
            cur_pre = Procedure.Test(dataset, Recmodel, epoch, wVal, world.config['multicore'], test=0)
            cprint("[TEST]")
            res = Procedure.Test(dataset, Recmodel, epoch, wTest, world.config['multicore'], test=1)
            # print('Test time(s):', int(time.time()-start))

            with open(world.evolution_process_record_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch, res["recall"][0],res["ndcg"][0]])


            cur_best_pre, best_res, stopping_step, should_stop = utils.early_stopping(cur_pre, cur_best_pre, res, best_res,
                                                    stopping_step, expected_order='acc', flag_step=early_stopping_step)



            # early stopping when cur_best_pre is decreasing for early_stopping_step.
            if should_stop == True:
                with open(world.RES_PATH, "a", newline="") as f:
                    f.write("Early stopping is triggered, the best epoch is " + str(epoch - early_stopping_step))
                break

        # if epoch % 100 == 0:
        #     file = f"{world.dataset}-{world.config['lightGCN_n_layers']}-{world.config['latent_dim_rec']}-epoch{epoch}.pth.tar"
        #     weight_path = join(world.WEIGHT_PATH, file)
        #     torch.save(Recmodel.state_dict(), weight_path)
finally:
    with open(world.RES_PATH, "a", newline="") as f:
        f.write("VALIDATION\n")
        f.write("recall: " + str(cur_best_pre))
        f.write("\nTEST\n")
        for key,val in best_res.items():
            f.write(f'{key}: {val}\n')

    if world.tensorboard:
        wTest.close()
        wVal.close()