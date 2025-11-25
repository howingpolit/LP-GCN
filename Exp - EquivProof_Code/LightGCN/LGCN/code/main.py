from Dataloader import Dataloader
from sample import Sample
from model import Light_GCN
from configue import configues
from torch import optim
import procedure
import time
from torch.utils.tensorboard import SummaryWriter
import csv
dataloader=Dataloader()
sampeler=Sample(dataloader)
model=Light_GCN(dataloader,configues=configues)
model=model.to(configues["device"])
opt=optim.Adam(model.parameters(),lr=configues["lr"])
tb_writer=SummaryWriter(log_dir="../experiment/tensorboard")

best_precision=0
best_recall=0
best_NDCG_5=0
best_NDCG_10=0
best_NDCG_15=0
best_NDCG_20=0
best_epoch=0

#
evaluation_filename = f"../experiment/evaluation_results-{configues['layer']}-{configues['weight_decay']}.csv"
train_loss_filename = f"../experiment/train_loss-{configues['layer']}-{configues['weight_decay']}.csv"
#
with open(evaluation_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['epoch', 'precision', 'recall','NDCG_5','NDCG_10','NDCG_15','NDCG_20'])

#
with open(train_loss_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['epoch', 'loss'])

for epoch in range(configues["epochs"]):
    #if epoch % 10 == 0:
        #print("test")
    precision,recall,NDCG_5,NDCG_10,NDCG_15,NDCG_20=procedure.test(model,configues,dataloader)
    print("precision:"+str(precision))
    print("recall:"+str(recall))
    print("NDCG_5:"+str(NDCG_5))
    print("NDCG_10:"+str(NDCG_10))
    print("NDCG_15:"+str(NDCG_15))
    print("NDCG_20:"+str(NDCG_20))
    #
    with open(evaluation_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch+1, precision, recall,NDCG_5,NDCG_10,NDCG_15,NDCG_20])

    if precision>best_precision:
        best_precision=precision
        best_recall=recall
        best_NDCG_5=NDCG_5
        best_NDCG_10=NDCG_10
        best_NDCG_15=NDCG_15
        best_NDCG_20=NDCG_20
        best_epoch=epoch

    tb_writer.add_scalar("precision",precision,epoch)
    tb_writer.add_scalar("recall", recall, epoch)
    start=time.time()
    aver_loss=procedure.train(sampeler,dataloader,model,configues,opt,epoch)
    tb_writer.add_scalar("aver_loss",aver_loss,epoch)
    end=time.time()
    #print(end-start)
    print("epoch-"+str(epoch+1)+":"+str(aver_loss))
    print("----------------------------------------------")
    #
    with open(train_loss_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch+1, aver_loss])
print("best_precision:",best_precision)
print("best_recall:",best_recall)
print("best_NDCG_5:",best_NDCG_5)
print("best_NDCG_10:",best_NDCG_10)
print("best_NDCG_15:",best_NDCG_15)
print("best_NDCG_20:",best_NDCG_20)
print("best_epoch:",best_epoch+1)

with open(evaluation_filename, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["best_result", best_precision, best_recall, best_NDCG_5,best_NDCG_10,best_NDCG_15,best_NDCG_20])
    writer.writerow(["best_epoch", best_epoch+1])

