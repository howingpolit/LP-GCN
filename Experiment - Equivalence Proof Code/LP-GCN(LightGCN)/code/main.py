from dataloader import Dataloader
import numpy as np
from configue import configues
import torch
from server import Server
import utils
from torch import optim
from client_classify import find_minimum_user_set_sparse
from convolution_client import Convolution_Client
from ordinary_client import Ordinary_Client
import time
import evalution
import csv
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



#
dataloader=Dataloader()

# print(dataloader.user_number)
# print(dataloader.item_number)
#
server=Server(dataloader=dataloader)
server.split_clients()
server.mark_convolution_items()
server.get_convolution_items_of_convolution_clients()
# print(server.convolution_items_of_convolution_clients)


#
torch.manual_seed(configues["seed"])
users_embedding_weight = torch.nn.Embedding(num_embeddings=dataloader.user_number, embedding_dim=configues["embedding_dim"]).weight
items_embedding_weight = torch.nn.Embedding(num_embeddings=dataloader.item_number, embedding_dim=configues["embedding_dim"]).weight
# print(items_embedding_weight[15])

#----------------------------------
#
convolution_clients={}
for u_id in server.convolution_clients_id_set:
    item_id=list(dataloader.train_user_item_matrix[u_id].nonzero()[1])
    convolution_client=Convolution_Client(u_id,item_id)
    user_embedding=users_embedding_weight.data[u_id]
    convolution_client.user_embedding[0]=user_embedding

    for i in item_id:
        item_embedding=items_embedding_weight.data[i]
        convolution_client.item_embedding[convolution_client.hash_id_to_index_of_item[i]] = item_embedding
    convolution_clients[u_id]=convolution_client

#
ordinary_clients={}
for u_id in server.ordinary_clients_id_set:
    item_id=list(dataloader.train_user_item_matrix[u_id].nonzero()[1])
    ordinary_client=Ordinary_Client(u_id,item_id)
    user_embedding=users_embedding_weight.data[u_id]
    ordinary_client.user_embedding[0]=user_embedding

    for i in item_id:
        item_embedding=items_embedding_weight.data[i]
        ordinary_client.item_embedding[ordinary_client.hash_id_to_index_of_item[i]] = item_embedding
    ordinary_clients[u_id]=ordinary_client
#
clients = {**convolution_clients, **ordinary_clients}
#
for u_id,convolution_client in convolution_clients.items():
    convolution_client.convolution_items_id=server.convolution_items_of_convolution_clients[u_id]
    # print(convolution_client.convolution_items_id)
    convolution_client.get_ordinary_items_id()  #

#----------------------------------------------------

#
#
server.get_subgraph_data_structure_of_convolution_clients()

# print(server.subgraph_data_structure_of_convolution_clients.keys())
# print(type(server.subgraph_data_structure_of_convolution_clients[4]))
# print(type(server.subgraph_data_structure_of_convolution_clients[4][0]))
#
for con_client in convolution_clients.values():
    con_client_id = con_client.user_id
    con_client.subgraph_data_structure=server.subgraph_data_structure_of_convolution_clients[con_client_id]


for con_client in convolution_clients.values():
    con_client.get_neighbor_users_id()
    con_client.get_hash_index_to_id_of_neighbor_users()
    con_client.get_hash_id_to_index_of_neighbor_users()


server.get_degree_of_all_users()

server.get_degree_of_all_items()
# print(convolution_clients.keys())

for con_client in convolution_clients.values():
    con_client.get_neighbor_users_degree(server.degree_of_all_users)
    # print(con_client.user_id)
    con_client.get_item_degree(server.degree_of_all_items)
    con_client.get_adjacency_matrix()

#
for ord_client in ordinary_clients.values():
    ord_client.get_item_degree(server.degree_of_all_items)
    ord_client.get_adjacency_matrix()

for con_client in convolution_clients.values():
    con_client.initialize_all_layers_embeddings()
    con_client.initial_embeddings_gd()

#
for ord_client in ordinary_clients.values():
    ord_client.initialize_all_layers_embeddings()
    ord_client.initial_embedding_gd()

#
# for client in clients.values():
#     client.initial_embeddings_gd()



#
embedding_study=[]
#
for con_client in convolution_clients.values():
    embedding_study.append(con_client.embeddings[0].weight)
#
for ord_client in ordinary_clients.values():
    embedding_study.append(ord_client.embeddings[0].weight)
#
opt=optim.Adam(embedding_study,lr=configues["lr"])


#
for epoch in range(configues["epochs"]):
    test_flag = 1
    #
    batch_size = configues["train_batch"]
    S = server.sample_pairs(epoch)

    #
    users=S[:,0]
    pos_items = S[:,1]
    neg_items = S[:,2]


    users,pos_items,neg_items=utils.shuffle(users,pos_items,neg_items)
    iter_time=len(users)//batch_size+1
    aver_loss=0.

    flag=1
    for (batch_users,batch_pos_items,batch_neg_items) in utils.minibatch(users,
                                                                         pos_items,
                                                                         neg_items,
                                                                         batch_size=configues["train_batch"]):
        # print(batch_users)
        # print(batch_pos_items)
        # print(batch_neg_items)
        opt.zero_grad()
        #
        for l in range(configues["layer"]-1):
            l_th_user_embedding_of_all_clients={}   #{user_id:user_embedding}
            #
            for u_id, convolution_client in convolution_clients.items():
                l_th_user_embedding_of_all_clients[u_id]=convolution_client.send_l_th_user_embedding(l)
            #
            for u_id, ord_client in ordinary_clients.items():
                l_th_user_embedding_of_all_clients[u_id]=ord_client.send_l_th_user_embedding(l)
            #
            # if l==1:
            #     print(1)
            #
            for convolution_client in convolution_clients.values():
                neighbor_users_embedding_mid={}
                for i in convolution_client.neighbor_users_id:
                    neighbor_users_embedding_mid[i]=l_th_user_embedding_of_all_clients[i]
                convolution_client.neighbor_users_embedding=neighbor_users_embedding_mid
            #
            for convolution_client in convolution_clients.values():
                convolution_client.update_l_th_neighbor_users_embedding(l)
                convolution_client.GNN(l)
            #
            for ordinary_client in ordinary_clients.values():
                ordinary_client.GNN(l)
            #
            l_plus_1_item_embeddings={}
            for convolution_client in convolution_clients.values():
                l_plus_1_item_embeddings_of_convolution_client=convolution_client.send_l_plus_1_th_con_item_embeddings(l)
                l_plus_1_item_embeddings={**l_plus_1_item_embeddings,**l_plus_1_item_embeddings_of_convolution_client}
            server.l_plus_1_item_embeddings=l_plus_1_item_embeddings
            server.all_item_embedding_of_all_layer[l+1]=l_plus_1_item_embeddings

            for convolution_client in convolution_clients.values():
                ord_items_embedding_mid = {}
                for i in convolution_client.ordinary_items_id:
                    ord_items_embedding_mid[i]=server.l_plus_1_item_embeddings[i]
                convolution_client.ord_items_embedding=ord_items_embedding_mid
                convolution_client.update_l_plus_1_ord_item_embedding(l)

            for ordinary_client in ordinary_clients.values():
                ord_items_embedding_mid = {}
                for i in ordinary_client.item_id:
                    ord_items_embedding_mid[i]=server.l_plus_1_item_embeddings[i]
                ordinary_client.ord_items_embedding=ord_items_embedding_mid
                ordinary_client.update_l_plus_1_ord_item_embedding(l)


        zero_th_item_embeddings = {}
        for convolution_client in convolution_clients.values():
            zero_th_item_embeddings_of_convolution_client = convolution_client.send_zero_th_con_item_embeddings()
            zero_th_item_embeddings = {**zero_th_item_embeddings, **zero_th_item_embeddings_of_convolution_client}

        server.all_item_embedding_of_all_layer[0]=zero_th_item_embeddings   #{layer:{item_id:embedding}}


        negative_item_id_of_sample_clients=server.get_negative_item_id_of_sample_clients(batch_users,batch_neg_items)


        for sample_client in negative_item_id_of_sample_clients.keys():

            clients[sample_client].negative_item_id = negative_item_id_of_sample_clients[sample_client]

            clients[sample_client].get_hash_id_to_index_of_negative_item()
            clients[sample_client].get_hash_index_to_id_of_negative_item()
            clients[sample_client].initial_negative_item_embedding()

            for index in range(len(clients[sample_client].negative_item_id)):
                negative_item_id = clients[sample_client].hash_index_to_id_of_negative_item[index]
                for i, value in enumerate(clients[sample_client].negative_item_embedding):
                    # print(index,i,negative_item_id)
                    value.weight.data[index] = server.all_item_embedding_of_all_layer[i][negative_item_id]



        sample_pairs_for_loss_of_sample_clients=server.get_sample_pairs_for_loss(batch_users,batch_pos_items,batch_neg_items)

        sum_loss=0.0
        for sample_client in negative_item_id_of_sample_clients.keys():
            clients[sample_client].sample_pairs_for_loss = sample_pairs_for_loss_of_sample_clients[sample_client]
            clients[sample_client].batch_size=float(len(batch_users))
            loss_of_one_client=clients[sample_client].get_loss()
            sum_loss+=loss_of_one_client
        aver_loss+=sum_loss
        #print("-------------------")
        # print(sum_loss.item())

        #-----------------------------------------------------------
        #
        # if flag:
        #     test()
        #     flag=0

        #
        if test_flag:
            user_embedding_matrix,item_embedding_matrix=utils.get_user_and_item_embedding_matrix(clients,server)
            evalutioner = evalution.Evalution(user_embedding_matrix,item_embedding_matrix,dataloader,configues)
            precision = evalutioner.get_precision_of_all_u()
            recall = evalutioner.get_recall_of_all_u()
            NDCG_5 = evalutioner.get_NDCG_of_all_u(5)
            NDCG_10 = evalutioner.get_NDCG_of_all_u(10)
            NDCG_15 = evalutioner.get_NDCG_of_all_u(15)
            NDCG_20 = evalutioner.get_NDCG_of_all_u(15)
            print("precision:" + str(precision))
            print("recall:" + str(recall))
            print("NDCG_5:" + str(NDCG_5))
            print("NDCG_10:" + str(NDCG_10))
            print("NDCG_15:" + str(NDCG_15))
            print("NDCG_20:" + str(NDCG_20))
            #
            with open(evaluation_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch + 1, precision, recall, NDCG_5, NDCG_10, NDCG_15, NDCG_20])

            if precision > best_precision:
                best_precision = precision
                best_recall = recall
                best_NDCG_5 = NDCG_5
                best_NDCG_10 = NDCG_10
                best_NDCG_15 = NDCG_15
                best_NDCG_20 = NDCG_20
                best_epoch = epoch
            test_flag=0

        #
        for sample_client in negative_item_id_of_sample_clients.keys():
            clients[sample_client].loss_backword()
        #

        #
        for sample_client in negative_item_id_of_sample_clients.keys():
            clients[sample_client].get_last_layer_embedding_gd()


        negative_item_gd={}     #{item_id:}
        for i in batch_neg_items:
            negative_item_gd[i]=torch.zeros(configues["embedding_dim"])
        #
        for sample_client in negative_item_id_of_sample_clients.keys():
            neg_item_gd_of_one_client=clients[sample_client].send_l_th_neg_item_gd(configues["layer"]-1)
            for i in neg_item_gd_of_one_client.keys():
                negative_item_gd[i]+=neg_item_gd_of_one_client[i]
        #
        ordinary_item_gd={}
        for i in range(dataloader.item_number):
            ordinary_item_gd[i]=torch.zeros(configues["embedding_dim"])
        for client in clients.values():
            ord_item_gd_of_one_client=client.send_current_l_th_ord_item_gd()
            client.reset_ord_item_embedding_gd()
            for i in ord_item_gd_of_one_client.keys():
                ordinary_item_gd[i] += ord_item_gd_of_one_client[i]
        #
        for i in negative_item_gd.keys():
            ordinary_item_gd[i] += negative_item_gd[i]

        server.ord_and_neg_embedding_gd_of_last_layer=ordinary_item_gd
        #
        for id,con_client in convolution_clients.items():
            con_items_id=con_client.convolution_items_id
            con_embedding_gd = {}
            for i in con_items_id:
                con_embedding_gd[i]=server.ord_and_neg_embedding_gd_of_last_layer[i]
            con_client.con_embedding_gd=con_embedding_gd
            con_client.update_embedding_gd()
        #
        #
        for layer in range(configues["layer"]-2,-1,-1):
            #
            item_gd = {}
            for i in range(dataloader.item_number):
                item_gd[i] = torch.zeros(configues["embedding_dim"])
            for client_id,client in clients.items():
                client.get_next_layer_gd()
                #
                client.add_l_th_layer_embedding_gd(layer)
                #
                ord_item_gd_of_one_client=client.send_current_l_th_ord_item_gd()
                client.reset_ord_item_embedding_gd()
                for i in ord_item_gd_of_one_client.keys():
                    item_gd[i] += ord_item_gd_of_one_client[i]
            #
            for sample_client in negative_item_id_of_sample_clients.keys():
                neg_item_gd_of_one_client = clients[sample_client].send_l_th_neg_item_gd(layer)
                for i in neg_item_gd_of_one_client.keys():
                    item_gd[i] += neg_item_gd_of_one_client[i]
            #
            server.item_gd=item_gd
            #
            for id, con_client in convolution_clients.items():
                con_items_id = con_client.convolution_items_id
                con_embedding_gd = {}
                for i in con_items_id:
                    con_embedding_gd[i] = server.item_gd[i]
                con_client.con_embedding_gd = con_embedding_gd
                con_client.update_embedding_gd()


            neighbor_users_embedding_gd = {}
            for i in range(dataloader.user_number):
                neighbor_users_embedding_gd[i] = torch.zeros(configues["embedding_dim"])
            for id, con_client in convolution_clients.items():
                neighbor_users_embedding_gd_of_one_client=con_client.send_current_layer_neighbor_users_gd()
                for i in neighbor_users_embedding_gd_of_one_client.keys():
                    neighbor_users_embedding_gd[i] += neighbor_users_embedding_gd_of_one_client[i]
            #
            server.neighbor_users_embedding_gd=neighbor_users_embedding_gd
            for client_id in neighbor_users_embedding_gd.keys():
                clients[client_id].embedding_gd[0]+=neighbor_users_embedding_gd[client_id]
            # print(1)

        #
        for client in clients.values():
            client.embeddings[0].weight.grad=client.embedding_gd
        opt.step()

        #
        zero_th_item_embeddings = {}
        for convolution_client in convolution_clients.values():
            zero_th_item_embeddings_of_convolution_client = convolution_client.send_zero_th_con_item_embeddings()
            zero_th_item_embeddings = {**zero_th_item_embeddings, **zero_th_item_embeddings_of_convolution_client}
        #
        for client in clients.values():
            zero_th_ord_item_embedding_updated={}
            for i in client.ordinary_items_id:
                zero_th_ord_item_embedding_updated[i]=zero_th_item_embeddings[i]
            client.zero_th_ord_item_embedding_updated=zero_th_ord_item_embedding_updated
            client.update_zero_th_ord_item_embedding()


        for client in clients.values():
            for i in client.embeddings:
                i.weight.grad=None
            client.initial_embedding_gd()
    aver_loss = aver_loss / iter_time
    print("epoch-"+str(epoch+1)+":"+str(aver_loss.item()))
    print("----------------------------------------------")

    with open(train_loss_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch+1, aver_loss.item()])

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












