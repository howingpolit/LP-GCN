import scipy.sparse as sp
import numpy as np
import torch
from configue import configues
import random
from client_classify import find_minimum_user_set_sparse
class Server():
    def __init__(self,dataloader):
        self.dataloader=dataloader
        self.convolution_clients_id_set=None    #set
        self.ordinary_clients_id_set=None    #set
        self.item_id_to_convolution_client_id=None  #{item_id:convolution_client_id}
        self.convolution_items_of_convolution_clients=None  #{convolution_client_id:[item_ids]}
        self.subgraph_data_structure_of_convolution_clients=None    #{convolution_client:{common_item:[neighbor_users]}}
        self.degree_of_all_users=None   #{user_id:user_degree}
        self.l_plus_1_item_embeddings=None  #{item_id:l+1_item_embbeding}
        self.all_item_embedding_of_all_layer={}   #{layer:{item_id:embedding}}
        self.negtive_item_id_of_sample_clients=None #{client_id:[negative_item_id]}
        self.ord_and_neg_embedding_gd_of_last_layer=None    #{item_id:embedding_gd}
        self.item_gd=None   #{item_id:embedding_gd}
        self.neighbor_users_embedding_gd=None     #{client_id:embedding_gd}

    #
    def split_clients(self):
        convolution_clients_id_set = find_minimum_user_set_sparse(self.dataloader.train_user_item_matrix)
        ordinary_clients_id_set = set(range(self.dataloader.user_number)) - convolution_clients_id_set
        self.convolution_clients_id_set=convolution_clients_id_set
        self.ordinary_clients_id_set=ordinary_clients_id_set
        return None


    #
    def mark_convolution_items(self):
        random.seed(configues["seed"])
        item_id_to_convolution_client_id={}
        for i in range(self.dataloader.item_number):
            convolution_client_id=random.choice(list(set(self.dataloader.hash_i_to_u_train[i]).intersection(self.convolution_clients_id_set)))
            item_id_to_convolution_client_id[i]=convolution_client_id
        self.item_id_to_convolution_client_id=item_id_to_convolution_client_id
        return None

    #
    def get_convolution_items_of_convolution_clients(self):
        inverted_dict = {}
        for key, value in self.item_id_to_convolution_client_id.items():
            #
            if value in inverted_dict:
                inverted_dict[value].append(key)
            else:
                inverted_dict[value] = [key]
        for i in self.convolution_clients_id_set:
            if i not in inverted_dict:
                inverted_dict[i] = []
        self.convolution_items_of_convolution_clients=inverted_dict
        return None

    #
    def get_subgraph_data_structure_of_convolution_clients(self):
        subgraph_data_structure_of_convolution_clients={}
        for con_client in self.convolution_clients_id_set:
            item = list(self.dataloader.train_user_item_matrix[con_client].nonzero()[1])
            dict_mid={}
            for j in item:
                colj = self.dataloader.train_user_item_matrix[:, j]
                neighbor_users = list(colj.nonzero()[0])
                neighbor_users.remove(con_client)
                dict_mid[j]=neighbor_users
            subgraph_data_structure_of_convolution_clients[con_client]=dict_mid
        self.subgraph_data_structure_of_convolution_clients=subgraph_data_structure_of_convolution_clients
        return None

    #
    def get_degree_of_all_users(self):
        degree_of_all_users={}
        for key,value in self.dataloader.hash_u_to_i_train.items():
            degree_of_all_users[key]=len(value)
        self.degree_of_all_users=degree_of_all_users
        return None
    #
    def get_degree_of_all_items(self):
        degree_of_all_items = {}
        for key,value in self.dataloader.hash_i_to_u_train.items():
            degree_of_all_items[key]=len(value)
        self.degree_of_all_items=degree_of_all_items
        return None


    def sample_pairs(self,epoch):
        sample_pairs = []
        np.random.seed(epoch)
        users_sample_id = np.random.randint(0, self.dataloader.user_number, 1 * self.dataloader.train_data_num)
        for index, user_id in enumerate(users_sample_id):
            pos_item_of_user = self.dataloader.all_pos_items[user_id]
            if len(pos_item_of_user) == 0:
                continue
            pos_index = np.random.randint(0, len(pos_item_of_user))
            pos_item_id = pos_item_of_user[pos_index]
            while True:
                neg_item_id = np.random.randint(0, self.dataloader.item_number)
                if neg_item_id in pos_item_of_user:
                    continue
                else:
                    break
            sample_pairs.append([user_id, pos_item_id, neg_item_id])
        sample_pairs = np.array(sample_pairs)
        # self.sample_pairs=sample_pairs
        return sample_pairs

    #
    def get_negative_item_id_of_sample_clients(self,batch_users,batch_neg_items):
        users = batch_users
        negative_item_id_of_sample_clients={}
        users=list(set(users))
        for u in users:
            negative_item_id_of_sample_clients[u]=[]
        for index,value in enumerate(batch_users):
            negative_item_id_of_sample_clients[value].append(batch_neg_items[index])
        self.negtive_item_id_of_sample_clients=negative_item_id_of_sample_clients
        return negative_item_id_of_sample_clients

    #
    def get_sample_pairs_for_loss(self,batch_users,batch_pos_items,batch_neg_items):
        users = batch_users
        sample_pairs_for_loss_of_sample_clients={}
        users=list(set(users))
        for u in users:
            sample_pairs_for_loss_of_sample_clients[u]=[]
        for index,value in enumerate(batch_users):
            sample_pairs_for_loss_of_sample_clients[value].append((batch_pos_items[index],batch_neg_items[index]))
        return sample_pairs_for_loss_of_sample_clients

#----------------------------------------------------------------------
    def get_info_for_expansion(self):
        """
        get user embedding,item_id and degree of user from all clients
        :return:
        """
        info_for_expansion=[]
        for u in self.clients:
            info_for_expansion_of_u=[u.user_degree,u.item_id]
            info_for_expansion.append(info_for_expansion_of_u)
        return info_for_expansion



    def get_neighbors_users_id_of_all_clients(self,neighbor_users_of_all_clients):
        """
        :param neighbor_users_of_all_clients:
        :return: {client_id:[neighbor_users_id]}
        """
        neighbors_users_id_of_all_clients={}
        for u in range(self.dataloader.user_number):
            neighbors_users_id_of_one_client=[]
            for i in neighbor_users_of_all_clients[u].values():
                neighbors_users_id_of_one_client.extend(i)
            neighbors_users_id_of_one_client=list(set(neighbors_users_id_of_one_client))
            neighbors_users_id_of_all_clients[u]=neighbors_users_id_of_one_client
        self.neighbors_users_id_of_all_clients=neighbors_users_id_of_all_clients
        return neighbors_users_id_of_all_clients


    #
    def get_k_th_users_embedding(self,k):
        k_th_users_embedding={}
        for client in self.clients:
            k_th_users_embedding[client.user_id]=client.send_k_th_user_embedding(k)
        return k_th_users_embedding

    #
    def distribute_k_th_users_embedding(self,k):
        k_th_users_embedding=self.get_k_th_users_embedding(k)
        k_th_users_embedding_of_all_clients={}
        for client in self.neighbors_users_id_of_all_clients.keys():
            neighbor_users_id_of_one_client=self.neighbors_users_id_of_all_clients[client]
            k_th_users_embedding_of_one_client={}
            for u in neighbor_users_id_of_one_client:
                k_th_users_embedding_of_one_client[u]=k_th_users_embedding[u]
            k_th_users_embedding_of_all_clients[client]=k_th_users_embedding_of_one_client
        return k_th_users_embedding_of_all_clients



    #
    def get_all_item_embedding(self,batch_neg_items):
        all_item_embedding={}
        for i in batch_neg_items:
            user_id=self.dataloader.hash_i_to_u_train[i][0]
            item_embedding=self.clients[user_id].send_item_all_layer_embedding(i)
            all_item_embedding[i]=item_embedding
        return all_item_embedding





    #
    def get_negative_item_gd(self,batch_neg_items,layer):
        #
        negative_item_gd={}
        for i in batch_neg_items:
            negative_item_gd[i]=torch.zeros(configues["embedding_dim"])

        #
        for client_id in self.negtive_item_id_of_sample_clients.keys():
            neg_item_gd_of_one_client=self.clients[client_id].send_k_th_neg_item_gd(layer)
            for i in neg_item_gd_of_one_client.keys():
                negative_item_gd[i]+=neg_item_gd_of_one_client[i]


        #
        negative_item_gd_of_clients={}
        for i in negative_item_gd.keys():
            user_id = self.dataloader.hash_i_to_u_train[i][0]
            negative_item_gd_of_clients[user_id]=[]
        for i in negative_item_gd.keys():
            negative_item_gd_of_i={}
            negative_item_gd_of_i[i]=negative_item_gd[i]
            user_id = self.dataloader.hash_i_to_u_train[i][0]
            negative_item_gd_of_clients[user_id].append(negative_item_gd_of_i)
        return negative_item_gd_of_clients


    #
    def get_k_th_neighbor_users_embedding_gd(self):
        #
        neighbor_users_embedding_gd={}
        for i in range(len(self.clients)):
            neighbor_users_embedding_gd[i]=torch.zeros(configues["embedding_dim"])

        #
        for client in self.clients:
            neighbor_users_embedding_gd_of_one_client=client.send_k_th_neighbor_users_gd()
            for i in neighbor_users_embedding_gd_of_one_client.keys():
                neighbor_users_embedding_gd[i] += neighbor_users_embedding_gd_of_one_client[i]

        return neighbor_users_embedding_gd

    #
    def transmit_item_embedding_gd(self):
        item_embedding_gd_of_all_clients={}
        for u in range(len(self.clients)):
            item_embedding_gd_of_all_clients[u]=[]
        for index,client in enumerate(self.clients):
            random_item_gd_of_one_client=client.split_the_item_gradient()
            for i in random_item_gd_of_one_client.keys():
                one_item_embedding_gd={i:random_item_gd_of_one_client[i]}
                users_id=self.dataloader.hash_i_to_u_train[i].copy()
                users_id.remove(index)
                user_id_chosen=random.choice(users_id)
                item_embedding_gd_of_all_clients[user_id_chosen].append(one_item_embedding_gd)
        return item_embedding_gd_of_all_clients     #{user_id:[{item_id:item_embedding_gd]}



    #
    def aggregate_item_embedding_gd(self):
        #
        item_gd={}
        for i in range(self.dataloader.item_number):
            item_gd[i]=torch.zeros(configues["embedding_dim"])

        #
        for client in self.clients:
            item_gd_of_one_client=client.send_item_gd_for_aggregation()
            for i in item_gd_of_one_client.keys():
                item_gd[i]+=item_gd_of_one_client[i]
        return item_gd

    #----------------------------------------------------------------------------------------------
    #for test
    def get_all_item_embedding_for_test(self):
        all_item_embedding_for_test={}
        for i in range(self.dataloader.item_number):
            user_id=self.dataloader.hash_i_to_u_train[i][0]
            item_embedding=self.clients[user_id].send_item_aggregated_embedding(i)
            all_item_embedding_for_test[i]=item_embedding
        return all_item_embedding_for_test


    def get_precision(self):
        precision_mid=0.0
        recall_mid=0.0
        for u in self.dataloader.hash_u_to_i_test.keys():
            truth_item=self.dataloader.hash_u_to_i_test[u]

            rec_item=self.clients[u].topk_item
            common_item=list(set(rec_item) & set(truth_item))
            precision_mid+=len(common_item)/configues["topk"]
            recall_mid+=len(common_item)/len(truth_item)
        precision=precision_mid/len(self.dataloader.hash_u_to_i_test.keys())
        recall=recall_mid/len(self.dataloader.hash_u_to_i_test.keys())
        return precision,recall
