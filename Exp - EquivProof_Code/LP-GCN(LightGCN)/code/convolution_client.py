import torch
from configue import configues
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as sp







class Convolution_Client():
    def __init__(self,user_id,item_id):
        #
        self.item_id = item_id  #list
        self.user_id = user_id  #int
        self.user_degree = len(self.item_id)
        self.hash_index_to_id_of_item=self.get_hash_index_to_id_of_item()   #dict
        self.hash_id_to_index_of_item=self.get_hash_id_to_index_of_item()   #dict
        self.subgraph_data_structure=None   #{item_id:[neighbor_users_id]}
        self.neighbor_users_id=None  #list
        self.hash_index_to_id_of_neighbor_users = None    #{neighbor_user_index:neighbor_user_id}
        self.hash_id_to_index_of_neighbor_users = None      #{neighbor_user_id:neighbor_user_index}
        self.neighbor_users_degree = None  # dict:{neighbor_user_id:degree}
        self.item_degree = None  # dict:{item_id:degree}
        self.convolution_items_id= None  #list
        self.ordinary_items_id=None     #list
        self.adjacency_matrix=None
        #
        self.user_embedding=torch.empty(1,configues["embedding_dim"])   #tensor
        self.item_embedding=torch.empty(len(self.item_id),configues["embedding_dim"])    #tensor
        self.embeddings=None
        self.neighbor_users_embedding = None  # {neighbor_users_id:embedding(tensor)}
        self.ord_items_embedding=None   # {ord_items_id:embedding(tensor)}
        self.negative_item_id = None  # list
        self.hash_id_to_index_of_negative_item=None  #dict
        self.hash_index_to_id_of_negative_item=None  #dict
        self.negative_item_embedding=None
        self.sample_pairs_for_loss=None     #[tuple]
        self.loss=None
        self.embedding_gd=None
        self.con_embedding_gd=None
        self.zero_th_ord_item_embedding_updated=None    #{zero_th_ord_items_id:embedding(tensor)}
        self.batch_size=None    #float
        self.average_user_embedding=None #for test stage
    def get_hash_index_to_id_of_item(self):
        hash_index_to_id_of_item={}
        for index,value in enumerate(self.item_id):
            hash_index_to_id_of_item[index]=value
        return hash_index_to_id_of_item

    def get_hash_id_to_index_of_item(self):
        hash_id_to_index_of_item={}
        for index,value in enumerate(self.item_id):
            hash_id_to_index_of_item[value]=index
        return hash_id_to_index_of_item

    #
    def get_neighbor_users_id(self):
        neighbor_uses_id=[]
        for i in self.subgraph_data_structure.values():
            neighbor_uses_id.extend(i)
        neighbor_uses_id=list(set(neighbor_uses_id))
        self.neighbor_users_id=neighbor_uses_id
        return None
    #
    def get_hash_index_to_id_of_neighbor_users(self):
        hash_index_to_id_of_neighbor_users={}
        for index,value in enumerate(self.neighbor_users_id):
            hash_index_to_id_of_neighbor_users[index]=value
        self.hash_index_to_id_of_neighbor_users=hash_index_to_id_of_neighbor_users
        return None

    #
    def get_hash_id_to_index_of_neighbor_users(self):
        hash_id_to_index_of_neighbor_users={}
        for index,value in enumerate(self.neighbor_users_id):
            hash_id_to_index_of_neighbor_users[value]=index
        self.hash_id_to_index_of_neighbor_users=hash_id_to_index_of_neighbor_users
        return None

    #
    def get_neighbor_users_degree(self,degree_of_all_users):
        neighbor_users_degree={}
        for i in self.neighbor_users_id:
            neighbor_users_degree[i]=degree_of_all_users[i]
        self.neighbor_users_degree=neighbor_users_degree
        return None

    #
    def get_item_degree(self,degree_of_all_items):
        item_degree={}
        for i in self.item_id:
            item_degree[i]=degree_of_all_items[i]
        self.item_degree=item_degree
        return None

    #
    def get_ordinary_items_id(self):
        # print(self.convolution_items_id)
        con_items_id_set=set(self.convolution_items_id)
        ord_items_id = [item for item in self.item_id if item not in con_items_id_set]
        self.ordinary_items_id=ord_items_id
        return None

    #
    def initialize_all_layers_embeddings(self):
        embedding_mid = []
        row_length = len(self.neighbor_users_id) + len(self.item_id) + 1
        for l in range(configues["layer"]):
            embedding=torch.nn.Embedding(row_length, configues["embedding_dim"])
            embedding_mid.append(embedding)
        self.embeddings=embedding_mid
        #
        self.neighbor_users_embedding=torch.empty(len(self.neighbor_users_id),configues["embedding_dim"])
        embedding_data=torch.cat([self.user_embedding,self.neighbor_users_embedding],dim=0)
        embedding_data=torch.cat([embedding_data,self.item_embedding],dim=0)
        self.embeddings[0].weight.data=embedding_data
        return None

    #
    def get_adjacency_matrix(self):


        user_array=np.ones((1,len(self.item_id)))
        neighbor_users_array=np.zeros((len(self.neighbor_users_id), len(self.item_id)))
        #print(neighbor_users_array.shape)
        for i in self.subgraph_data_structure.keys():
            users_of_i=self.subgraph_data_structure[i]
            i_index=self.hash_id_to_index_of_item[i]
            for u in users_of_i:
                u_index=self.hash_id_to_index_of_neighbor_users[u]
                neighbor_users_array[u_index][i_index]=1.0
        local_ui_interect_matrix=np.concatenate((user_array, neighbor_users_array), axis=0)
        local_user_number=local_ui_interect_matrix.shape[0]
        local_item_number=local_ui_interect_matrix.shape[1]
        local_ui_interect_matrix=csr_matrix(local_ui_interect_matrix)

        #
        adj_matrix = sp.dok_matrix((local_user_number + local_item_number,local_user_number + local_item_number),
                                   dtype=np.float32)
        adj_matrix = adj_matrix.tolil()
        R = local_ui_interect_matrix.tolil()
        adj_matrix[:local_user_number,local_user_number:]=R
        adj_matrix[local_user_number:,:local_user_number]=R.T
        adj_matrix=adj_matrix.todok()

        degree=[float(self.user_degree)]
        for neighbor_u in self.hash_index_to_id_of_neighbor_users.keys():
            id_of_neighbor_user=self.hash_index_to_id_of_neighbor_users[neighbor_u]
            neighbor_u_degree=float(self.neighbor_users_degree[id_of_neighbor_user])
            degree.append(neighbor_u_degree)
        for i in self.hash_index_to_id_of_item.keys():
            id_of_item=self.hash_index_to_id_of_item[i]
            # print(self.user_id)
            i_degree=float(self.item_degree[id_of_item])
            degree.append(i_degree)

        row_sum = np.array(degree)
        d_inv=np.power(row_sum,-0.5).flatten()
        d_mat=sp.diags(d_inv)

        norm_adj = d_mat.dot(adj_matrix)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()

        norm_adj_tensor=self.conver_sp_matrix_to_sp_tensor(norm_adj)
        self.adjacency_matrix=norm_adj_tensor
        return None
    def conver_sp_matrix_to_sp_tensor(self,x):
        coo=x.tocoo().astype(np.float32)
        row=torch.Tensor(coo.row).long()
        col=torch.Tensor(coo.col).long()
        index=torch.stack([row,col])
        data=torch.FloatTensor(coo.data)
        sp_tensor=torch.sparse.FloatTensor(index,data,torch.Size(coo.shape))
        return sp_tensor

    #
    def initial_embedding_gd(self):
        self.embedding_gd=torch.zeros(self.embeddings[0].weight.shape)
        return None

    #
    def send_l_th_user_embedding(self,l):
        return self.embeddings[l].weight.data[0]

    #
    def update_l_th_neighbor_users_embedding(self,l):
        embedding_weight_data=self.embeddings[l].weight.data
        for u in range(len(self.neighbor_users_id)):
            neighbor_user_index=u+1
            neighbor_user_id=self.hash_index_to_id_of_neighbor_users[u]
            embedding_weight_data[neighbor_user_index]=self.neighbor_users_embedding[neighbor_user_id]
        return None


    #
    def GNN(self,l):
        embedding_weight_data=self.embeddings[l].weight.data
        embedding_weight_data_next_layer = torch.sparse.mm(self.adjacency_matrix, embedding_weight_data)
        # print(self.embeddings)
        # print(l)
        self.embeddings[l+1].weight.data=embedding_weight_data_next_layer
        return None

    #
    def send_l_plus_1_th_con_item_embeddings(self,l):
        l_plus_1_th_con_item_embeddings={}
        for i in self.convolution_items_id:
            con_item_index=self.hash_id_to_index_of_item[i]
            con_item_embedding=self.embeddings[l+1].weight.data[len(self.neighbor_users_id)+1+con_item_index]
            l_plus_1_th_con_item_embeddings[i]=con_item_embedding
        return l_plus_1_th_con_item_embeddings

    #
    def update_l_plus_1_ord_item_embedding(self,l):
        embedding_weight_data = self.embeddings[l+1].weight.data
        for i in self.ordinary_items_id:
            ord_item_index=self.hash_id_to_index_of_item[i]
            ord_item_ui_index=ord_item_index+len(self.neighbor_users_id)+1
            embedding_weight_data[ord_item_ui_index]=self.ord_items_embedding[i]
        return None

    #
    def send_zero_th_con_item_embeddings(self):
        zero_th_con_item_embeddings={}
        for i in self.convolution_items_id:
            con_item_index=self.hash_id_to_index_of_item[i]
            con_item_embedding=self.embeddings[0].weight.data[len(self.neighbor_users_id)+1+con_item_index]
            zero_th_con_item_embeddings[i]=con_item_embedding
        return zero_th_con_item_embeddings

    def get_hash_id_to_index_of_negative_item(self):
        hash_id_to_index_of_negative_item={}
        for index,value in enumerate(self.negative_item_id):
            hash_id_to_index_of_negative_item[value]=index
        self.hash_id_to_index_of_negative_item=hash_id_to_index_of_negative_item
        return None

    def get_hash_index_to_id_of_negative_item(self):
        hash_index_to_id_of_negative_item={}
        for index,value in enumerate(self.negative_item_id):
            hash_index_to_id_of_negative_item[index]=value
        self.hash_index_to_id_of_negative_item=hash_index_to_id_of_negative_item
        return None

    #
    def initial_negative_item_embedding(self):
        negative_item_embedding_mid=[]
        for i in range(configues["layer"]):
            negative_item_embedding = torch.nn.Embedding(len(self.negative_item_id), configues["embedding_dim"])
            negative_item_embedding_mid.append(negative_item_embedding)
        self.negative_item_embedding=negative_item_embedding_mid
        return None

    #
    def get_loss(self):
        sum_loss=[]
        for sample_pairs in self.sample_pairs_for_loss:
            pos_item_id=sample_pairs[0]
            neg_item_id=sample_pairs[1]
            pos_item_index=self.hash_id_to_index_of_item[pos_item_id]+len(self.neighbor_users_id)+1
            neg_item_index=self.hash_id_to_index_of_negative_item[neg_item_id]

            #
            final_user_embedding=[]
            for layer in range(configues["layer"]):
                final_user_embedding.append(self.embeddings[layer].weight[0])
            #print(final_user_embedding)
            final_user_embedding=sum(final_user_embedding)/len(self.embeddings)

            final_pos_item_embedding=[]
            for layer in range(configues["layer"]):
                final_pos_item_embedding.append(self.embeddings[layer].weight[pos_item_index])
            final_pos_item_embedding=sum(final_pos_item_embedding)/len(self.embeddings)


            final_neg_item_embedding=[]
            for layer in range(configues["layer"]):
                final_neg_item_embedding.append(self.negative_item_embedding[layer].weight[neg_item_index])
            final_neg_item_embedding=sum(final_neg_item_embedding)/len(self.embeddings)

            pos_scores=torch.mul(final_user_embedding,final_pos_item_embedding)
            pos_scores=torch.sum(pos_scores)

            neg_scores = torch.mul(final_user_embedding, final_neg_item_embedding)
            neg_scores = torch.sum(neg_scores)

            loss=torch.nn.functional.softplus(neg_scores-pos_scores)
            #计算正则化项的loss

            reg_loss=(1/2)*(self.embeddings[0].weight[0].norm(2).pow(2)+
                            self.embeddings[0].weight[pos_item_index].norm(2).pow(2)+
                            self.negative_item_embedding[0].weight[neg_item_index].norm(2).pow(2))

            loss=loss+configues["weight_decay"]*reg_loss
            sum_loss.append(loss)
            # sum_loss.append(reg_loss)
        sum_loss=sum(sum_loss)/self.batch_size

        self.loss = sum_loss
        return sum_loss

    #
    def loss_backword(self):
        self.loss.backward()
        return None

    #
    def initial_embeddings_gd(self):
        self.embedding_gd=torch.zeros(self.embeddings[0].weight.shape)
        return None

    #
    def send_l_th_neg_item_gd(self,l):      #{item_id:gradient}
        neg_item_gd={}
        for index in self.hash_index_to_id_of_negative_item.keys():
            neg_item_id=self.hash_index_to_id_of_negative_item[index]
            neg_item_gd[neg_item_id]=self.negative_item_embedding[l].weight.grad[index]
        return neg_item_gd

    #
    def send_current_l_th_ord_item_gd(self):
        ord_item_gd={}
        for id in self.ordinary_items_id:
            embedding_gd_mid=self.embedding_gd.clone()
            ord_item_index=self.hash_id_to_index_of_item[id]+len(self.neighbor_users_id)+1
            ord_item_gd[id]=embedding_gd_mid[ord_item_index]
        return ord_item_gd


    #
    def get_last_layer_embedding_gd(self):
        self.embedding_gd=self.embeddings[-1].weight.grad
        return None

    #
    def update_embedding_gd(self):
        for id,embedding_gd in self.con_embedding_gd.items():
            con_item_id=id
            con_item_index=self.hash_id_to_index_of_item[con_item_id]+len(self.neighbor_users_id)+1
            self.embedding_gd[con_item_index]+=embedding_gd
        return None


    #
    def reset_ord_item_embedding_gd(self):
        for id in self.ordinary_items_id:
            ord_item_index=self.hash_id_to_index_of_item[id]+len(self.neighbor_users_id)+1
            self.embedding_gd[ord_item_index]=torch.zeros(configues["embedding_dim"])
        return None


    #
    def get_next_layer_gd(self):
        self.embedding_gd=torch.sparse.mm(self.adjacency_matrix, self.embedding_gd)
        return None

    #
    def add_l_th_layer_embedding_gd(self,l):
        if self.embeddings[l].weight.grad==None:
            pass
        else:
            self.embedding_gd+=self.embeddings[l].weight.grad
        return None

    #
    def send_current_layer_neighbor_users_gd(self):
        neighbor_users_gd = {}
        for index in range(len(self.neighbor_users_id)):
            embedding_gd=self.embedding_gd.clone()
            neighbor_user_id=self.hash_index_to_id_of_neighbor_users[index]
            neighbor_users_gd[neighbor_user_id]=embedding_gd[index+1]
            self.embedding_gd[index+1]=torch.zeros(configues["embedding_dim"])
        return neighbor_users_gd
    #
    def update_zero_th_ord_item_embedding(self):
        embedding_weight_data=self.embeddings[0].weight.data
        for i in self.ordinary_items_id:
            ord_item_index=self.hash_id_to_index_of_item[i]
            ord_item_ui_index=ord_item_index+len(self.neighbor_users_id)+1
            embedding_weight_data[ord_item_ui_index]=self.zero_th_ord_item_embedding_updated[i]
        return None
    def get_average_user_embedding(self):
        user_embedding_of_all_layer=[]
        for embedding in self.embeddings:
            user_embedding_of_all_layer.append(embedding.weight[0])
        stacked_tensors = torch.stack(user_embedding_of_all_layer)
        self.average_user_embedding=torch.mean(stacked_tensors, dim=0)
        return None