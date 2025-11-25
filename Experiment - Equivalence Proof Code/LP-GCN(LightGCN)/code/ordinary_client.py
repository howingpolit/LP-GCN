import torch
from configue import configues
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as sp


class Ordinary_Client():
    def __init__(self,user_id,item_id):
        #
        self.item_id = item_id  #list
        self.user_id = user_id  #int
        self.ordinary_items_id=item_id  #list
        self.user_degree = len(self.item_id)
        self.hash_index_to_id_of_item=self.get_hash_index_to_id_of_item()   #dict
        self.hash_id_to_index_of_item=self.get_hash_id_to_index_of_item()   #dict
        self.item_degree = None  # dict:{item_id:degree}
        self.adjacency_matrix = None


        #
        self.user_embedding=torch.empty(1,configues["embedding_dim"])   #tensor
        self.item_embedding=torch.empty(len(self.item_id),configues["embedding_dim"])    #tensor
        self.embeddings=None
        self.ord_items_embedding = None  # {ord_items_id:embedding(tensor)}
        self.negative_item_id = None  # list
        self.hash_id_to_index_of_negative_item=None  #dict
        self.hash_index_to_id_of_negative_item=None  #dict
        self.negative_item_embedding=None
        self.sample_pairs_for_loss=None     #[tuple]
        self.loss=None
        self.embedding_gd=None
        self.zero_th_ord_item_embedding_updated=None    #{zero_th_ord_items_id:embedding(tensor)}
        self.batch_size = None  # float
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
    def get_item_degree(self,degree_of_all_items):
        item_degree={}
        for i in self.item_id:
            item_degree[i]=degree_of_all_items[i]
        self.item_degree=item_degree
        return None

    #
    def get_adjacency_matrix(self):


        user_array=np.ones((1,len(self.item_id)))

        local_ui_interect_matrix=user_array
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

        for i in self.hash_index_to_id_of_item.keys():
            id_of_item=self.hash_index_to_id_of_item[i]
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
    def initialize_all_layers_embeddings(self):
        embedding_mid=[]
        row_length = len(self.item_id) + 1
        for l in range(configues["layer"]):
            embedding=torch.nn.Embedding(row_length, configues["embedding_dim"])
            embedding_mid.append(embedding)
        self.embeddings=embedding_mid
        #
        embedding_data=torch.cat([self.user_embedding,self.item_embedding],dim=0)
        self.embeddings[0].weight.data=embedding_data
        return None

    #
    def initial_embedding_gd(self):
        self.embedding_gd=torch.zeros(self.embeddings[0].weight.shape)
        return None

    #
    def send_l_th_user_embedding(self,l):
        return self.embeddings[l].weight.data[0]

    #
    def GNN(self,l):
        embedding_weight_data=self.embeddings[l].weight.data
        embedding_weight_data_next_layer = torch.sparse.mm(self.adjacency_matrix, embedding_weight_data)
        self.embeddings[l+1].weight.data=embedding_weight_data_next_layer
        return None
    #
    def update_l_plus_1_ord_item_embedding(self,l):
        embedding_weight_data = self.embeddings[l+1].weight.data
        for i in self.item_id:
            ord_item_index=self.hash_id_to_index_of_item[i]
            ord_item_ui_index=ord_item_index+1
            embedding_weight_data[ord_item_ui_index]=self.ord_items_embedding[i]
        return None

    def get_hash_id_to_index_of_negative_item(self):
        hash_id_to_index_of_negative_item = {}
        for index, value in enumerate(self.negative_item_id):
            hash_id_to_index_of_negative_item[value] = index
        self.hash_id_to_index_of_negative_item = hash_id_to_index_of_negative_item
        return None

    def get_hash_index_to_id_of_negative_item(self):
        hash_index_to_id_of_negative_item = {}
        for index, value in enumerate(self.negative_item_id):
            hash_index_to_id_of_negative_item[index] = value
        self.hash_index_to_id_of_negative_item = hash_index_to_id_of_negative_item
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
    def send_l_th_neg_item_gd(self,l):      #{item_id:gradient}
        neg_item_gd={}
        for index in self.hash_index_to_id_of_negative_item.keys():
            neg_item_id=self.hash_index_to_id_of_negative_item[index]
            neg_item_gd[neg_item_id]=self.negative_item_embedding[l].weight.grad[index]
        return neg_item_gd

    #
    def get_loss(self):
        sum_loss=[]
        for sample_pairs in self.sample_pairs_for_loss:
            pos_item_id=sample_pairs[0]
            neg_item_id=sample_pairs[1]
            pos_item_index=self.hash_id_to_index_of_item[pos_item_id]+1
            neg_item_index=self.hash_id_to_index_of_negative_item[neg_item_id]

            #
            final_user_embedding=[]
            for layer in range(configues["layer"]):
                final_user_embedding.append(self.embeddings[layer].weight[0])

            final_user_embedding=sum(final_user_embedding)/len(self.embeddings)

            final_pos_item_embedding=[]
            for layer in range(configues["layer"]):
                final_pos_item_embedding.append(self.embeddings[layer].weight[pos_item_index])
            final_pos_item_embedding=sum(final_pos_item_embedding)/len(self.embeddings)

            final_neg_item_embedding = []
            for layer in range(configues["layer"]):
                final_neg_item_embedding.append(self.negative_item_embedding[layer].weight[neg_item_index])
            final_neg_item_embedding=sum(final_neg_item_embedding)/len(self.embeddings)

            pos_scores=torch.mul(final_user_embedding,final_pos_item_embedding)
            pos_scores=torch.sum(pos_scores)

            neg_scores = torch.mul(final_user_embedding, final_neg_item_embedding)
            neg_scores = torch.sum(neg_scores)

            loss=torch.nn.functional.softplus(neg_scores-pos_scores)/self.batch_size
            #
            # print(self.embedding[0].weight[0].data[0].item())
            # print(self.embedding[0].weight[pos_item_index].data[0].item())
            # print(self.negative_item_embedding[0].weight[neg_item_index].data[0].item())
            reg_loss=(1/2)*(self.embeddings[0].weight[0].norm(2).pow(2)+
                            self.embeddings[0].weight[pos_item_index].norm(2).pow(2)+
                            self.negative_item_embedding[0].weight[neg_item_index].norm(2).pow(2))/self.batch_size
            #
            loss=loss+configues["weight_decay"]*reg_loss
            sum_loss.append(loss)
            # sum_loss.append(reg_loss)
        sum_loss=sum(sum_loss)

        self.loss = sum_loss
        return sum_loss

    #
    def loss_backword(self):
        self.loss.backward()
        return None


    #
    def get_last_layer_embedding_gd(self):
        self.embedding_gd=self.embeddings[-1].weight.grad
        return None

    #
    def send_current_l_th_ord_item_gd(self):
        ord_item_gd={}
        for i in self.ordinary_items_id:
            embedding_gd_mid=self.embedding_gd.clone()  #
            ord_item_index=self.hash_id_to_index_of_item[i]+1
            ord_item_gd[i]=embedding_gd_mid[ord_item_index]
        return ord_item_gd

    #
    def reset_ord_item_embedding_gd(self):
        for id in self.ordinary_items_id:
            ord_item_index=self.hash_id_to_index_of_item[id]+1
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
    def update_zero_th_ord_item_embedding(self):
        embedding_weight_data=self.embeddings[0].weight.data
        for i in self.ordinary_items_id:
            ord_item_index=self.hash_id_to_index_of_item[i]
            ord_item_ui_index=ord_item_index+1
            embedding_weight_data[ord_item_ui_index]=self.zero_th_ord_item_embedding_updated[i]
        return None
    #
    def get_average_user_embedding(self):
        user_embedding_of_all_layer=[]
        for embedding in self.embeddings:
            user_embedding_of_all_layer.append(embedding.weight[0])
        stacked_tensors = torch.stack(user_embedding_of_all_layer)
        self.average_user_embedding=torch.mean(stacked_tensors, dim=0)
        return None