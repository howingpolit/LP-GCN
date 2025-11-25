import torch
import numpy as np
import math
from torch import nn
class Evalution():
    def __init__(self,user_embedding_matrix,item_embedding_matrix,dataloader,configues):
        self.user_embedding_matrix=user_embedding_matrix
        self.item_embedding_matrix=item_embedding_matrix
        self.dataloader=dataloader
        self.configues=configues
        self.k=self.configues["topk"]
        self.f=nn.Sigmoid()
    def get_top_k(self):
        '''
        :return: top_k items of all users
        '''
        top_k={}
        for user_id in range(self.dataloader.user_number):
            rec_items=[]
            user_id_longtensor=torch.LongTensor([user_id])
            rating=self.f(torch.matmul(self.user_embedding_matrix[user_id_longtensor], self.item_embedding_matrix.t()))
            sort_index=torch.topk(rating,self.dataloader.item_number)
            rating_index=np.array(sort_index[1]).squeeze()
            pos_item=self.dataloader.all_pos_items[user_id]
            for item in rating_index:
                if item in pos_item:
                    continue
                else:
                    rec_items.append(item)
                if len(rec_items)==self.k:
                    break
            top_k[user_id]=rec_items
        return top_k

    # -----------------------------
    def get_precision_of_all_u(self):
        haxi_of_u_to_i_test=self.dataloader.hash_u_to_i_test
        topk_of_rec_list_of_all_u=self.get_top_k()
        precision_of_all_u_sum = 0
        haxi_of_u_to_i_test_keys = haxi_of_u_to_i_test.keys()
        for u_test in haxi_of_u_to_i_test_keys:
            # print(topk_of_rec_list_of_all_u)
            # print(u_test)
            topk_of_rec_list_of_u = topk_of_rec_list_of_all_u[u_test]
            true_list = haxi_of_u_to_i_test[u_test]
            common_item = list(set(topk_of_rec_list_of_u) & set(true_list))
            precision_of_all_u_sum += len(common_item) / self.k
        precision_of_all_u = precision_of_all_u_sum / len(haxi_of_u_to_i_test.keys())
        return precision_of_all_u

    # -----------------------------
    def get_recall_of_all_u(self):
        topk_of_rec_list_of_all_u = self.get_top_k()
        haxi_of_u_to_i_test=self.dataloader.hash_u_to_i_test
        recall_of_all_u_sum = 0
        haxi_of_u_to_i_test_keys = haxi_of_u_to_i_test.keys()
        for u_test in haxi_of_u_to_i_test_keys:
            # print(topk_of_rec_list_of_all_u)
            # print(u_test)
            topk_of_rec_list_of_u = topk_of_rec_list_of_all_u[u_test]
            true_list = haxi_of_u_to_i_test[u_test]
            common_item = list(set(topk_of_rec_list_of_u) & set(true_list))
            recall_of_all_u_sum += len(common_item) / len(true_list)
        recall_of_all_u = recall_of_all_u_sum / len(haxi_of_u_to_i_test.keys())
        return recall_of_all_u

    # -----------------------------
    def get_f1_of_all_u(self):
        topk_of_rec_list_of_all_u = self.get_top_k()
        haxi_of_u_to_i_test=self.dataloader.hash_u_to_i_test
        f1_of_all_u_sum = 0
        haxi_of_u_to_i_test_keys = haxi_of_u_to_i_test.keys()
        for u_test in haxi_of_u_to_i_test_keys:
            topk_of_rec_list_of_u = topk_of_rec_list_of_all_u[u_test]
            true_list = haxi_of_u_to_i_test[u_test]
            common_item = list(set(topk_of_rec_list_of_u) & set(true_list))
            pre = len(common_item) / self.k
            recall = len(common_item) / len(true_list)
            if pre == 0:
                continue
            else:
                f1_of_all_u_sum += (2 * pre * recall) / (pre + recall)
        f1_of_all_u = f1_of_all_u_sum / len(haxi_of_u_to_i_test.keys())
        return f1_of_all_u

    #
    def get_one_call(self):
        topk_of_rec_list_of_all_u = self.get_top_k()
        haxi_of_u_to_i_test=self.dataloader.hash_u_to_i_test
        one_call_sum = 0
        haxi_of_u_to_i_test_keys = haxi_of_u_to_i_test.keys()
        for u_test in haxi_of_u_to_i_test_keys:
            topk_of_rec_list_of_u = topk_of_rec_list_of_all_u[u_test]
            true_list = haxi_of_u_to_i_test[u_test]
            for i in topk_of_rec_list_of_u:
                if i in true_list:
                    one_call_sum += 1
                    break
        one_call = one_call_sum / len(haxi_of_u_to_i_test.keys())
        return one_call

    #
    def get_NDCG_of_all_u(self,a):
        topk_of_rec_list_of_all_u = self.get_top_k()
        haxi_of_u_to_i_test=self.dataloader.hash_u_to_i_test
        NDCG_of_all_u_sum = 0
        haxi_of_u_to_i_test_keys = haxi_of_u_to_i_test.keys()
        #
        Zu = 0
        Zu_list = [0] * a
        for i in range(a):
            Zu += 1 / math.log(i + 2, 2)
            Zu_list[i] = Zu
        for u_test in haxi_of_u_to_i_test_keys:
            topk_of_rec_list_of_u = topk_of_rec_list_of_all_u[u_test]
            true_list = haxi_of_u_to_i_test[u_test]
            k = len(true_list)
            if k > a:
                k = a
            #
            Zu = Zu_list[k - 1]
            DCG_sum = 0
            for index, value in enumerate(topk_of_rec_list_of_u):
                if value in true_list:
                    flag = 1
                else:
                    flag = 0
                DCG_sum += flag / math.log(index + 2, 2)
            DCG_of_u = DCG_sum / Zu
            NDCG_of_all_u_sum += DCG_of_u
        NDCG_of_all_u = NDCG_of_all_u_sum / len(haxi_of_u_to_i_test_keys)
        return NDCG_of_all_u


