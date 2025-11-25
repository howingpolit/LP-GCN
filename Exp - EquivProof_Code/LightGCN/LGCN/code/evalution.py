import torch
import numpy as np
import math
class Evalution():
    def __init__(self,model,dataloader,configues):
        self.model=model
        self.dataloader=dataloader
        self.configues=configues
        self.k=self.configues["topk"]
    def get_top_k(self):
        '''

        :return: top_k items of all users
        '''
        top_k={}
        for user_id in range(self.dataloader.user_number):
            rec_items=[]
            user_id_longtensor=torch.LongTensor([user_id])
            rating=self.model.get_one_user_rating(user_id_longtensor).cpu()
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

    def get_sorted_rating(self):
        '''
        :return: sorted items of all users according rating
        dict{user_id:sorted items}
        '''
        sorted_rating={}
        for user_id in range(self.dataloader.user_number):
            rec_items=[]
            user_id_longtensor=torch.LongTensor([user_id])
            rating=self.model.get_one_user_rating(user_id_longtensor)
            sort_index=torch.topk(rating,self.dataloader.item_number)
            rating_index=np.array(sort_index[1]).squeeze()
            sorted_rating[user_id]=list(rating_index)
        return sorted_rating

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

    # ----------------------------
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

    # -----------------------------
    def get_MRR(self):
        haxi_rec_list_of_all_u=self.get_sorted_rating()
        haxi_of_u_to_i_test=self.dataloader.hash_u_to_i_test
        MRR = 0
        haxi_of_u_to_i_test_keys = haxi_of_u_to_i_test.keys()
        for u_test in haxi_of_u_to_i_test_keys:
            rec_list_of_u = haxi_rec_list_of_all_u[u_test]
            true_list = haxi_of_u_to_i_test[u_test]
            for index, value in enumerate(rec_list_of_u):
                if value in true_list:
                    MRR += 1 / (index + 1)
                    break
        result = MRR / len(haxi_of_u_to_i_test_keys)
        return result

    # -----------------------------
    def get_MAP(self):
        haxi_rec_list_of_all_u = self.get_sorted_rating()
        haxi_of_u_to_i_test=self.dataloader.hash_u_to_i_test
        MAP_sum = 0
        haxi_of_u_to_i_test_keys = haxi_of_u_to_i_test.keys()
        for u_test in haxi_of_u_to_i_test_keys:
            rec_list_of_u = haxi_rec_list_of_all_u[u_test]
            true_list = haxi_of_u_to_i_test[u_test]
            APU_sum = 0
            common_item = list(set(rec_list_of_u) & set(true_list))
            for i in common_item:
                pui = rec_list_of_u.index(i) + 1
                slide_of_re = rec_list_of_u[:pui - 1]
                mid = list(set(slide_of_re) & set(common_item))
                APU_sum += (len(mid) + 1) / pui
            APU = APU_sum / len(true_list)
            MAP_sum += APU
        MAP = MAP_sum / len(haxi_of_u_to_i_test_keys)
        return MAP

    # -----------------------------
    def get_ARP(self):
        haxi_rec_list_of_all_u = self.get_sorted_rating()
        haxi_of_u_to_i_test=self.dataloader.hash_u_to_i_test
        haxi_of_u_to_i_train=self.dataloader.hash_u_to_i_train
        haxi_of_i_to_u_test=self.dataloader.hash_i_to_u_test
        haxi_of_i_to_u_train=self.dataloader.hash_i_to_u_train
        ARP_sum = 0
        item_of_train_data = haxi_of_i_to_u_train.keys()
        item_of_test_data = haxi_of_i_to_u_test.keys()
        J_list = list(set(item_of_train_data).union(set(item_of_test_data)))
        J = len(J_list)
        # print(J)
        haxi_of_u_to_i_test_keys = haxi_of_u_to_i_test.keys()
        for u_test in haxi_of_u_to_i_test_keys:
            rec_list_of_u = haxi_rec_list_of_all_u[u_test]
            true_list = haxi_of_u_to_i_test[u_test]
            train_list = haxi_of_u_to_i_train[u_test]
            Ju = len(train_list)
            # 计算分母部分
            denominator = J - Ju
            RPU_sum = 0
            for i in true_list:
                pui = rec_list_of_u.index(i) + 1
                RPU_sum += pui / denominator
            RPU = RPU_sum / len(true_list)
            ARP_sum += RPU
        ARP = ARP_sum / len(haxi_of_u_to_i_test_keys)
        return ARP

    # -----------------------------
    def get_AUC(self):
        n=self.dataloader.user_number
        m=self.dataloader.item_number
        final_users_embeddings, final_items_embeddings = self.model.forward()
        haxi_of_u_to_i_test=self.dataloader.hash_u_to_i_test
        haxi_of_u_to_i_train=self.dataloader.hash_u_to_i_train
        haxi_of_i_to_u_test=self.dataloader.hash_i_to_u_test
        haxi_of_i_to_u_train=self.dataloader.hash_i_to_u_train
        f=self.model.f
        AUC_sum = 0
        # all_item=list(range(0,m))
        haxi_of_u_to_i_test_keys = haxi_of_u_to_i_test.keys()
        haxi_of_i_to_u_train_keys = haxi_of_i_to_u_train.keys()
        haxi_of_i_to_u_test_keys = haxi_of_i_to_u_test.keys()
        all_item = list(set(haxi_of_i_to_u_train_keys).union(set(haxi_of_i_to_u_test_keys)))
        for u_test in haxi_of_u_to_i_test_keys:
            true_list = haxi_of_u_to_i_test[u_test]
            train_list = haxi_of_u_to_i_train[u_test]
            #
            negative_item = list(set(all_item) ^ set(true_list) ^ set(train_list))
            counter = 0
            for i in true_list:
                for j in negative_item:
                    i=torch.LongTensor([i])
                    j=torch.LongTensor([j])
                    u=torch.LongTensor([u_test])
                    i_embedding=final_items_embeddings[i]
                    j_embedding=final_items_embeddings[j]
                    u_embedding=final_users_embeddings[u]

                    predict_of_ui=f(torch.matmul(u_embedding,i_embedding.t())).item()
                    predict_of_uj=f(torch.matmul(u_embedding,j_embedding.t())).item()
                    if predict_of_ui > predict_of_uj:
                        counter += 1
            AUCu = counter / (len(true_list) * len(negative_item))
            AUC_sum += AUCu
        AUC = AUC_sum / len(haxi_of_u_to_i_test_keys)
        return AUC

