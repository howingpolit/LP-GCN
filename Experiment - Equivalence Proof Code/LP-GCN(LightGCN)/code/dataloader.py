from os.path import join

import torch
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset
from configue import configues
import numpy as np
import scipy.sparse as sp


class Dataloader(Dataset):
    def __init__(self,configues=configues):
        DATA_PATH=configues["DATA_PATH"]
        train_file = join(DATA_PATH, "ml-100k")
        self.train_file = join(train_file, "u1.base")

        test_file = join(DATA_PATH, "ml-100k")
        self.test_file = join(test_file, "u1.test")

        with open(self.train_file) as f:
            train_org_matrix_array = []
            train_user_index = []
            train_item_index = []
            for l in f.readlines():
                l = l.strip().split("\t")
                l = [int(i) for i in l[:-1]]
                train_org_matrix_array.append(l)
            train_org_matrix_np = np.array(train_org_matrix_array)
            #print(train_org_matrix_np.shape)

            mid_array = np.array([1, 1, 0])
            #print(mid_array.shape)
            train_org_matrix_np = train_org_matrix_np - mid_array
            #print(train_org_matrix_np)

            for data in train_org_matrix_np:
                if data[2] >= 4:
                    train_user_index.append(data[0])
                    train_item_index.append(data[1])

        with open(self.test_file) as f:
            test_org_matrix_array=[]
            test_user_index=[]
            test_item_index=[]
            for l in f.readlines():
                l=l.strip().split("\t")
                l=[int(i) for i in l[:-1]]
                test_org_matrix_array.append(l)
            test_org_matrix_np=np.array(test_org_matrix_array)
            #print(test_org_matrix_np)

            mid_array=np.array([1,1,0])
            test_org_matrix_np=test_org_matrix_np-mid_array
            #print(test_org_matrix_np)

            for data in test_org_matrix_np:
                if data[2]>=4:
                    test_user_index.append(data[0])
                    test_item_index.append(data[1])




        train_cold_item=[]
        for i in range(max(train_item_index)+1):
            if i not in train_item_index:
                train_cold_item.append(i)


        train_item_index_np=np.array(train_item_index)
        test_item_index_np=np.array(test_item_index)

        for count,i in enumerate(train_cold_item):
            train_item_index_np[train_item_index_np>(i-count)]-=1
            test_item_index_np[test_item_index_np>(i-count)]-=1


        # print(train_item_index_np)
        # print(test_item_index_np)


        train_cold_user=[]
        for i in range(max(train_user_index)+1):
            if i not in train_user_index:
                train_cold_user.append(i)

        train_user_index_np=np.array(train_user_index)
        test_user_index_np=np.array(test_user_index)

        for count,i in enumerate(train_cold_user):
            train_user_index_np[train_user_index_np>(i-count)]-=1
            test_user_index_np[test_user_index_np>(i-count)]-=1

        #print(train_user_index_np)
        #print(test_user_index_np)

        self.user_number=max(max(train_user_index_np).item(),max(test_user_index_np).item())+1
        self.item_number=max(max(train_item_index_np).item(),max(test_item_index_np).item())+1
        print("item_number:",self.item_number)
        print("user_number:",self.user_number)

        #train_sparse_matrix
        self.train_user_item_matrix = csr_matrix(
            (np.ones(len(train_user_index_np)), (train_user_index_np, train_item_index_np)),
            shape=(self.user_number, self.item_number))

        self.train_items_of_user = np.array(self.train_user_item_matrix.sum(axis=1)).squeeze()
        self.train_users_of_item = np.array(self.train_user_item_matrix.sum(axis=0)).squeeze()

        #test_sparse_matrix
        self.test_user_item_matrix=csr_matrix((np.ones(len(test_user_index_np)),(test_user_index_np,test_item_index_np)),
                                              shape=(self.user_number,self.item_number))
        self.test_items_of_user = np.array(self.test_user_item_matrix.sum(axis=1)).squeeze()
        self.test_users_of_item = np.array(self.test_user_item_matrix.sum(axis=0)).squeeze()



        self.train_data_num=len(train_user_index_np)
        self.test_data_num=len(test_user_index_np)

        # pre_calculate(for sample)
        self.all_pos_items = self.get_all_pos_items(
            list(range(self.user_number)))

        #
        self.hash_u_to_i_test=self.get_hash_u_to_i_test(list(range(self.user_number)))
        self.hash_i_to_u_test=self.get_hash_i_to_u_test(list(range(self.item_number)))
        self.hash_u_to_i_train=self.get_hash_u_to_i_train(list(range(self.user_number)))
        self.hash_i_to_u_train=self.get_hash_i_to_u_train(list(range(self.item_number)))


    def get_all_pos_items(self,users):
        all_pos_items=[]
        for user in users:
            all_pos_items.append(self.train_user_item_matrix[user].nonzero()[1])
        return all_pos_items


    def get_hash_u_to_i_test(self,users):
        '''
        :return: test_dict{user_id:[items_id]}
        '''
        hash_u_to_i_test={}
        for user in users:
            items=list(self.test_user_item_matrix[user].nonzero()[1])
            if len(items)==0:
                continue
            else:
                hash_u_to_i_test[user]=items
        return hash_u_to_i_test

    def get_hash_i_to_u_test(self,items):
        '''

        :param items: list_of_items
        :return: test_dict{item_id:[users_id]}
        '''
        hash_i_to_u_test={}
        for item in items:
            users=list(self.test_user_item_matrix[:,item].nonzero()[0])
            if len(users)==0:
                continue
            else:
                hash_i_to_u_test[item]=users
        return hash_i_to_u_test

    def get_hash_u_to_i_train(self,users):
        '''
        :return: train_dict{user_id:[items_id]}
        '''
        hash_u_to_i_train={}
        for user in users:
            items=list(self.train_user_item_matrix[user].nonzero()[1])
            if len(items)==0:
                #print("cunzai")
                #print(user)
                continue
            else:
                hash_u_to_i_train[user]=items
        return hash_u_to_i_train

    def get_hash_i_to_u_train(self,items):
        '''

        :param items: list_of_items
        :return: train_dict{item_id:[users_id]}
        '''
        hash_i_to_u_train={}
        #count=0
        for item in items:
            users=list(self.train_user_item_matrix[:,item].nonzero()[0])
            if len(users)==0:
                #count+=1
                #print(item)
                continue
            else:
                hash_i_to_u_train[item]=users
        #print(count)
        return hash_i_to_u_train


if __name__ == '__main__':
    dataloader=Dataloader()
    # print(dataloader.hash_i_to_u_train[102])
    # print(dataloader.hash_u_to_i_train[])