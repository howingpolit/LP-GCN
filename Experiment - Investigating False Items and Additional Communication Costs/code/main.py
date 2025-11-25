from client_classify import find_minimum_user_set_sparse
from dataloader import Loader
import world
import pickle
import util
import csv
import numpy as np
from datetime import datetime


dataloader = Loader(path="../data/"+world.dataset)
# convolution_clients_id_set = find_minimum_user_set_sparse(dataloader.UserItemNet)
# filename = f'{world.dataset}.pickle'
# with open(filename, 'wb') as file:
#     pickle.dump(convolution_clients_id_set, file)
now = datetime.now()
now_str = now.strftime("%Y-%m-%d-%H-%M-%S")
result_filename=f"../experiment/{world.dataset}-results-"+now_str+".csv"
with open(result_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['add_number_rate','rate of convolution client', 'neighboring_users_number_sum', 'average_neighboring_users_number','average_computation','average_neighbor_users_utilize_rate'])
for rate in np.arange(0, 0.87, 0.1):
    seeds=[0,1,2,3,4]
    ratio_of_con_clients_sum=0.0
    neighboring_users_number_sum=0.0
    average_neighboring_users_number_sum=0.0
    average_computation_sum=0.0
    average_neighbor_users_utilize_rate_sum=0.0
    for seed in seeds:
        filename=f'{world.dataset}.pickle'

        with open(filename, 'rb') as file:
            convolution_clients_id_set = pickle.load(file)


        try:
            with open(f"{world.dataset}_item_to_users.pickle", 'rb') as file:
                item_to_users = pickle.load(file)
            with open(f"{world.dataset}_user_to_items.pickle", 'rb') as file:
                user_to_items = pickle.load(file)
            print("load item_to_users,user_to_items successfully")
        except:
            #
            item_to_users,user_to_items=util.get_hash_between_user_and_item(dataloader)
            print("Obtain the set of users who interact with each item i, and the set of items that each user interacts with")
            with open(f"{world.dataset}_item_to_users.pickle", 'wb') as file:
                pickle.dump(item_to_users, file)
            with open(f"{world.dataset}_user_to_items.pickle", 'wb') as file:
                pickle.dump(user_to_items, file)
            print("generate item_to_users,user_to_items")

        user_to_items,item_to_users,convolution_clients_id_set=util.add_fake_item(convolution_clients_id_set,user_to_items,item_to_users,dataloader,m=0.1,p=0.01)

        convolution_clients_id_set=util.get_new_convolution_clients_id_set(add_number_rate=rate,convolution_clients_id_set=convolution_clients_id_set,user_to_items=user_to_items)

        #
        item_id_to_convolution_client_id=util.get_item_id_to_con_client_id(dataloader,item_to_users,convolution_clients_id_set,seed)


        #
        con_items_of_each_con_client=util.get_con_items_of_each_con_client(item_id_to_convolution_client_id)


        #
        neighboring_users_of_each_con_client,ordinary_client_number,total_repeat_neighbor_user_number=util.get_neighboring_users_of_each_con_client(convolution_clients_id_set,con_items_of_each_con_client,item_to_users)

        #
        neighboring_users_number,average_neighboring_users_number=util.count_neighboring_users_number(neighboring_users_of_each_con_client)

        ratio_of_con_clients=(len(convolution_clients_id_set)-ordinary_client_number)/dataloader.n_users
        average_computation=dataloader.m_item/len(convolution_clients_id_set)
        neighbor_users_utilize_rate=total_repeat_neighbor_user_number/neighboring_users_number


        ratio_of_con_clients_sum+=ratio_of_con_clients
        neighboring_users_number_sum+=neighboring_users_number
        average_neighboring_users_number_sum+=average_neighboring_users_number
        average_computation_sum+=average_computation
        average_neighbor_users_utilize_rate_sum+=neighbor_users_utilize_rate


    ratio_of_con_clients=ratio_of_con_clients_sum/len(seeds)
    neighboring_users_number=neighboring_users_number_sum/len(seeds)
    average_neighboring_users_number=average_neighboring_users_number_sum/len(seeds)
    average_computation=average_computation_sum/len(seeds)
    average_neighbor_users_utilize_rate=average_neighbor_users_utilize_rate_sum/len(seeds)



    print("Convolutional client ratio：",ratio_of_con_clients)
    print("neighboring_users_number:",neighboring_users_number)
    print("average_neighboring_users_number:",average_neighboring_users_number)
    print("average_computation:",average_computation)
    print("average_neighbor_users_utilize_rate:", average_neighbor_users_utilize_rate)
    with open(result_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([rate, ratio_of_con_clients, neighboring_users_number,average_neighboring_users_number, average_computation,average_neighbor_users_utilize_rate])
