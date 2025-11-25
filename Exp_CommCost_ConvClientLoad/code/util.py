import random


#
def get_hash_between_user_and_item(dataloader):
    UserItemNet=dataloader.UserItemNet
    m_item=dataloader.m_item
    n_user=dataloader.n_user
    #
    item_to_users = {i: UserItemNet[:, i].nonzero()[0].tolist() for i in range(m_item) if UserItemNet[:, i].nnz > 0}
    user_to_items = {i: UserItemNet[i, :].nonzero()[1].tolist() for i in range(n_user) if UserItemNet[i, :].nnz > 0}
    return item_to_users,user_to_items


#
def get_item_id_to_con_client_id(dataloader,item_to_users,convolution_clients_id_set,seed):
    print(seed)
    random.seed(seed)
    item_id_to_convolution_client_id = {}
    for i in range(dataloader.m_item):
        if i not in item_to_users.keys():
            continue
        convolution_client_id = random.choice(list(set(item_to_users[i]).intersection(convolution_clients_id_set)))
        item_id_to_convolution_client_id[i] = convolution_client_id
    return item_id_to_convolution_client_id


#
def get_con_items_of_each_con_client(item_id_to_convolution_client_id):
    #
    con_items_of_each_con_client = {}

    #
    for key, value in item_id_to_convolution_client_id.items():
        #
        if value not in con_items_of_each_con_client:
            con_items_of_each_con_client[value] = []
        #
        con_items_of_each_con_client[value].append(key)
    return con_items_of_each_con_client

#
def get_neighboring_users_of_each_con_client(convolution_clients_id_set,con_items_of_each_con_client,item_to_users):
    #
    neighboring_users_of_each_con_client={}
    ordinary_client_number=0
    total_repeat_neighbor_user_number=0
    for con_client_id in convolution_clients_id_set:
        if con_client_id not in con_items_of_each_con_client:
            ordinary_client_number+=1
            continue
        interacted_con_items = con_items_of_each_con_client[con_client_id]
        neighbors = set()
        total_users = 0  #
        #
        for item in interacted_con_items:
            users = item_to_users[item].copy()
            users.remove(con_client_id)
            total_users += len(users)
            neighbors.update(users)
        neighbors.discard(con_client_id)
        repeat_neighbor_number=total_users - len(neighbors)
        total_repeat_neighbor_user_number+=repeat_neighbor_number
        neighboring_users_of_each_con_client[con_client_id]=neighbors
    # print(ordinary_client_number)
    return neighboring_users_of_each_con_client,ordinary_client_number,total_repeat_neighbor_user_number
#
def count_neighboring_users_number(neighboring_users_of_each_con_client):
    neighboring_users_number=0.
    for con_client_id in neighboring_users_of_each_con_client.keys():
        neighboring_users_number+=len(neighboring_users_of_each_con_client[con_client_id])
    average_neighboring_users_number=neighboring_users_number/len(neighboring_users_of_each_con_client.keys())
    return neighboring_users_number,average_neighboring_users_number

#
def get_new_convolution_clients_id_set(add_number_rate,convolution_clients_id_set,user_to_items):
    users_set=set(user_to_items.keys())
    k=round(len(users_set)*add_number_rate)
    #
    difference = users_set - convolution_clients_id_set
    elements_to_add = random.sample(difference, k)
    new_convolution_clients_id_set = convolution_clients_id_set.union(elements_to_add)
    return new_convolution_clients_id_set


#
def add_fake_item(user_to_items,item_to_users,dataloader,m,p):
    user_to_items_include_fake_item=user_to_items.copy()
    item_to_users_include_fake_item=item_to_users.copy()
    item_max_number=dataloader.m_item
    fake_number=round(m*item_max_number)
    fake_pair_number=round(p*len(dataloader.trainUser))
    users_id_list=list(user_to_items.keys)
    item_id_range=(item_max_number,item_max_number+fake_number)
    fake_sample_pairs=[]
    for _ in range(fake_pair_number):
        user_id=random.choice(users_id_list)
        item_id=random.randint(item_id_range[0],item_id_range[1])
        sample_pair=[user_id,item_id]
        fake_sample_pairs.append(sample_pair)
    for p in fake_sample_pairs:
        user_to_items_include_fake_item[p[0]].append(p[1])
        item_to_users_include_fake_item[p[1]].append(p[0])
    return user_to_items_include_fake_item,item_to_users_include_fake_item
