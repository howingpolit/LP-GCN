import numpy as np
import torch
def minibatch(*array,batch_size):
    '''

    :param tensors: tuple(user_id,pos_item,neg_item)
    :param batch_size: int
    :return:tuple(user_id,pos_item,neg_item)after split
    '''
    for i in range(0,len(array[0]),batch_size):
        yield tuple(x[i:i+batch_size] for x in array)

def shuffle(*array):
    '''
    :param tensors: tuple(user_id,pos_item,neg_item)
    :return: tuple(user_id,pos_item,neg_item)after shuffle
    '''
    if len(set(len(x) for x in array)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')
    shuffle_index=np.arange(len(array[0]))
    np.random.shuffle(shuffle_index)

    result = tuple(x[shuffle_index] for x in array)
    return result

def get_user_and_item_embedding_matrix(clients, server):
    #
    for client in clients.values():
        client.get_average_user_embedding()
    #
    max_id = max(clients.keys())
    num_features = clients[next(iter(clients))].average_user_embedding.shape[0]
    # print(clients[next(iter(clients))].user_embedding)
    # print(num_features)
    #
    user_embedding_matrix = torch.zeros(max_id + 1, num_features)

    #
    for user_id, client in clients.items():
        # print(client.user_embedding)
        # print(user_embedding_matrix.shape)
        # print(user_embedding_matrix)
        user_embedding_matrix[user_id] = client.average_user_embedding.clone()

    item_embedding_org=server.all_item_embedding_of_all_layer
    #
    item_embeddings = {}
    # print(data.values())
    #
    for layer in item_embedding_org.values():
        for item_id, embedding in layer.items():
            if item_id in item_embeddings:
                item_embeddings[item_id].append(embedding)
            else:
                item_embeddings[item_id] = [embedding]

    #
    for item_id in item_embeddings:
        item_embeddings[item_id] = torch.stack(item_embeddings[item_id]).mean(dim=0)

    #
    max_id = max(item_embeddings.keys())
    num_features = next(iter(item_embeddings.values())).shape[0]

    #
    item_embedding_matrix = torch.zeros(max_id + 1, num_features)

    #
    for item_id, avg_embedding in item_embeddings.items():
        item_embedding_matrix[item_id] = avg_embedding
    return user_embedding_matrix,item_embedding_matrix