# import numpy as np
# from scipy.sparse import csr_matrix
#
#
# trainUser = np.array([0, 0, 2, 2, 3])
# trainItem = np.array([0, 2, 0, 1, 2])
#
#
# n_user = 5
# m_item = 4
#
#
# UserItemNet = csr_matrix((np.ones(len(trainUser)), (trainUser, trainItem)), shape=(n_user, m_item))
#
#
# item_to_users = {i: UserItemNet[:, i].nonzero()[0].tolist() for i in range(m_item) if UserItemNet[:, i].nnz > 0}
# user_to_items = {i: UserItemNet[i, :].nonzero()[1].tolist() for i in range(n_user) if UserItemNet[i, :].nnz > 0}
#
# print(item_to_users)
# print(user_to_items)

#
#
# import numpy as np
# # for i in np.arange(0, 0.93, 0.01):
# #     print(i)
# from datetime import datetime
#
#
# now = datetime.now()
#
#
# now_str = now.strftime("%Y-%m-%d-%H:%M:%S")
#
#
# print(now_str)
#
#
import random

#
user_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

#
item_id_range = (1, 1+3)

#
def sample_user_item_pair(user_list, item_id_range):
    user_id = random.choice(user_list)
    item_id = random.randint(item_id_range[0], item_id_range[1])
    return user_id, item_id

#
user_item_pair = sample_user_item_pair(user_list, item_id_range)

#
print("Sampled User-Item Pair:", user_item_pair)
