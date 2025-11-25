# import numpy as np
# from scipy.sparse import csr_matrix
#
# #
# trainUser = np.array([0, 0, 2, 2, 3])
# trainItem = np.array([0, 2, 0, 1, 2])
#
# #
# n_user = 5
# m_item = 4
#
# #
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
#
a=[[1,2],[3,4]]
print([1,5] in a)
# print("Result using set operation:", result)

