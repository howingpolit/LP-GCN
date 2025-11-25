from typing import List
from scipy.sparse import csr_matrix

import time
import csv





def find_minimum_user_set_sparse(interaction_matrix: csr_matrix):
    """
    Finds the minimum set of users that covers all items for a sparse matrix, considering empty rows and columns.

    :param interaction_matrix: A CSR sparse matrix representing the user-item interaction.
    :return: A set of users that covers all items.
    """
    num_users, num_items = interaction_matrix.shape
    print("num_users:",num_users)
    print("num_items:", num_items)
    # start_time = time.time()
    #
    row_sums = interaction_matrix.sum(axis=1)
    col_sums = interaction_matrix.sum(axis=0)

    #
    empty_rows = set(i for i in range(interaction_matrix.shape[0]) if row_sums[i, 0] == 0)
    empty_cols = set(j for j in range(interaction_matrix.shape[1]) if col_sums[0, j] == 0)
    print("Empty rows and columns have been found")
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"Total time: {elapsed_time} seconds")


    #
    single_interacted_items = [item for item in range(interaction_matrix.shape[1]) if col_sums[0, item] == 1]
    user_set = set()

    for item in single_interacted_items:
        user = interaction_matrix.getcol(item).nonzero()[0][0]
        user_set.add(user)

    # for item in range(num_items):
    #
    #     if interaction_matrix.getcol(item).sum() == 1:
    #         #
    #         user = interaction_matrix.getcol(item).nonzero()[0][0]
    #         user_set.add(user)
    print("All users have been found")

    exclude_item=[]
    for user in user_set:
        user_items = interaction_matrix.getrow(user).nonzero()[1]
        exclude_item.extend(user_items)
    print("Found the union of items")

    covered_items = empty_cols.union(set(exclude_item))  # Initialize with empty columns as they are already 'covered'
    selected_users = set(user_set)

    while len(covered_items) < num_items:
        best_user = None
        max_cover = 0

        for user in range(num_users):
            if user in empty_rows:
                continue

            user_items = interaction_matrix.getrow(user).nonzero()[1]
            cover = len([item for item in user_items if item not in covered_items])

            if cover > max_cover:
                max_cover = cover
                best_user = user
            # print(num_users,user)

        selected_users.add(best_user)
        covered_items.update(interaction_matrix.getrow(best_user).nonzero()[1])
        ratio=len(covered_items) / num_items
        print("Arrived：",ratio)
        # if ratio>0.97:
        #     break
    selected_users={int(x) for x in selected_users}
    return selected_users

if __name__ == '__main__':
    result=find_minimum_user_set_sparse()



