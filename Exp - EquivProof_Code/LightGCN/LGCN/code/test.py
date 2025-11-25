import torch

class Client:
    def __init__(self, user_embedding):
        self.user_embedding = user_embedding

#
clients = {
    1: Client(torch.tensor([[0.1, 0.2, 0.3]])),
    17: Client(torch.tensor([0.4, 0.5, 0.6])),
    12: Client(torch.tensor([0.7, 0.8, 0.9]))
}

#
max_id = max(clients.keys())
num_features = clients[next(iter(clients))].user_embedding.shape[1]
print(clients[next(iter(clients))].user_embedding)
print(num_features)
#
embeddings_matrix = torch.zeros(max_id + 1, num_features)

#
for user_id, client in clients.items():
    embeddings_matrix[user_id] = client.user_embedding.clone()

#
print(embeddings_matrix)
