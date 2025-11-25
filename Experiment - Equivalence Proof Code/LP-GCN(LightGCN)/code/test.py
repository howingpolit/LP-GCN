import torch

#
tensors = [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0]), torch.tensor([7.0, 8.0, 9.0])]

#
stacked_tensors = torch.stack(tensors)
average_tensor = torch.mean(stacked_tensors, dim=0)

print(average_tensor)
