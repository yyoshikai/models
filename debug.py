import torch

def count_nan(name: str, tensor: torch.Tensor):
    print(f"{name}: {torch.sum(torch.isnan(tensor))}/{torch.numel(tensor)}")