import torch

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        empty_tensor = torch.empty(num_embeddings, embedding_dim, device = device, dtype = dtype)
        weight_initialized = torch.nn.init.trunc_normal_(tensor = empty_tensor,mean = 0, std = 1, a = -3, b = 3)
        self.embedding = torch.nn.Parameter(data = weight_initialized, requires_grad = True)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding[token_ids]