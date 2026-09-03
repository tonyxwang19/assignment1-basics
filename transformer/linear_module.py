from einops import einsum
import torch
import math

class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        # Register and Initialize the Parameter
        std = math.sqrt(2/(self.in_features+self.out_features))
        empty_tensor = torch.empty(out_features, in_features, device = self.device, dtype = self.dtype)
        weight_initialized = torch.nn.init.trunc_normal_(tensor = empty_tensor,mean = 0, std = std, a = -3*std, b = 3*std)
        self.weight = torch.nn.Parameter(data = weight_initialized)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")