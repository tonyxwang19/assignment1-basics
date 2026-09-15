import torch

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        empty_tensor = torch.empty(d_model, device = device, dtype = dtype)
        weight_initialized = torch.nn.init.constant_(empty_tensor, 1.0)
        self.gain = torch.nn.Parameter(data = weight_initialized, requires_grad = True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(
            x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )

        result = x * self.gain / rms

        return result.to(in_dtype)

