import torch

class SwiGLU(torch.nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
    def forward(x: torch.Tensor) -> torch.Tensor:
        silu = torch.sigmoid(x) * x