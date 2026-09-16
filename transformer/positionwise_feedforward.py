import torch

class FFN(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int = None, device: torch.device = None, dtype: torch.dtype = None) -> None:
        super().__init__()

        self.d_ff = int(8 / 3 * d_model) if d_ff is None else d_ff
        self.d_model = d_model

        self.w1 = torch.nn.Parameter(
            torch.empty(self.d_ff, self.d_model, device=device, dtype=dtype)
        )
        self.w2 = torch.nn.Parameter(
            torch.empty(self.d_model, self.d_ff, device=device, dtype=dtype)
        )
        self.w3 = torch.nn.Parameter(
            torch.empty(self.d_ff, self.d_model, device=device, dtype=dtype)
        )

        torch.nn.init.xavier_uniform_(self.w1)
        torch.nn.init.xavier_uniform_(self.w2)
        torch.nn.init.xavier_uniform_(self.w3)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = x @ self.w1.T
        silu = torch.sigmoid(w1x) * (w1x)
        result = (silu * (x @ self.w3.T)) @ self.w2.T

        return result