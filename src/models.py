import torch.nn as nn
import torch
from torch.distributions.normal import Normal


class BayesianLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.mu: nn.Parameter = nn.Parameter(torch.randn(in_features, out_features))
        self.ro: nn.Parameter = nn.Parameter(torch.randn(in_features, out_features))

        self.mu_bias: nn.Parameter = nn.Parameter(torch.randn(1, out_features))
        self.ro_bias: nn.Parameter = nn.Parameter(torch.randn(1, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size: int = x.shape[0]
        device = x.device

        eps: torch.Tensor = torch.normal(
            torch.zeros(batch_size, self.in_features, self.out_features)
        ).to(device)
        eps_bias: torch.Tensor = torch.normal(
            torch.ones(batch_size, self.out_features)
        ).to(device)

        sigma: torch.Tensor = torch.log(1 + torch.exp(self.ro)).to(device)
        sigma_bias: torch.Tensor = torch.log(1 + torch.exp(self.ro_bias)).to(device)

        weights: torch.Tensor = eps * sigma + self.mu
        bias: torch.Tensor = eps_bias * sigma_bias + self.mu_bias

        return (x.unsqueeze(1) @ weights).squeeze() + bias


class BayesModel(nn.Module):
    def __init__(
        self, input_size: int, output_size: int, hidden_sizes: tuple[int, ...]
    ) -> None:
        super().__init__()

        self.model: nn.Sequential = nn.Sequential(
            BayesianLayer(input_size, hidden_sizes[0]),
            nn.ReLU(inplace=True),
            *[
                torch.nn.Sequential(
                    BayesianLayer(hidden_sizes[i], hidden_sizes[i + 1]),
                    nn.ReLU(inplace=True),
                )
                for i in range(0, len(hidden_sizes) - 1)
            ],
            BayesianLayer(hidden_sizes[-1], output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.view(x.shape[0], -1))
