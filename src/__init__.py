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

        self.normal: Normal = Normal(0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size: int = x.shape[0]

        eps: torch.Tensor = self.normal.rsample(
            (batch_size, self.in_features, self.out_features)
        )
        eps_bias: torch.Tensor = self.normal.rsample((batch_size, self.out_features))

        sigma: torch.Tensor = torch.log(1 + torch.exp(self.ro))
        sigma_bias: torch.Tensor = torch.log(1 + torch.exp(self.ro_bias))

        weights: torch.Tensor = eps * sigma + self.mu
        bias: torch.Tensor = eps_bias * sigma_bias + self.mu_bias

        return (x.unsqueeze(1) @ weights).squeeze() + bias


class BayesModel(nn.Module):
    def __init__(
        self, input_size: int, hidden_layers: tuple[int, ...], output_size: int
    ) -> None:
        super().__init__()

        self.model: nn.Sequential = nn.Sequential(
            BayesianLayer(input_size, hidden_layers[0]),
            nn.ReLU(inplace=True),
            *[
                torch.nn.Sequential(
                    BayesianLayer(hidden_layers[i], hidden_layers[i + 1]),
                    nn.ReLU(inplace=True),
                )
                for i in range(1, len(hidden_layers) - 1)
            ],
            BayesianLayer(hidden_layers[-1], output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
