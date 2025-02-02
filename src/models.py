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
            torch.zeros(batch_size, self.out_features)
        ).to(device)

        self.sigma: torch.Tensor = torch.log(1 + torch.exp(self.ro)).to(device)
        self.sigma_bias: torch.Tensor = torch.log(1 + torch.exp(self.ro_bias)).to(
            device
        )

        self.weights: torch.Tensor = eps * self.sigma + self.mu
        self.bias: torch.Tensor = eps_bias * self.sigma_bias + self.mu_bias

        return (x.unsqueeze(1) @ self.weights).squeeze() + self.bias


class BayesModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: tuple[int, ...],
        neglog_sigma1=1,
        neglog_sigma2=6,
        pi=1 / 4,
    ) -> None:
        super().__init__()

        self.sigma1 = 10**-neglog_sigma1
        self.sigma2 = 10**-neglog_sigma2
        self.pi = pi

        self.model: nn.Sequential = nn.Sequential(
            BayesianLayer(input_size, hidden_sizes[0], self.sigma1, self.sigma2),
            nn.ReLU(inplace=True),
            *[
                torch.nn.Sequential(
                    BayesianLayer(
                        hidden_sizes[i], hidden_sizes[i + 1], self.sigma1, self.sigma2
                    ),
                    nn.ReLU(inplace=True),
                )
                for i in range(0, len(hidden_sizes) - 1)
            ],
            BayesianLayer(hidden_sizes[-1], output_size, self.sigma1, self.sigma2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.view(x.shape[0], -1))

    def log_prior(self) -> torch.Tensor:
        prior: torch.Tensor = torch.tensor(0.0)
        for layer in self.model:
            if isinstance(layer, BayesianLayer):
                for weights in (layer.weights, layer.bias):
                    prior = prior.to(weights.device)
                    try:
                        prior += (
                            (
                                self.pi * gaussian(weights, 0, self.sigma1)
                                + (1 - self.pi) * gaussian(weights, 0, self.sigma2)
                            )
                            .log()
                            .sum()
                        ) / weights.numel()
                    except Exception as e:
                        print(e)
                        print(weights.max(), weights.min(), weights.mean())
        return prior / 1000

    def log_p_weights(self):
        p = torch.tensor(0.0)
        for layer in self.model:
            if isinstance(layer, BayesianLayer):
                for weights, name in zip(
                    [layer.weights, layer.bias], ["weights", "bias"]
                ):
                    p = p.to(weights.device)

                    if name == "weights":
                        mu = layer.mu
                        sigma = layer.sigma
                    else:
                        mu = layer.mu_bias
                        sigma = layer.sigma_bias

                    p += gaussian(weights, mu, sigma).log().sum() / weights.numel()

        return p / 1000


def gaussian(x, mu, sigma):
    scaler = 1.0 / ((2.0 * torch.pi) ** 0.5 * sigma)
    bell = scaler * torch.exp(-((x - mu) ** 2) / (2.0 * sigma**2))
    return torch.clamp(bell, 1e-10, 10)
