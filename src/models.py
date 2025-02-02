import torch.nn as nn
import torch
from torch.distributions.normal import Normal


class BayesianLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma1: int,
        sigma2: int,
        pi: float,
        repeat_n_times: int = 1,
    ) -> None:
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.mu: nn.Parameter = nn.Parameter(torch.randn(in_features, out_features))
        self.ro: nn.Parameter = nn.Parameter(torch.randn(in_features, out_features))

        self.mu_bias: nn.Parameter = nn.Parameter(torch.randn(1, out_features))
        self.ro_bias: nn.Parameter = nn.Parameter(torch.randn(1, out_features))

        self.sigma1: int = sigma1
        self.sigma2: int = sigma2
        self.pi: float = pi

        self.repeat_n_times: int = repeat_n_times

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size: int = x.shape[0]
        device = x.device

        eps: torch.Tensor = torch.normal(
            torch.zeros(
                batch_size, self.repeat_n_times, self.in_features, self.out_features
            )
        ).to(device)
        eps_bias: torch.Tensor = torch.normal(
            torch.zeros(batch_size, self.repeat_n_times, self.out_features)
        ).to(device)

        sigma: torch.Tensor = torch.log(1 + torch.exp(self.ro)).to(device)
        sigma_bias: torch.Tensor = torch.log(1 + torch.exp(self.ro_bias)).to(device)

        weights: torch.Tensor = eps * sigma + self.mu
        bias: torch.Tensor = eps_bias * sigma_bias + self.mu_bias

        # calculate prior
        self.log_prior = (
            scale_gaussian_mixture(weights, self.pi, self.sigma1, self.sigma2)
            .log()
            .sum()
            + scale_gaussian_mixture(bias, self.pi, self.sigma1, self.sigma2)
            .log()
            .sum()
        ) / (batch_size * self.repeat_n_times)

        # Calculate log probability of weights
        self.log_p_weights = (
            gaussian(weights, self.mu, sigma).log().sum()
            + gaussian(bias, self.mu_bias, sigma_bias).log().sum()
        ) / (batch_size * self.repeat_n_times)

        return (
            (x.unsqueeze(1).repeat(1, self.repeat_n_times, 1).unsqueeze(1) @ weights)[
                :, :, 0, :
            ]
            + bias
        ).mean(dim=1)


class BayesModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: tuple[int, ...],
        neglog_sigma1: int = 1,
        neglog_sigma2: int = 6,
        pi=1 / 4,
        repeat_n_times: int = 1,
    ) -> None:
        super().__init__()

        self.sigma1: float = 10**-neglog_sigma1
        self.sigma2: float = 10**-neglog_sigma2
        self.pi: float = pi

        self.repeat_n_times = repeat_n_times

        self.model: nn.Sequential = nn.Sequential(
            BayesianLayer(
                input_size,
                hidden_sizes[0],
                self.sigma1,
                self.sigma2,
                pi,
                repeat_n_times,
            ),
            nn.ReLU(inplace=True),
            *[
                torch.nn.Sequential(
                    BayesianLayer(
                        hidden_sizes[i],
                        hidden_sizes[i + 1],
                        self.sigma1,
                        self.sigma2,
                        pi,
                        repeat_n_times,
                    ),
                    nn.ReLU(inplace=True),
                )
                for i in range(0, len(hidden_sizes) - 1)
            ],
            BayesianLayer(
                hidden_sizes[-1],
                output_size,
                self.sigma1,
                self.sigma2,
                pi,
                repeat_n_times,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.view(x.shape[0], -1))

    def log_prior(self) -> torch.Tensor:
        prior: float = 0.0
        num_layers: int = 0
        for layer in self.model:
            if isinstance(layer, BayesianLayer):
                prior += layer.log_prior
                num_layers += 1
        return prior / num_layers

    def log_p_weights(self):
        p: float = 0.0
        num_layers: int = 0
        for layer in self.model:
            if isinstance(layer, BayesianLayer):
                p += layer.log_p_weights
                num_layers += 1

        return p / num_layers


def gaussian(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    scaler = 1.0 / ((2.0 * torch.pi) ** 0.5 * sigma)
    bell = scaler * torch.exp(-((x - mu) ** 2) / (2.0 * sigma**2))
    return torch.clamp(bell, 1e-10, 10)


def scale_gaussian_mixture(
    x: torch.Tensor, pi: float, sigma1: float, sigma2: float
) -> torch.Tensor:
    return pi * gaussian(x, torch.tensor(0.0), torch.tensor(sigma1)) + (
        1 - pi
    ) * gaussian(x, torch.tensor(0.0), torch.tensor(sigma2))
