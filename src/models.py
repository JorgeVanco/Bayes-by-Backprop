import torch.nn as nn
import torch
import torch.nn.functional as F
import math

from src.bayes_utils import gaussian, scale_gaussian_mixture


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
        self.ro: nn.Parameter = nn.Parameter(
            torch.Tensor(in_features, out_features).normal_(-8, 0.05)
        )
        torch.nn.init.kaiming_uniform_(self.mu, nonlinearity="relu")

        self.mu_bias: nn.Parameter = nn.Parameter(torch.randn(1, out_features))
        self.ro_bias: nn.Parameter = nn.Parameter(
            torch.Tensor(1, out_features).normal_(-8, 0.05)
        )

        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.mu)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(self.mu_bias, -bound, bound)

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
            / weights.numel()
            + scale_gaussian_mixture(bias, self.pi, self.sigma1, self.sigma2)
            .log()
            .sum()
            / bias.numel()
        ) / (batch_size * self.repeat_n_times)

        # Calculate log probability of weights
        self.log_p_weights = (
            gaussian(weights, self.mu, sigma).log().sum() / weights.numel()
            + gaussian(bias, self.mu_bias, sigma_bias).log().sum() / bias.numel()
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
        pi: float = 1 / 4,
        repeat_n_times: int = 1,
    ) -> None:
        super().__init__()

        self.sigma1: float = 10**-neglog_sigma1
        self.sigma2: float = 10**-neglog_sigma2
        self.pi: float = pi

        self.repeat_n_times: int = repeat_n_times

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


class BayesianConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        sigma1: int,
        sigma2: int,
        pi: float,
        repeat_n_times: int = 1,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ):
        super().__init__()
        self.shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.mu: nn.Parameter = nn.Parameter(torch.randn(self.shape))
        self.ro: nn.Parameter = nn.Parameter(torch.randn(self.shape).normal_(-8, 0.05))
        torch.nn.init.kaiming_uniform_(self.mu, nonlinearity="relu")

        self.mu_bias: nn.Parameter = nn.Parameter(torch.randn(out_channels))
        self.ro_bias: nn.Parameter = nn.Parameter(
            torch.randn(out_channels).normal_(-8, 0.05)
        )

        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.mu)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(self.mu_bias, -bound, bound)

        self.sigma1: int = sigma1
        self.sigma2: int = sigma2
        self.pi: float = pi

        self.repeat_n_times: int = repeat_n_times

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size: int = x.shape[0]
        device = x.device

        eps: torch.Tensor = torch.normal(torch.zeros(self.shape)).to(device)
        eps_bias: torch.Tensor = torch.normal(torch.zeros(self.out_channels)).to(device)

        sigma: torch.Tensor = torch.log(1 + torch.exp(self.ro)).to(device)
        sigma_bias: torch.Tensor = torch.log(1 + torch.exp(self.ro_bias)).to(device)

        weights: torch.Tensor = eps * sigma + self.mu
        bias: torch.Tensor = eps_bias * sigma_bias + self.mu_bias

        # calculate prior
        self.log_prior = (
            scale_gaussian_mixture(weights, self.pi, self.sigma1, self.sigma2)
            .log()
            .sum()
            / weights.numel()
            + scale_gaussian_mixture(bias, self.pi, self.sigma1, self.sigma2)
            .log()
            .sum()
            / bias.numel()
        ) / (batch_size * self.repeat_n_times)

        # Calculate log probability of weights
        self.log_p_weights = (
            gaussian(weights, self.mu, sigma).log().sum() / weights.numel()
            + gaussian(bias, self.mu_bias, sigma_bias).log().sum() / bias.numel()
        ) / (batch_size * self.repeat_n_times)

        return F.conv2d(x, weights, bias, self.stride, self.padding, self.dilation)
        return (
            (x.unsqueeze(1).repeat(1, self.repeat_n_times, 1).unsqueeze(1) @ weights)[
                :, :, 0, :
            ]
            + bias
        ).mean(dim=1)


class BayesConvModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: tuple[int, ...],
        neglog_sigma1: int = 1,
        neglog_sigma2: int = 6,
        pi: float = 1 / 4,
        repeat_n_times: int = 1,
    ):
        super().__init__()
        self.sigma1: float = 10**-neglog_sigma1
        self.sigma2: float = 10**-neglog_sigma2
        self.pi: float = pi

        self.repeat_n_times: int = repeat_n_times
        self.conv = BayesianConv(
            input_size, 64, self.sigma1, self.sigma2, self.pi, self.repeat_n_times
        )
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear = BayesianLayer(64, output_size, self.sigma1, self.sigma2, self.pi)
        self.model = torch.nn.Sequential(
            self.conv, self.relu, self.avg_pool, self.flatten, self.linear
        )

    def forward(self, x):
        o = self.conv(x)
        o = self.relu(o)
        o = self.avg_pool(o)
        o = self.flatten(o)

        return self.linear(o)

    def log_prior(self) -> torch.Tensor:
        prior: float = 0.0
        num_layers: int = 0
        for layer in self.model:
            if isinstance(layer, BayesianLayer) or isinstance(layer, BayesianConv):
                prior += layer.log_prior
                num_layers += 1
        return prior / num_layers

    def log_p_weights(self):
        p: float = 0.0
        num_layers: int = 0
        for layer in self.model:
            if isinstance(layer, BayesianLayer) or isinstance(layer, BayesianConv):
                p += layer.log_p_weights
                num_layers += 1

        return p / num_layers


# BayesianConv(3, 10, 3, 2, 1)
# torch.nn.functional.conv2d(
#     torch.randn(8, 3, 24, 24), torch.randn(10, 3, 3, 3), torch.randn(10), stride=1
# )
