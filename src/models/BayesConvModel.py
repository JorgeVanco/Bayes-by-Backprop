import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.models.BayesModels import BayesianLayer, BayesianModule


class BayesianConv(BayesianLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        sigma1: float,
        sigma2: float,
        pi: float,
        repeat_n_times: int = 1,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ) -> None:
        super().__init__(sigma1, sigma2, pi, repeat_n_times)
        self.shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.stride: int = stride
        self.padding: int = padding
        self.dilation: int = dilation
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size: int = x.shape[0]
        device = x.device

        results = []
        log_prior: torch.Tensor = torch.tensor(0.0)
        log_p_weights: torch.Tensor = torch.tensor(0.0)
        for _ in range(self.repeat_n_times):
            eps: torch.Tensor = torch.normal(torch.zeros(self.shape)).to(device)
            eps_bias: torch.Tensor = torch.normal(torch.zeros(self.out_channels)).to(
                device
            )

            sigma: torch.Tensor = torch.log(1 + torch.exp(self.ro)).to(device)
            sigma_bias: torch.Tensor = torch.log(1 + torch.exp(self.ro_bias)).to(device)

            weights: torch.Tensor = eps * sigma + self.mu
            bias: torch.Tensor = eps_bias * sigma_bias + self.mu_bias

            # calculate prior
            self.calculate_log_prior(weights, bias, batch_size)
            log_prior += self.log_prior  # Sum over all n times

            # Calculate log probability of weights
            self.calculate_log_p_weights(weights, bias, batch_size, sigma, sigma_bias)
            log_p_weights += self.log_p_weights  # Sum over all n times

            result = F.conv2d(
                x, weights, bias, self.stride, self.padding, self.dilation
            )
            results.append(result)

        self._log_prior = log_prior
        self._log_p_weights = log_p_weights
        return torch.stack(results).mean(0)


class BayesConvBlock(BayesianModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        sigma1: float,
        sigma2: float,
        pi: float,
        repeat_n_times: int,
        stride: int,
    ) -> None:
        super().__init__()

        self.model = nn.Sequential(
            BayesianConv(
                in_channels,
                out_channels,
                sigma1,
                sigma2,
                pi,
                repeat_n_times,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            BayesianConv(
                out_channels,
                out_channels,
                sigma1,
                sigma2,
                pi,
                repeat_n_times,
                kernel_size=3,
                stride=stride,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            BayesianConv(
                out_channels,
                out_channels,
                sigma1,
                sigma2,
                pi,
                repeat_n_times,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class BayesConvModel(BayesianModule):
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

        self.model = torch.nn.Sequential(
            BayesConvBlock(
                input_size,
                hidden_sizes[0],
                self.sigma1,
                self.sigma2,
                self.pi,
                self.repeat_n_times,
                stride=2,
            ),
            *[
                BayesConvBlock(
                    hidden_sizes[i],
                    hidden_sizes[i + 1],
                    self.sigma1,
                    self.sigma2,
                    self.pi,
                    self.repeat_n_times,
                    stride=2,
                )
                for i in range(0, len(hidden_sizes) - 1)
            ],
            BayesConvBlock(
                hidden_sizes[-1],
                4 * output_size,
                self.sigma1,
                self.sigma2,
                self.pi,
                self.repeat_n_times,
                stride=2,
            ),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(4 * output_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
