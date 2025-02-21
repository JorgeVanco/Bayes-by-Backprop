import torch
import torch.nn as nn
import math

from src.models.BayesModels import BayesianLayer, BayesianModule


class BayesianLinearLayer(BayesianLayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma1: float,
        sigma2: float,
        pi: float,
        repeat_n_times: int = 1,
    ) -> None:
        super().__init__(sigma1, sigma2, pi, repeat_n_times)
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
        self.calculate_log_prior(weights, bias, batch_size)

        # Calculate log probability of weights
        self.calculate_log_p_weights(weights, bias, batch_size, sigma, sigma_bias)
        return (x.unsqueeze(1).unsqueeze(2) @ weights).squeeze().mean(dim=1)


class BayesModel(BayesianModule):
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
            BayesianLinearLayer(
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
                    BayesianLinearLayer(
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
            BayesianLinearLayer(
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
