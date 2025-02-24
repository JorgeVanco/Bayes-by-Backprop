import torch
import torch.nn as nn
from src.bayes_utils import gaussian, scale_gaussian_mixture


class BayesianModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.is_bayesian: bool = True

    @property
    def log_prior(self) -> torch.Tensor:
        prior: torch.Tensor = torch.tensor(0.0)
        num_layers: int = 0
        for layer in self.model:
            if hasattr(
                layer, "is_bayesian"
            ):  # issubclass(type(layer), (BayesianLayer, BayesianModule)):
                prior += layer.log_prior
                num_layers += 1
        return prior / num_layers

    @property
    def log_p_weights(self) -> float:
        p: float = 0.0
        num_layers: int = 0
        for layer in self.model:
            if hasattr(
                layer, "is_bayesian"
            ):  # issubclass(type(layer), (BayesianLayer, BayesianModule)):
                p += layer.log_p_weights
                num_layers += 1

        return p / num_layers


class BayesianLayer(nn.Module):
    def __init__(
        self, sigma1: float, sigma2: float, pi: float, repeat_n_times: int
    ) -> None:
        super().__init__()
        self.is_bayesian: bool = True
        self._log_prior: torch.Tensor = torch.empty(1)
        self._log_p_weights: torch.Tensor = torch.empty(1)
        self.sigma1: int = sigma1
        self.sigma2: int = sigma2
        self.pi: float = pi
        self.repeat_n_times: int = repeat_n_times

    @property
    def log_prior(self) -> float:
        return self._log_prior.item()

    @property
    def log_p_weights(self) -> float:
        return self._log_p_weights.item()

    def calculate_log_prior(
        self, weights: torch.Tensor, bias: torch.Tensor, batch_size: int
    ) -> None:
        self._log_prior = (
            scale_gaussian_mixture(weights, self.pi, self.sigma1, self.sigma2)
            .log()
            .sum()
            / weights.numel()
            + scale_gaussian_mixture(bias, self.pi, self.sigma1, self.sigma2)
            .log()
            .sum()
            / bias.numel()
        ) / (batch_size * self.repeat_n_times)

    def calculate_log_p_weights(
        self,
        weights: torch.Tensor,
        bias: torch.Tensor,
        batch_size: int,
        sigma: torch.Tensor,
        sigma_bias: torch.Tensor,
    ) -> None:
        self._log_p_weights = (
            gaussian(weights, self.mu, sigma).log().sum() / weights.numel()
            + gaussian(bias, self.mu_bias, sigma_bias).log().sum() / bias.numel()
        ) / (batch_size * self.repeat_n_times)
