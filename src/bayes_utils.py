import torch


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
