import torch.nn as nn
import torch
from torch.distributions.normal import Normal


class BayesianLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mu: nn.Parameter = nn.Parameter(torch.randn(in_features, out_features))
        self.ro: nn.Parameter = nn.Parameter(torch.randn(in_features, out_features))

        self.mu_bias: nn.Parameter = nn.Parameter(torch.randn(1, out_features))
        self.ro_bias: nn.Parameter = nn.Parameter(torch.randn(1, out_features))

        self.normal: Normal = Normal(0, 1)

    def forward(self, x):
        batch_size = x.shape[0]

        eps = self.normal.rsample((batch_size, self.in_features, self.out_features))
        eps_bias = self.normal.rsample((batch_size, self.out_features))

        sigma = torch.log(1 + torch.exp(self.ro))
        sigma_bias = torch.log(1 + torch.exp(self.ro_bias))

        weights = eps * sigma + self.mu
        bias = eps_bias * sigma_bias + self.mu_bias

        return (x.unsqueeze(1) @ weights).squeeze() + bias
