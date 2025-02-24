import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
    ) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ConvModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: tuple[int, ...],
    ) -> None:
        super().__init__()

        self.model = torch.nn.Sequential(
            ConvBlock(
                input_size,
                hidden_sizes[0],
                stride=2,
            ),
            *[
                ConvBlock(
                    hidden_sizes[i],
                    hidden_sizes[i + 1],
                    stride=2,
                )
                for i in range(0, len(hidden_sizes) - 1)
            ],
            ConvBlock(
                hidden_sizes[-1],
                4 * output_size,
                stride=2,
            ),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(4 * output_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
