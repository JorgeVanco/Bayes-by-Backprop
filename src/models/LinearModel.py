import torch
import torch.nn as nn


class LinearModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: tuple[int, ...],
    ) -> None:
        super().__init__()

        self.model: nn.Sequential = nn.Sequential(
            nn.Linear(
                input_size,
                hidden_sizes[0],
            ),
            nn.ReLU(inplace=True),
            *[
                torch.nn.Sequential(
                    nn.Linear(
                        hidden_sizes[i],
                        hidden_sizes[i + 1],
                    ),
                    nn.ReLU(inplace=True),
                )
                for i in range(0, len(hidden_sizes) - 1)
            ],
            nn.Linear(
                hidden_sizes[-1],
                output_size,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.view(x.shape[0], -1))
