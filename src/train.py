# deep learning libraries
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# other libraries
from tqdm.auto import tqdm
from typing import Final, Literal

# own modules
from src.utils import load_data, save_model
from src.models import BayesModel, BayesConvModel, ConvModel, LinearModel
from src.train_functions import (
    train_step,
    val_step,
    loss_function,
    cross_entropy_loss_function,
)

torch.autograd.set_detect_anomaly(True)
# static variables
DATA_PATH: Final[str] = "data"
NUM_CLASSES: Final[int] = 10

# set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main() -> None:
    """
    This function is the main program for training.
    """
    print("Using: ", device)

    # hyperparameters
    model_type: Literal["conv", "linear", "convBayes", "linearBayes"] = "linearBayes"
    epochs: int = 10
    lr: float = 1e-3
    batch_size: int = 128
    hidden_sizes: tuple[int, ...] = (256, 128)  # (64, 128)  # (256, 128, 64)
    repeat_n_times: int = 1
    kl_reweighting: bool = True

    # empty nohup file
    open("nohup.out", "w").close()

    # load data
    train_data: DataLoader
    val_data: DataLoader
    train_data, val_data, _ = load_data(DATA_PATH, batch_size=batch_size)

    # define name and writer
    name: str = f"{model_type}_lr_{lr}_hs_{hidden_sizes}_{batch_size}_{epochs}"
    writer: SummaryWriter = SummaryWriter(f"runs/{name}")

    # define model
    inputs: torch.Tensor = next(iter(train_data))[0]

    if model_type == "linearBayes":
        model: torch.nn.Module = BayesModel(
            inputs.shape[2] * inputs.shape[3],
            NUM_CLASSES,
            hidden_sizes=hidden_sizes,
            repeat_n_times=repeat_n_times,
        ).to(device)
    elif model_type == "convBayes":
        model: torch.nn.Module = BayesConvModel(
            inputs.shape[1], NUM_CLASSES, hidden_sizes, repeat_n_times=repeat_n_times
        ).to(device)
    elif model_type == "conv":
        model: torch.nn.Module = ConvModel(
            inputs.shape[1], NUM_CLASSES, hidden_sizes
        ).to(device)
    elif model_type == "linear":
        model: torch.nn.Module = LinearModel(
            inputs.shape[2] * inputs.shape[3], NUM_CLASSES, hidden_sizes
        ).to(device)

    # define loss and optimizer
    loss = loss_function if "Bayes" in model_type else cross_entropy_loss_function
    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Training: {name}")
    # train loop
    for epoch in tqdm(range(epochs), desc="epochs", position=0):
        # call train step
        train_step(
            model, train_data, loss, optimizer, writer, epoch, device, kl_reweighting
        )

        # call val step
        val_step(model, val_data, loss, writer, epoch, device)

    # save model
    save_model(model, name)

    return None


if __name__ == "__main__":
    main()
