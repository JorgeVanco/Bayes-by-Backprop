# deep learning libraries
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# other libraries
from tqdm.auto import tqdm
from typing import Final

# own modules
from src.utils import load_data, save_model
from src.models import BayesModel
from src.train_functions import train_step, val_step

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
    epochs: int = 100
    lr: float = 1e-3
    batch_size: int = 32
    hidden_sizes: tuple[int, ...] = (256, 128, 64)

    # empty nohup file
    open("nohup.out", "w").close()

    # load data
    train_data: DataLoader
    val_data: DataLoader
    train_data, val_data, _ = load_data(DATA_PATH, batch_size=64)

    # define name and writer
    name: str = f"model_lr_{lr}_hs_{hidden_sizes}_{batch_size}_{epochs}"
    writer: SummaryWriter = SummaryWriter(f"runs/{name}")

    # define model
    inputs: torch.Tensor = next(iter(train_data))[0]
    model: torch.nn.Module = BayesModel(
        inputs.shape[2] * inputs.shape[3], NUM_CLASSES, hidden_sizes=hidden_sizes
    ).to(device)

    # define loss and optimizer
    loss: torch.nn.Module = torch.nn.CrossEntropyLoss()
    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # train loop
    for epoch in tqdm(range(epochs)):
        # call train step
        train_step(model, train_data, loss, optimizer, writer, epoch, device)

        # call val step
        val_step(model, val_data, loss, writer, epoch, device)

    # save model
    save_model(model, name)

    return None


if __name__ == "__main__":
    main()
