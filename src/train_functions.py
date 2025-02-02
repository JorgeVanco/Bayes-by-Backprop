# deep learning libraries
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# own modules
from src.utils import accuracy


def train_step(
    model: torch.nn.Module,
    train_data: DataLoader,
    loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    """
    This function computes the training step.

    Args:
        model: pytorch model.
        train_data: train dataloader.
        loss: loss function.
        optimizer: optimizer object.
        writer: tensorboard writer.
        epoch: epoch number.
        device: device of model.
    """

    # define metric lists
    losses: list[float] = []
    accuracies: list[float] = []

    model.train()
    for batch, targets in tqdm(train_data, desc="batches", position=1, leave=False):
        optimizer.zero_grad()

        batch = batch.to(device)
        targets = targets.to(device)

        outputs = model(batch)

        loss_val = loss(outputs, targets, model)
        loss_val.backward()
        optimizer.step()
        losses.append(loss_val.item())
        accuracies.append(accuracy(outputs, targets).item())

    # write on tensorboard
    writer.add_scalar("train/loss", np.mean(losses), epoch)
    writer.add_scalar("train/accuracy", np.mean(accuracies), epoch)


def val_step(
    model: torch.nn.Module,
    val_data: DataLoader,
    loss: torch.nn.Module,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    """
    This function computes the validation step.

    Args:
        model: pytorch model.
        val_data: dataloader of validation data.
        loss: loss function.
        writer: tensorboard writer.
        epoch: epoch number.
        device: device of model.
    """

    running_loss: float = 0.0
    running_accuracy: float = 0.0
    model.eval()
    with torch.no_grad():
        for batch, targets in tqdm(val_data, desc="batches", position=1, leave=False):
            batch = batch.to(device)
            targets = targets.to(device)

            outputs = model(batch)

            loss_val = loss(outputs, targets, model)

            running_loss += loss_val.item()
            running_accuracy += accuracy(outputs, targets).item()

    avg_loss = running_loss / len(val_data)
    avg_accuracy = running_accuracy / len(val_data)

    writer.add_scalar("val/loss", avg_loss, epoch)
    writer.add_scalar("val/accuracy", avg_accuracy, epoch)


def t_step(
    model: torch.nn.Module,
    test_data: DataLoader,
    device: torch.device,
) -> float:
    """
    This function computes the test step.

    Args:
        model: pytorch model.
        val_data: dataloader of test data.
        device: device of model.

    Returns:
        average accuracy.
    """

    model.eval()
    running_accuracy: float = 0.0
    with torch.no_grad():
        for batch, targets in tqdm(test_data, desc="batches", position=1, leave=False):
            batch = batch.to(device)
            targets = targets.to(device)

            outputs = model(batch)
            running_accuracy += accuracy(outputs, targets).item()

    avg_accuracy = running_accuracy / len(test_data)

    return avg_accuracy
