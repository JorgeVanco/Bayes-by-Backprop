# deep learning libraries
import torch
import numpy as np
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# own modules
from src.utils import accuracy


def loss_function(inputs, targets, model, kl_weight) -> torch.Tensor:
    kl_loss = model.log_p_weights() - model.log_prior()
    return kl_weight * kl_loss + cross_entropy(inputs, targets)


def train_step(
    model: torch.nn.Module,
    train_data: DataLoader,
    loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
    kl_reweighting: bool = False,
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
    M: int = len(train_data)
    pi_i: float = 1.0 / M
    reweighting_denominator: int = 2**M - 1
    model.train()
    for batch_idx, (batch, targets) in enumerate(
        tqdm(train_data, desc="batches", position=1, leave=False)
    ):
        optimizer.zero_grad()

        batch = batch.to(device)
        targets = targets.to(device)

        outputs = model(batch)

        if kl_reweighting:
            pi_i = (2 ** (M - batch_idx + 1)) / reweighting_denominator
        loss_val = loss(outputs, targets, model, pi_i)
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
    pi_i: float = 1.0 / len(val_data)
    model.eval()
    with torch.no_grad():
        for batch, targets in tqdm(val_data, desc="batches", position=1, leave=False):
            batch = batch.to(device)
            targets = targets.to(device)

            outputs = model(batch)

            loss_val = loss(outputs, targets, model, pi_i)

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
