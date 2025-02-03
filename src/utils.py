# deep learning libraries
import torch
import torchvision
import numpy as np
import pandas as pd
from torch.jit import RecursiveScriptModule
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# other libraries
import os
import random


def load_data(
    path: str, batch_size: int = 128
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    This function loads the data from mnist dataset. All batches must
    be equal size. The division between train and val must be 0.8-0.2.

    Args:
        path: path to save the datasets.
        batch_size: batch size. Defaults to 128.

    Returns:
        tuple of three dataloaders, train, val and test in respective order.
    """

    # define transforms
    transformations = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda img: img * 255.0 / 126.0)]
    )

    train_dataset: torchvision.datasets.MNIST = torchvision.datasets.MNIST(
        path, download=True, transform=transformations, train=True
    )
    test_dataset: torchvision.datasets.MNIST = torchvision.datasets.MNIST(
        path, download=True, transform=transformations, train=False
    )

    train_data, val_data = random_split(train_dataset, [0.8, 0.2])
    train_dataloader: DataLoader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_dataloader: DataLoader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )
    test_dataloader: DataLoader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )
    return train_dataloader, val_dataloader, test_dataloader


def save_model(model: torch.nn.Module, name: str) -> None:
    """
    This function saves a model in the 'models' folder as a torch.jit.
    It should create the 'models' if it doesn't already exist.

    Args:
        model: pytorch model.
        name: name of the model (without the extension, e.g. name.pt).
    """

    # create folder if it does not exist
    if not os.path.isdir("models"):
        os.makedirs("models")

    # save scripted model
    model_scripted = torch.jit.script(model.cpu())
    model_scripted.save(f"models/{name}.pt")

    return None


def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior

    Args:
        seed: seed number to fix radomness
    """

    # set seed in numpy and random
    np.random.seed(seed)
    random.seed(seed)

    # set seed and deterministic algorithms for torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Ensure all operations are deterministic on GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # for deterministic behavior on cuda >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return None


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    This function computes the accuracy.

    Args:
        predictions: predictions tensor. Dimensions:
            [batch, num classes] or [batch].
        targets: targets tensor. Dimensions: [batch, 1] or [batch].

    Returns:
        the accuracy in a tensor of a single element.
    """

    predictions = torch.argmax(predictions, dim=-1)
    return torch.sum(predictions == targets) / predictions.shape[0]
