# deep learning libraries
import torch
from torch.utils.data import DataLoader
from torch.jit import RecursiveScriptModule

# other libraries
from tqdm.auto import tqdm
from typing import Final

# own modules
from src.utils import load_data, save_model
from src.models import BayesModel
from src.train_functions import t_step

# static variables
DATA_PATH: Final[str] = "data"
NUM_CLASSES: Final[int] = 10

# set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main() -> None:
    """
    This function is the main program.
    """

    # load data
    test_data: DataLoader
    _, _, test_data = load_data(DATA_PATH, batch_size=64)

    # define name and writer
    name: str = "run_shape_paper_normalized_loss"

    # define model
    model: RecursiveScriptModule = torch.jit.load(f"models/{name}.pt").to(device)

    # call test step and evaluate accuracy
    accuracy: float = t_step(model, test_data, device)
    print(f"accuracy: {accuracy}")

    return None


if __name__ == "__main__":
    main()
