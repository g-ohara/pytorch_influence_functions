# pyright: strict

from typing import Callable

import matplotlib.pyplot as plt
import torch
import torch.utils.data
from torch.nn import Module


def load_or_train(
    train_model: Callable[[], tuple[Module, int] | None],
    new_model: Callable[[], Module],
    model_filename: str,
    size_filename: str | None = None,
    max_size: int | None = None,
) -> tuple[Module, int] | None:
    """Load the trained model if it exists, otherwise train a new one
    and save it to a file.

    Args:
        model_filename (str): The name of the file to save the model to.
        size_filename (str): The name of the file to save the best size
            of the subset to.
        device (str): The device to use for training ("cpu" or "cuda").
        train_model (Callable[[], Module]): The function to train the
            model called if the trained model does not exist.
        new_model (Callable[[], Module]): The function to create the
            new model called if the trained model exists.

    Returns:
        Module: The trained model.
    """
    try:

        # Load model
        print("Loading model...")
        model = new_model()
        model.load_state_dict(torch.load(model_filename))  # type: ignore
        print("Model loaded.")

        # Load the best size of the subset if size_filename is provided
        if size_filename is None:
            return model, 0
        else:
            with open(size_filename, "r") as f:
                best_size = int(f.read())
            if max_size is not None and best_size > max_size:
                raise ValueError("Loaded best size exceeds max size.")
            return model, best_size

    except FileNotFoundError:
        print("Model not found. Learning model...")

        # Train model if the trained model does not exist
        result = train_model()

        # Return None if training failed, otherwise save and return
        # trained model and the best size of the subset
        if result is None:
            return None
        else:
            model, best_size = result
            torch.save(model.state_dict(), model_filename)  # type: ignore
            if size_filename is not None:
                with open(size_filename, "w") as f:
                    f.write(str(best_size))
            print("Model learned.")
            return model, best_size


def load_or_predict(
    filename: str,
    device: str,
    model: torch.nn.Module,
    data_list: list[tuple[torch.Tensor, int]],
) -> torch.Tensor:
    filename += f"-{device}.pth"
    try:
        print("Loading predictions...")
        pred = torch.load(filename)  # type: ignore
        print("Predictions loaded.")
        return pred
    except FileNotFoundError:
        print("Predictions not found. Obtaining predictions...")
        pred = model(torch.stack([x[0].squeeze() for x in data_list]))
        torch.save(pred, filename)  # type: ignore
        print("Predictions obtained.")
        return pred


def pred_to_label(pred: torch.Tensor) -> int:
    if pred.ndim == 1:
        pred = pred.unsqueeze(0)
    return pred.argmax(1).item()  # type: ignore


def test_accuracy(
    model: torch.nn.Module, test_data: list[tuple[torch.Tensor, int]]
):
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_data:
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += 1
    return correct / total


def save_image(
    filename: str,
    trg: tuple[torch.Tensor, torch.Tensor],
    pred: int,
    dataset_str: str,
):
    if dataset_str == "mnist":
        labels_map = {
            0: "0",
            1: "1",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
        }
    elif dataset_str == "fashionmnist":
        labels_map = {
            0: "T-shirt/top",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle Boot",
        }
    elif dataset_str == "cifar10":
        labels_map = {
            0: "airplane",
            1: "automobile",
            2: "bird",
            3: "cat",
            4: "deer",
            5: "dog",
            6: "frog",
            7: "horse",
            8: "ship",
            9: "truck",
        }
    else:
        raise NotImplementedError

    _, ax = plt.subplots()  # type: ignore
    x, y = trg
    _x = x.squeeze()
    if _x.ndim == 3:
        _x = torch.swapaxes(_x, 0, 1).swapaxes(1, 2)
    ax.imshow(_x.cpu(), cmap="gray")
    true_label = labels_map[int(y.item())]
    pred_label = labels_map[pred]
    ax.set_title(
        f"True: {true_label}, Original Pred: {pred_label}", fontsize=18
    )
    plt.show()  # type: ignore
    plt.savefig(filename)  # type: ignore
    plt.close()
