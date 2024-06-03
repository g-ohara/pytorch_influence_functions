# pyright: strict

from typing import Callable

import matplotlib.pyplot as plt
import torch
import torch.utils.data
from torch.nn import Module


def load_or_train(
    filename: str,
    device: str,
    train_model: Callable[[], Module],
    new_model: Callable[[], Module],
) -> Module:
    """Load the trained model if it exists, otherwise train a new one.

    Args:
        filename (str): The name of the file to save the model to.
        device (str): The device to use for training ("cpu" or "cuda").
        train_model (Callable[[], Module]): The function to train the
            model called if the trained model does not exist.
        new_model (Callable[[], Module]): The function to create the
            new model called if the trained model exists.

    Returns:
        Module: The trained model.
    """
    filename += f"-{device}.pth"
    try:
        print("Loading model...")
        model = new_model()
        model.load_state_dict(torch.load(filename))  # type: ignore
        print("Model loaded.")
        return model
    except FileNotFoundError:
        print("Model not found. Learning model...")
        model = train_model()
        torch.save(model.state_dict(), filename)  # type: ignore
        print("Model learned.")
        return model


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
