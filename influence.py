# pyright: strict

import copy
import csv
import os
from functools import partial

import torch
import torch.nn.functional as F
import torchvision.datasets
import torchvision.transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

import learning
from pytorch_influence_functions import calc_influence_function as calc_if
from pytorch_influence_functions import utils as ptif_utils
from utils import *


def calc_grad(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    trg: tuple[torch.Tensor, torch.Tensor],
) -> list[torch.Tensor]:

    x, y = trg
    model.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()

    grad: list[torch.Tensor] = []
    for param in model.parameters():
        if param.grad is None:
            raise ValueError("param.grad is None")
        grad.append(torch.flatten(param.grad))
    return grad


def calc_cos_sim(
    grad1: list[torch.Tensor], grad2: list[torch.Tensor]
) -> float:
    sum_cos_sim = 0
    for g1, g2 in zip(grad1, grad2):
        sum_cos_sim += F.cosine_similarity(g1, g2, dim=0).item()
    return sum_cos_sim / len(grad1)


def binary_search(length: int, to_left: Callable[[int], bool]) -> int:
    """Search first index where to_left is true

    Args:
        length (int): length of the list
        to_left (Callable[[int], bool]): whether to search left or right

    Returns:
        int: first index where to_left is true
    """
    left = 0
    right = length
    while left < right:
        mid = (left + right) // 2
        if to_left(mid):
            right = mid
        else:
            left = mid + 1
    return left


def load_or_calc_cos_sims(
    trg_idx: int,
    grad: Callable[[int], list[torch.Tensor]],
    train_list: list[tuple[torch.Tensor, torch.Tensor]],
    dataset_str: str,
) -> list[float]:
    filename = f"output/{dataset_str}/cos-sims-{trg_idx:03d}.csv"
    cos_sims: list[float] = []
    try:
        with open(filename, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                cos_sims = list(map(float, row))
            return cos_sims
    except FileNotFoundError:
        trg_grad = grad(trg_idx)
        for i in tqdm(range(len(train_list))):
            cos_sims.append(calc_cos_sim(grad(i), trg_grad))
        with open(filename, "w") as f:
            writer = csv.writer(f)
            writer.writerow(cos_sims)
    return cos_sims


def main():

    # if len(sys.argv) < 2:
    #     print("Usage: experiment.py <dataset_str>")
    #     sys.exit(1)

    # Load dataset
    dataset_str = "cifar10"
    learner = learning.Learner(dataset_str)
    train_list, test_list = learner.get_data()

    # Make output directory if it doesn't exist
    output_dir = f"output/{dataset_str}"
    os.makedirs(output_dir, exist_ok=True)

    print("Size of training data: ", len(train_list))
    print("Size of test     data: ", len(test_list))

    # train model
    model = load_or_train(
        f"{output_dir}/original-model",
        learner.device,
        partial(learner.train_model, remove_index=None),
        learner.new_model,
    )

    # Get predictions for test data
    test_pred = load_or_predict(
        f"{output_dir}/test-predictions",
        learner.device,
        model,
        test_list,
    )
    original_acc = test_accuracy(model, test_list)

    # Get misclassified samples in test data
    mis_idxs = [
        i
        for i, (_, y) in enumerate(test_list)
        if pred_to_label(test_pred[i]) != y
    ]
    print(f"Found {len(mis_idxs)} misclassified samples in test data.")

    updated_accs: list[float] = []
    new_retrain_accs: list[float] = []

    best_sizes: list[float] = []

    for i, mis_idx in enumerate(mis_idxs):

        x, y = test_list[mis_idx]

        def learn_clone() -> torch.nn.Module:
            clone = copy.deepcopy(model)
            while clone(x).argmax(1) != y:
                clone_pred = clone(x)
                loss = torch.nn.NLLLoss()(clone_pred, y)
                optimizer = torch.optim.Adam(clone.parameters(), lr=5e-3)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            return clone

        # clone model learns target sample until it is correctly classified
        print("Training additionally...")
        updated = load_or_train(
            f"{output_dir}/updated-model-{i:03d}",
            learner.device,
            learn_clone,
            learner.new_model,
        )

        print("Saving images...")
        save_image(
            f"{output_dir}/target-sample-{i:03d}.png",
            (x, y),
            model(x).argmax(1).item(),
            dataset_str,
        )

        MAX_SIZE = 100

        def retrain_model_by_gradient() -> torch.nn.Module:

            # If cosine similarities are already calculated, load them
            # Otherwise calculate and save them
            cos_sims: list[float] = load_or_calc_cos_sims(
                mis_idx,
                lambda i: calc_grad(model, learner.loss_fn, train_list[i]),
                train_list,
                dataset_str,
            )
            # Sort training samples based on the cosine similarities in
            # ascending order
            sorted_idxs: list[int]
            _, sorted_idxs = zip(*sorted(zip(cos_sims, range(len(cos_sims)))))
            for j, idx in enumerate(sorted_idxs[:10]):
                trg_x, trg_y = train_list[idx]
                save_image(
                    f"{output_dir}/grad-{i:03d}-{j:03d}.png",
                    (trg_x, trg_y),
                    model(trg_x).argmax(1).item(),
                    dataset_str,
                )

            # Get minimum size of subset of training dataset that need to be
            # removed to predict the right label for the target sample
            best_size = binary_search(
                min(MAX_SIZE, len(sorted_idxs)),
                lambda i: learner.train_model(sorted_idxs[:i])(x).argmax(1)
                == y,
            )
            print(f"We found {best_size} samples needed!")
            best_sizes.append(float(best_size))

            # Save best_sizes in a text file
            with open(f"{output_dir}/best-idx-{i:03d}.txt", "w") as f:
                f.write(str(best_size))

            return learner.train_model(sorted_idxs[:best_size])

        def new_model() -> torch.nn.Module:
            with open(f"{output_dir}/best-idx-{i:03d}.txt") as f:
                best_size = int(f.read())
                best_sizes.append(float(best_size))
            return learner.new_model()

        print("Gradient-based retraining...")
        new_retrain = load_or_train(
            f"{output_dir}/new-retrain-{i:03d}",
            learner.device,
            retrain_model_by_gradient,
            new_model,
        )

        batch_size = 4

        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                ),
            ]
        )

        trainset = torchvision.datasets.CIFAR10(
            root="data",
            train=True,
            download=True,
            transform=transform,
        )

        trainloader: DataLoader[tuple[torch.Tensor, int]] = DataLoader(
            trainset, batch_size=batch_size, shuffle=False
        )

        testset = torchvision.datasets.CIFAR10(
            root="data",
            train=False,
            download=True,
            transform=transform,
        )

        testloader: DataLoader[tuple[torch.Tensor, int]] = DataLoader(
            testset, batch_size=batch_size, shuffle=False
        )

        def retrain_model_by_influence() -> torch.nn.Module:
            ptif_utils.init_logging()
            config = ptif_utils.get_default_config()
            config["gpu"] = -1
            print(config)
            influences, _, _, _ = calc_if.calc_influence_single(
                model,
                trainloader,
                testloader,
                test_id_num=mis_idx,
                gpu=-1,
                recursion_depth=config["recursion_depth"],
                r=config["r_averaging"],
            )

            harmful_indices = sorted(
                range(len(influences)), key=lambda i: influences[i]
            )

            # Get minimum size of subset of training dataset that need to be
            # removed to predict the right label for the target sample
            best_size = binary_search(
                min(MAX_SIZE, len(harmful_indices)),
                lambda i: learner.train_model(harmful_indices[:i])(x).argmax(1)
                == y,
            )
            print(f"We found {best_size} samples needed!")
            best_sizes.append(float(best_size))

            # Save best_sizes in a text file
            with open(
                f"output/{dataset_str}/influence-best-idx-{i:03d}.txt", "w"
            ) as f:
                f.write(str(best_size))

            return learner.train_model(harmful_indices[:best_size])

        print("Influence-based retraining...")
        new_retrain = load_or_train(
            f"output/{dataset_str}/influence-{i:03d}",
            learner.device,
            retrain_model_by_influence,
            new_model,
        )

        print("Predictions:")
        print(f" True Label:      {y.item()}")
        print(f" Original model:  {model(x).argmax(1).item()} ({model(x)})")
        print(f" Updated model:   {pred_to_label(updated(x))} ({updated(x)})")
        print(
            f" Retrained model: {pred_to_label(new_retrain(x))} ({new_retrain(x)})"
        )

        updated_acc = test_accuracy(updated, test_list)
        new_retrain_acc = test_accuracy(new_retrain, test_list)
        print("Test accuracies:")
        print(f" Original model:  {original_acc}")
        print(f" Updated model:   {updated_acc}")
        print(f" Retrained model: {new_retrain_acc}")
        updated_accs.append(updated_acc)
        new_retrain_accs.append(new_retrain_acc)
        clean_accs = [
            acc for n, acc in zip(best_sizes, new_retrain_accs) if n < MAX_SIZE
        ]
        clean_idxs = [n for n in best_sizes if n < MAX_SIZE]
        if clean_idxs:
            print(
                f"Updated accuracy: Avg: {sum(updated_accs)/len(updated_accs)}, "
                f"std: {torch.std(torch.tensor(updated_accs))}"
            )
            print(
                f"New Retrain accuracy: Avg: {sum(clean_accs)/len(clean_accs)}, "
                f"std: {torch.std(torch.tensor(clean_accs))}"
            )
            print(
                f"Best indices: Avg: {sum(clean_idxs) / len(clean_idxs)}, "
                f"std: {torch.std(torch.tensor(clean_idxs))}, "
                f"Max: {max(clean_idxs)}, Min: {min(clean_idxs)}"
            )
        print(f"Success rate: {len(clean_accs) / len(best_sizes)}")
        print(sorted(best_sizes))


if __name__ == "__main__":
    main()
