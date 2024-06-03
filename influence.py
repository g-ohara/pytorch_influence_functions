# pyright: strict

import copy
import os
from functools import partial

import torch
import torchvision.datasets
import torchvision.transforms
from torch.utils.data import DataLoader

import learning
from grad_retrain import calc_grad, load_or_calc_cos_sims
from pytorch_influence_functions import calc_influence_function as calc_if
from pytorch_influence_functions import utils as ptif_utils
from utils import *


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


def opt_subset(
    sorted_idxs: list[int],
    trg: tuple[torch.Tensor, torch.Tensor],
    learner: learning.Learner,
    filename: str,
    max_size: int,
) -> int:

    # Get minimum size of subset of training dataset that need to be
    # removed to predict the right label for the target sample
    x, y = trg
    best_size = binary_search(
        min(max_size, len(sorted_idxs)),
        lambda i: learner.train_model(sorted_idxs[:i])(x).argmax(1) == y,
    )

    # Print and save the subset size into a file
    print(f"We found {best_size} samples needed!")
    with open(filename, "w") as f:
        f.write(str(best_size))

    return best_size


def retrain_model(
    sort_fn: Callable[[], list[int]],
    trg: tuple[torch.Tensor, torch.Tensor],
    learner: learning.Learner,
    filename: str,
    max_size: int,
) -> torch.nn.Module:

    sorted_idxs = sort_fn()
    best_size = opt_subset(sorted_idxs, trg, learner, filename, max_size)
    return learner.train_model(sorted_idxs[:best_size])


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
    infl_accs: list[float] = []

    best_sizes: list[float] = []
    infl_best_sizes: list[float] = []

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

        def sort_by_grad() -> list[int]:
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
            return sorted_idxs

        def retrain_model_by_gradient() -> torch.nn.Module:

            sorted_idxs = sort_by_grad()
            best_size = opt_subset(
                sorted_idxs,
                (x, y),
                learner,
                f"{output_dir}/grad-best-idx-{i:03d}.txt",
                MAX_SIZE,
            )
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

        def sort_by_influence() -> list[int]:
            ptif_utils.init_logging()
            config = ptif_utils.get_default_config()
            config["gpu"] = 0 if learner.device == "cuda" else -1
            print(config)
            influences, _, _, _ = calc_if.calc_influence_single(
                model,
                trainloader,
                testloader,
                test_id_num=mis_idx,
                gpu=config["gpu"],
                recursion_depth=config["recursion_depth"],
                r=config["r_averaging"],
            )
            return sorted(range(len(influences)), key=lambda i: influences[i])

        def retrain_model_by_influence() -> torch.nn.Module:

            sorted_idxs = sort_by_influence()
            best_size = opt_subset(
                sorted_idxs,
                (x, y),
                learner,
                f"{output_dir}/infl-best-idx-{i:03d}.txt",
                MAX_SIZE,
            )
            return learner.train_model(sorted_idxs[:best_size])

        def new_influence_model() -> torch.nn.Module:
            with open(f"{output_dir}/infl-best-idx-{i:03d}.txt") as f:
                best_size = int(f.read())
                best_sizes.append(float(best_size))
            return learner.new_model()

        print("Influence-based retraining...")
        infl_retrain = load_or_train(
            f"output/{dataset_str}/influence-{i:03d}",
            learner.device,
            retrain_model_by_influence,
            new_influence_model,
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
        infl_acc = test_accuracy(infl_retrain, test_list)
        print("Test accuracies:")
        print(f" Original model:  {original_acc}")
        print(f" Updated model:   {updated_acc}")
        print(f" Gradient-based:  {new_retrain_acc}")
        print(f" Influence-based: {infl_acc}")
        updated_accs.append(updated_acc)
        new_retrain_accs.append(new_retrain_acc)
        infl_accs.append(infl_acc)
        clean_accs = [
            acc for n, acc in zip(best_sizes, new_retrain_accs) if n < MAX_SIZE
        ]
        clean_idxs = [n for n in best_sizes if n < MAX_SIZE]
        clean_infl_accs = [
            acc for n, acc in zip(infl_best_sizes, infl_accs) if n < MAX_SIZE
        ]
        clean_infl_idxs = [n for n in infl_best_sizes if n < MAX_SIZE]
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
                f"Influence-based accuracy: Avg: {sum(infl_accs)/len(infl_accs)}, "
                f"std: {torch.std(torch.tensor(infl_accs))}"
            )
            print("Best indices")
            print(" Gradient-based")
            print(
                f"  Avg: {sum(clean_idxs) / len(clean_idxs)}, "
                f"  std: {torch.std(torch.tensor(clean_idxs))}, "
                f"  Max: {max(clean_idxs)}, Min: {min(clean_idxs)}"
            )
            print(" Influence-based")
            print(
                f"  Avg: {sum(clean_infl_idxs) / len(clean_infl_idxs)}, "
                f"  std: {torch.std(torch.tensor(clean_infl_idxs))}, "
                f"  Max: {max(clean_infl_idxs)}, Min: {min(clean_infl_idxs)}"
            )
        print("Success rate")
        print(f" Gradient-based:  {len(clean_accs) / len(best_sizes)}")
        print(f" Influence-based: {len(clean_infl_accs) / len(clean_idxs)}")
        print(sorted(best_sizes))


if __name__ == "__main__":
    main()
