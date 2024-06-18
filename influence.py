# pyright: strict

import copy
import os
from statistics import mean, stdev

import torch
import torchvision.datasets
import torchvision.transforms
from torch.utils.data import DataLoader

import learning
from grad_retrain import calc_grad, load_or_calc_cos_sims
from pytorch_influence_functions import calc_influence_function as calc_if
from pytorch_influence_functions import utils as ptif_utils
from utils import *


def retrain_model(
    sort_fn: Callable[[], list[int]],
    trg: tuple[torch.Tensor, torch.Tensor],
    learner: learning.Learner,
    max_size: int,
) -> tuple[torch.nn.Module, int] | None:

    # Sort training samples based on the given sort function
    sorted_idxs = sort_fn()
    max_size = min(max_size, len(sorted_idxs))

    # Retrain the model removing the top max_size samples and return
    # None if the target sample is not correctly predicted
    best_model = learner.train_model(sorted_idxs[:max_size])
    if best_model(trg[0]).argmax(1).item() != trg[1]:
        return None

    # Binary search for minimum size of subset of training dataset that
    # need to be removed to correctly predict the target sample
    left = 0
    right = max_size
    while left < right:
        mid = (left + right) // 2
        model = learner.train_model(sorted_idxs[:mid])
        if model(trg[0]).argmax(1).item() == trg[1]:
            right = mid
            best_model = model
        else:
            left = mid + 1

    return best_model, right


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

    print(f"Size of training data: {len(train_list)}")
    print(f"Size of test data:     {len(test_list)}")

    # Train an original model
    result = load_or_train(
        lambda: (learner.train_model(None), len(train_list)),
        learner.new_model,
        f"{output_dir}/original-model-{learner.device}.pth",
    )
    if result is None:
        raise ValueError("Original model could not be trained.")
    orig_model, _ = result

    # Get predictions for test data
    test_pred = load_or_predict(
        f"{output_dir}/test-predictions",
        learner.device,
        orig_model,
        test_list,
    )
    original_acc = test_accuracy(orig_model, test_list)

    # Get misclassified samples in test data
    mis_idxs = [
        i
        for i, (_, y) in enumerate(test_list)
        if pred_to_label(test_pred[i]) != y
    ]
    print(f"Found {len(mis_idxs)} misclassified samples in test data.")

    updated_accs: list[float] = []
    grad_accs: list[float] = []
    infl_accs: list[float] = []

    grad_best_sizes: list[float] = []
    infl_best_sizes: list[float] = []

    itr_idxs = mis_idxs[: min(1000, len(mis_idxs))]
    for i, mis_idx in enumerate(itr_idxs):

        x, y = test_list[mis_idx]

        def learn_clone() -> tuple[torch.nn.Module, int]:
            clone = copy.deepcopy(orig_model)
            while clone(x).argmax(1) != y:
                clone_pred = clone(x)
                loss = torch.nn.NLLLoss()(clone_pred, y)
                optimizer = torch.optim.Adam(clone.parameters(), lr=5e-3)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            return clone, len(train_list)

        # clone model learns target sample until it is correctly classified
        print("Training additionally...")
        result = load_or_train(
            learn_clone,
            learner.new_model,
            f"{output_dir}/updated-model-{i:03d}-{learner.device}.pth",
        )
        if result is None:
            raise ValueError("Updated model could not be trained.")
        updated, _ = result

        print("Saving images...")
        save_image(
            f"{output_dir}/target-sample-{i:03d}.png",
            (x, y),
            orig_model(x).argmax(1).item(),
            dataset_str,
        )

        MAX_SIZE = 100

        def sort_by_grad() -> list[int]:
            # If cosine similarities are already calculated, load them
            # Otherwise calculate and save them
            cos_sims: list[float] = load_or_calc_cos_sims(
                mis_idx,
                lambda i: calc_grad(
                    orig_model, learner.loss_fn, train_list[i]
                ),
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
                    orig_model(trg_x).argmax(1).item(),
                    dataset_str,
                )
            return sorted_idxs

        def retrain_model_by_gradient() -> tuple[torch.nn.Module, int] | None:
            return retrain_model(sort_by_grad, (x, y), learner, MAX_SIZE)

        print("Gradient-based retraining...")
        result_grad = load_or_train(
            retrain_model_by_gradient,
            learner.new_model,
            f"{output_dir}/new-retrain-{i:03d}-{learner.device}.pth",
            f"{output_dir}/grad-best-idx-{i:03d}.txt",
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
            influences, _, _, _ = calc_if.calc_influence_single(
                orig_model,
                trainloader,
                testloader,
                test_id_num=mis_idx,
                gpu=config["gpu"],
                recursion_depth=config["recursion_depth"],
                r=config["r_averaging"],
            )
            return sorted(range(len(influences)), key=lambda i: influences[i])

        def retrain_model_by_influence() -> tuple[torch.nn.Module, int] | None:
            return retrain_model(sort_by_influence, (x, y), learner, MAX_SIZE)

        print("Influence-based retraining...")
        result_infl = load_or_train(
            retrain_model_by_influence,
            learner.new_model,
            MAX_SIZE,
            f"{output_dir}/influence-{i:03d}-{learner.device}.pth",
            f"{output_dir}/infl-best-idx-{i:03d}.txt",
        )

        print("----------------------")

        def print_pred(model: torch.nn.Module) -> None:
            list_str = ""
            for i in model(x).squeeze().tolist():
                list_str += f"{i:.2f}, "
            print(f" {model(x).argmax(1).item()} ({list_str})")

        print("Predictions:")
        print(f" True Label:      {y.item()}")
        print(" Original model:  ", end="")
        print_pred(orig_model)
        print(" Updated model:   ", end="")
        print_pred(updated)
        print(" Gradient-based:  ", end="")
        if result_grad is None:
            print("Failed")
        else:
            print_pred(result_grad[0])
        print(" Influence-based: ", end="")
        if result_infl is None:
            print("Failed")
        else:
            print_pred(result_infl[0])
        print()

        print("Test Accuracies:")
        print(f" Original model:  {original_acc}")
        updated_acc = test_accuracy(updated, test_list)
        updated_accs.append(updated_acc)
        print(f" Updated model:   {updated_acc}")
        print(" Gradient-based:  ", end="")
        if result_grad is None:
            print("Failed")
        else:
            grad_model, grad_best_size = result_grad
            grad_acc = test_accuracy(grad_model, test_list)
            grad_accs.append(grad_acc)
            grad_best_sizes.append(grad_best_size)
            print(f"{grad_acc} (Size: {grad_best_size})")
        print(" Influence-based: ", end="")
        if result_infl is None:
            print("Failed")
        else:
            infl_model, infl_best_size = result_infl
            infl_acc = test_accuracy(infl_model, test_list)
            infl_accs.append(infl_acc)
            infl_best_sizes.append(infl_best_size)
            print(f"{infl_acc} (Size: {infl_best_size})")
        print()

        def my_stdev(nums: list[float]) -> float | None:
            return stdev(nums) if len(nums) > 1 else None

        print("------")

        print("Test Accuracies:")
        print(" Original:")
        print(f"       {original_acc}")
        print(" Updated:")
        print(f"  Avg: {mean(updated_accs)}")
        print(f"  Std: {my_stdev(updated_accs)}")
        print(" Gradient-based:")
        print(f"  Avg: {mean(grad_accs)}")
        print(f"  Std: {my_stdev(grad_accs)}")
        print(" Influence-based:")
        print(f"  Avg: {mean(infl_accs)}")
        print(f"  Std: {my_stdev(infl_accs)}")
        print()

        print("Best Sizes:")
        print(" Gradient-based:")
        print(f"  Avg: {mean(grad_best_sizes)}")
        print(f"  Std: {my_stdev(grad_best_sizes)}")
        print(" Influence-based:")
        print(f"  Avg: {mean(infl_best_sizes)}")
        print(f"  Std: {my_stdev(infl_best_sizes)}")
        print()

        print("Success rates:")
        print(f" Gradient-based:  {len(grad_accs) / (i + 1)}")
        print(f" Influence-based: {len(infl_accs) / (i + 1)}")


if __name__ == "__main__":
    main()
