# pyright: strict

import csv
from typing import Callable

import torch
import torch.nn.functional as F
from tqdm import tqdm


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
