from itertools import zip_longest
from typing import Iterable

import torch
import torch.nn as nn


def zip_strict(*iterables: Iterable) -> Iterable:
    sentinel = object()
    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            raise ValueError("Iterables have different lengths")
        yield combo


def polyak_update(
    params: Iterable[nn.Parameter],
    target_params: Iterable[nn.Parameter],
    tau: float,
) -> None:
    """
    Polyak (soft) target network update.

    Inspired by implementation of stable-baselines3:
    https://github.com/DLR-RM/stable-baselines3/
    """
    with torch.no_grad():
        # Use `zip_strict`, since `zip` does not raise an excception if
        # lengths of parameter iterables differ.
        for param, target_param in zip_strict(params, target_params):
            target_param.data.mul_(1 - tau)
            torch.add(
                target_param.data, param.data, alpha=tau, out=target_param.data)