from os import path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch

from hist_rep import get_parser


if __name__ == "__main__":
    # Load configuration
    default_config_files = [
        path.join(
            "/data/engs-oxfair3/orie4536/online-lever-adaptation/",
            "experiments",
            "history_representation",
            "defaults.conf",
        ),
    ]
    p = get_parser(default_config_files)
    opt = p.parse_args()

    pattern = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
    n_train_evals = 50

    reg_train_stats = np.zeros((n_train_evals, opt.n_epochs))
    rand_train_stats = np.zeros((n_train_evals, opt.n_epochs))
    for tid in range(n_train_evals):
        reg_train_stats[tid, :] = torch.load(
            f"train_stats/ODQL-{pattern}-{tid}.pickle"
        )["mean_eval"]
        rand_train_stats[tid, :] = torch.load(
            f"train_stats/RANDOM-ODQL-{pattern}-{tid}.pickle"
        )["mean_eval"]

    reg_mean = reg_train_stats.mean(axis=0)
    reg_std = reg_train_stats.std(axis=0)

    rand_mean = rand_train_stats.mean(axis=0)
    rand_std = rand_train_stats.std(axis=0)

    plt.plot(np.arange(opt.n_epochs), reg_mean)
    plt.fill_between(
        np.arange(opt.n_epochs),
        reg_mean - reg_std,
        reg_mean + reg_std,
        alpha=0.25,
    )
    plt.plot(np.arange(opt.n_epochs), rand_mean)
    plt.fill_between(
        np.arange(opt.n_epochs),
        rand_mean - rand_std,
        rand_mean + rand_std,
        alpha=0.25,
    )
    plt.ylim(275, 350)
    plt.savefig(f"results/mean-eval-comparison-{pattern}-{n_train_evals}.png")
    plt.close()
