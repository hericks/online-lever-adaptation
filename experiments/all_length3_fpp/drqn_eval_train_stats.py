from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import torch
import os

from levers.helpers import generate_binary_patterns


def save_smooth_loss_curve(losses: np.array, out_path: str):
    episodes = np.arange(losses.shape[1])
    plt.plot(episodes, losses.mean(0))
    plt.fill_between(
        episodes,
        losses.mean(0) - losses.std(0),
        losses.mean(0) + losses.std(0),
        alpha=0.25,
    )
    plt.ylim(-0.025, 0.35)
    plt.savefig(out_path)
    plt.close()


def save_smooth_log_loss_curve(losses: np.array, out_path: str):
    episodes = np.arange(losses.shape[1])
    plt.plot(episodes, losses.mean(0))
    plt.semilogy()
    plt.savefig(out_path)
    plt.close()


def save_smooth_returns_curve(returns: np.array, out_path: str):
    episodes = np.arange(returns.shape[1])
    plt.plot(episodes, returns.mean(0))
    plt.fill_between(
        episodes,
        returns.mean(0) - returns.std(0),
        returns.mean(0) + returns.std(0),
        alpha=0.25,
    )
    plt.ylim(0, 100)
    plt.savefig(out_path)
    plt.close()


if __name__ == "__main__":
    # Settings
    n_train_evals = 50
    n_episodes = 2000

    # Save figures
    out_path = f"results/train_figures/drqn"
    losses_template = "DRQN-smooth-loss-{pattern}.png"
    log_losses_template = "DRQN-smooth-log-loss-{pattern}.png"
    returns_template = "DRQN-returns-{pattern}.png"

    patterns = generate_binary_patterns(3)
    for pattern_id, pattern in enumerate(combinations(patterns, 4)):
        print(f"Pattern: {pattern} | {pattern_id+1:2d} / 70")

        losses = np.zeros((n_train_evals, n_episodes))
        returns = np.zeros_like(losses)

        for train_id in range(n_train_evals):
            try:
                train_stats = torch.load(
                    f"train_stats/drqn/DRQN-{pattern}-{train_id}.pickle"
                )
                losses[train_id, :] = np.array(train_stats["loss"])
                returns[train_id, :] = np.array(train_stats["return"])
            except:
                print(
                    f"Something went wrong with reading in data for: {pattern}-{train_id}."
                )

        save_smooth_loss_curve(
            losses,
            os.path.join(out_path, losses_template.format(pattern=pattern)),
        )
        save_smooth_log_loss_curve(
            losses,
            os.path.join(out_path, log_losses_template.format(pattern=pattern)),
        )
        save_smooth_returns_curve(
            returns,
            os.path.join(out_path, returns_template.format(pattern=pattern)),
        )
