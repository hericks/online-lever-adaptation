from itertools import combinations
import matplotlib
import matplotlib.pyplot as plt

import torch
import numpy as np
import numpy.ma as ma

from levers.helpers.helpers import generate_binary_patterns


def plot_comparison(ax, s1, s2, title=None, xlabel=None, ylabel=None, c=None):
    """Plot scores `s1` and `s2` against each other."""
    map = matplotlib.cm.get_cmap("Dark2")

    ax.plot([0, 1], [0, 1], transform=ax.transAxes)
    ax.scatter(s1, s2, s=3, c=c)

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


if __name__ == "__main__":
    # Load table with results
    drqn_results = torch.load("results/drqn-results.pickle")
    drqn_returns = drqn_results["return"]
    print(f"DRQN SHAPE: {drqn_returns.shape}")

    odql_results = torch.load("results/odql-results.pickle")
    odql_returns = odql_results["return"]
    odql_greedy_returns = odql_results["greedy_return"]
    odql_n_greedy_steps = odql_results["n_greedy_steps"]
    print(f"ODQL SHAPE: {odql_returns.shape}")

    # Create arrays for plotting
    drqn_rel_returns = drqn_returns.mean((0, 1)) / 100
    odql_rel_returns = odql_returns.mean((0, 1)) / 100
    odql_rel_greedy_returns = (odql_greedy_returns / odql_n_greedy_steps).mean(
        (0, 1)
    )

    # Create masked arrays for plotting
    color_mask = np.zeros((8, 70))
    train_mask = np.full((8, 70), False, dtype=bool)
    patterns = generate_binary_patterns(3)
    for tps_id, train_patterns in enumerate(combinations(patterns, 4)):
        for ep_id, eval_pattern in enumerate(patterns):
            train_mask[ep_id, tps_id] = eval_pattern in train_patterns
            color_mask[ep_id, tps_id] = ep_id

    drqn_masked_rel_returns = drqn_rel_returns[train_mask]
    odql_masked_rel_returns = odql_rel_returns[train_mask]
    odql_masked_rel_greedy_returns = odql_rel_greedy_returns[train_mask]

    # Create canvas
    figure, axis = plt.subplots(3, 2, figsize=(10, 16))

    # Full dataset
    plot_comparison(
        axis[0, 0],
        drqn_rel_returns,
        odql_rel_returns,
        "DRQN vs ODQL",
        "DRQN Score",
        "ODQL Score",
        color_mask,
    )
    plot_comparison(
        axis[0, 1],
        drqn_rel_returns,
        odql_rel_greedy_returns,
        "DRQN vs (greedy) ODQL",
        "DRQN Score",
        "ODQL Score",
        color_mask,
    )

    # Training
    plot_comparison(
        axis[1, 0],
        drqn_rel_returns[train_mask],
        odql_rel_returns[train_mask],
        "DRQN vs ODQL with train partners",
        "DRQN Score",
        "ODQL Score",
        color_mask[train_mask],
    )
    plot_comparison(
        axis[1, 1],
        drqn_rel_returns[train_mask],
        odql_rel_greedy_returns[train_mask],
        "DRQN vs (greedy) ODQL with train partners",
        "DRQN Score",
        "ODQL Score",
        color_mask[train_mask],
    )

    # Testing
    plot_comparison(
        axis[2, 0],
        drqn_rel_returns[~train_mask],
        odql_rel_returns[~train_mask],
        "DRQN vs ODQL with test partners",
        "DRQN Score",
        "ODQL Score",
        color_mask[~train_mask],
    )
    plot_comparison(
        axis[2, 1],
        drqn_rel_returns[~train_mask],
        odql_rel_greedy_returns[~train_mask],
        "DRQN vs (greedy) ODQL with test partners",
        "DRQN Score",
        "ODQL Score",
        color_mask[~train_mask],
    )

    # Save figure to disk
    figure.savefig("results/joint-results.png")
    plt.close()
