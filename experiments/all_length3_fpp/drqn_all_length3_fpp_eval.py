import os

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np

import torch

from levers.helpers import generate_binary_patterns

from config import (
    train_stats_path,
    experiment_name,
    figs_path,
    model_name_template
)

#
n_train_evals = 25
n_episodes = 2000

patterns = generate_binary_patterns(3)
for pattern in combinations(patterns, 4):
    losses = np.zeros((n_train_evals, n_episodes))
    returns = np.zeros_like(losses)

    # Load stats
    for train_id in range(n_train_evals):
        try:
            model_name = model_name_template.format(
                partner_patterns=pattern, train_id=train_id)
            in_path = os.path.join(
                train_stats_path, experiment_name, model_name + '.pickle')
            train_stats = torch.load(in_path)
        except:
            break

        losses[train_id,:] = np.array(train_stats['loss'])
        returns[train_id,:] = np.array(train_stats['return'])

    if train_id == 0:
        continue

    losses = losses[~np.all(losses == 0, axis=1)]
    returns = returns[~np.all(returns == 0, axis=1)]

    # Save smooth losses
    out_path = os.path.join(figs_path, experiment_name)
    fig_name = f'0-DRQN-{pattern}-smooth-loss-{train_id}.png'
    plt.plot(train_stats['episode'], losses.mean(0))
    plt.fill_between(
        train_stats['episode'],
        losses.mean(0) - losses.std(0),
        losses.mean(0) + losses.std(0),
        alpha=0.25
    )
    plt.savefig(os.path.join(out_path, fig_name))
    plt.close()

    # Save log losses
    out_path = os.path.join(figs_path, experiment_name)
    fig_name = f'0-DRQN-{pattern}-smooth-log-loss-{train_id}.png'
    plt.plot(train_stats['episode'], losses.mean(0))
    plt.semilogy()
    plt.savefig(os.path.join(out_path, fig_name))
    plt.close()

    # Save smooth returns
    fig_name = f'0-DRQN-{pattern}-smooth-returns-{train_id}.png'
    plt.plot(train_stats['episode'], returns.mean(0))
    plt.fill_between(
        train_stats['episode'],
        returns.mean(0) - returns.std(0),
        returns.mean(0) + returns.std(0),
        alpha=0.25
    )
    plt.axhline(y=85, color='r', linestyle='--')
    plt.ylim((0, 100))
    plt.savefig(os.path.join(out_path, fig_name))
    plt.close()