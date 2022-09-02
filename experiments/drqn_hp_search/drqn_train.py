from copy import deepcopy
from itertools import combinations
from typing import List
from os import path

import wandb
import random
import numpy as np

import torch
import torch.nn as nn

from levers import IteratedLeverEnvironment
from levers.partners import FixedPatternPartner
from levers.learner import DRQNAgent, DRQNetwork
from levers.helpers import generate_binary_patterns


def get_train_test_split(patterns):
    ret = []
    n_train_elements = len(patterns) - 1
    for train_patterns in combinations(patterns, n_train_elements):
        test_pattern = [p for p in patterns if p not in train_patterns][0]
        ret.append((train_patterns, test_pattern))
    return ret


def get_stats_dict(
    losses: torch.Tensor,
    train_returns: torch.Tensor,
    test_returns: torch.Tensor,
    train_test_splits: List,
):
    stats = {
        "mean_loss": losses.mean(),
        "sd_loss": losses.std(),
        "mean_train_return": train_returns.mean(),
        "sd_train_return": train_returns.std(),
        "mean_test_return": test_returns.mean(),
        "sd_test_return": test_returns.std(),
    }

    for sid, (train_patterns, test_pattern) in enumerate(train_test_splits):
        stats[
            f"mean_loss_split_train={train_patterns}_test={test_pattern}"
        ] = losses[:, sid].mean()
        stats[
            f"sd_loss_split_train={train_patterns}_test={test_pattern}"
        ] = losses[:, sid].std()
        stats[
            f"mean_train_return_split_train={train_patterns}_test={test_pattern}"
        ] = train_returns[:, sid].mean()
        stats[
            f"sd_train_return_split_train={train_patterns}_test={test_pattern}"
        ] = train_returns[:, sid].std()
        stats[
            f"mean_test_return_split_train={train_patterns}_test={test_pattern}"
        ] = test_returns[:, sid].mean()
        stats[
            f"sd_test_return_split_train={train_patterns}_test={test_pattern}"
        ] = test_returns[:, sid].std()

    return stats


def train():
    wandb.init()
    config = wandb.config
    wandb.run.name = f"lr={config.learning_rate:5.4f}-tau={config.tau:5.4f}-T={config.len_update_cycle}-cap={config.capacity}-bs={config.batch_size}"
    # wandb.run.name = wandb.run.id

    # (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)
    # (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)
    patterns = [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0)]

    dummy_env = IteratedLeverEnvironment(
        payoffs=[1.0, 1.0],
        n_iterations=100 + 1,
        partner=FixedPatternPartner(patterns[0]),
        include_step=False,
        include_payoffs=False,
    )
    # Create table with results
    train_test_splits = get_train_test_split(patterns)
    n_splits = len(train_test_splits)

    learners = [
        [
            DRQNAgent(
                q_net=DRQNetwork(
                    rnn=nn.LSTM(
                        input_size=len(dummy_env.dummy_obs()),
                        hidden_size=config.rnn_hidden_dim,
                        batch_first=True,
                    ),
                    fnn=nn.Sequential(
                        nn.Linear(
                            in_features=config.rnn_hidden_dim,
                            out_features=config.fnn_hidden_dim,
                        ),
                        nn.ReLU(),
                        nn.Linear(
                            in_features=config.fnn_hidden_dim,
                            out_features=dummy_env.n_actions(),
                        ),
                    ),
                ),
                capacity=config.capacity,
                batch_size=min(config.batch_size, config.capacity),
                lr=config.learning_rate,
                tau=config.tau,
                gamma=config.gamma,
                len_update_cycle=config.len_update_cycle,
            )
            for _ in range(n_splits)
        ]
        for _ in range(config.n_train_evals)
    ]

    epsilon = config.eps_start
    for episode in range(config.n_episodes):

        # Setup results tensors
        epi_losses = torch.zeros((config.n_train_evals, n_splits))
        epi_train_returns = torch.zeros_like(epi_losses)
        epi_test_returns = torch.zeros_like(epi_losses)

        for split_id, (train_patterns, test_pattern) in enumerate(
            train_test_splits
        ):

            # Setup training environments
            train_envs = [
                IteratedLeverEnvironment(
                    payoffs=[1.0, 1.0],
                    n_iterations=100 + 1,
                    partner=FixedPatternPartner(pattern),
                    include_step=False,
                    include_payoffs=False,
                )
                for pattern in train_patterns
            ]

            # Setup test environment
            test_env = deepcopy(train_envs[0])
            test_env.partner = FixedPatternPartner(test_pattern)

            for train_id in range(config.n_train_evals):

                # Obtain agent
                learner = learners[train_id][split_id]

                # Train agent on randomly samples training environment
                env = random.sample(train_envs, 1)[0]
                obs = env.reset()
                learner.reset_new_episode(init_obs=obs)
                for _ in range(env.episode_length - 1):
                    action = learner.act(obs, epsilon)
                    next_obs, reward, done = env.step(action)
                    learner.update_trajectory_buffer(
                        action, reward, next_obs, done
                    )
                    obs = next_obs
                    epi_train_returns[train_id, split_id] += reward

                # Update learner
                learner.flush_trajectory_buffer()
                loss = learner.train()
                epi_losses[train_id, split_id] = loss if loss is not None else 1

                # Test agent on test environment
                env = test_env
                obs = env.reset()
                learner.reset_new_episode(init_obs=obs)
                for _ in range(env.episode_length - 1):
                    action = learner.act(obs)
                    next_obs, reward, done = env.step(action)
                    obs = next_obs
                    epi_test_returns[train_id, split_id] += reward

        # Update epsilon
        epsilon = max(epsilon - config.eps_diff, config.eps_min)

        # Log current episode's stats
        stats = get_stats_dict(
            epi_losses, epi_train_returns, epi_test_returns, train_test_splits
        )
        wandb.log(stats)


if __name__ == "__main__":
    train()
