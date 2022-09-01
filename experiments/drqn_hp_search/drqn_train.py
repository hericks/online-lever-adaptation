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


def train():
    wandb.init()
    config = wandb.config
    wandb.run.name = f"drqn-lr={config.learning_rate:5.4f}-tau={config.tau:5.4f}-len_update_cycle{config.len_update_cycle}"

    # (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)
    # (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)
    train_patterns = ((0, 0, 1), (1, 0, 1), (1, 1, 0), (1, 1, 1))
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

    learners = [
        DRQNAgent(
            q_net=DRQNetwork(
                rnn=nn.LSTM(
                    input_size=len(train_envs[0].dummy_obs()),
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
                        out_features=train_envs[0].n_actions(),
                    ),
                ),
            ),
            capacity=config.capacity,
            batch_size=config.batch_size,
            lr=config.learning_rate,
            tau=config.tau,
            gamma=config.gamma,
            len_update_cycle=config.len_update_cycle,
        )
        for _ in range(config.n_train_evals)
    ]

    epsilon = config.eps_start
    for episode in range(config.n_episodes):
        losses = torch.zeros(config.n_train_evals)
        epi_returns = torch.zeros(config.n_train_evals)
        for train_id in range(config.n_train_evals):
            # Sample and reset environment
            env = random.sample(train_envs, 1)[0]
            obs = env.reset()

            # Reset learner
            learners[train_id].reset_new_episode(init_obs=obs)

            # Reset episode stats
            epi_return = 0

            # Step through environment
            for step in range(env.episode_length - 1):
                action = learners[train_id].act(obs, epsilon)
                next_obs, reward, done = env.step(action)
                learners[train_id].update_trajectory_buffer(
                    action, reward, next_obs, done
                )
                obs = next_obs
                #
                epi_return += reward

            # Update learner
            learners[train_id].flush_trajectory_buffer()
            epsilon = max(epsilon - config.eps_diff, config.eps_min)
            loss = learners[train_id].train()

            # Update stats
            if loss is not None:
                losses[train_id] = loss
            else:
                losses[train_id] = 3
            epi_returns[train_id] = epi_return

        # Prepare stats dict for submission to wand
        stats = {
            "train_loss_mean": losses.mean(),
            "train_loss_sd": losses.std(),
            "episode_return_mean": epi_returns.mean(),
            "episode_return_sd": epi_returns.std(),
        }
        for train_id in range(config.n_train_evals):
            stats[f"train_loss_{train_id}"] = losses[train_id]
            stats[f"episode_return_{train_id}"] = epi_returns[train_id]
        wandb.log(stats)


if __name__ == "__main__":
    train()
