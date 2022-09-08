from itertools import combinations
from time import gmtime, strftime

import wandb
import random
import numpy as np

import torch
import torch.nn as nn

from levers import IteratedLeverEnvironment
from levers.partners import FixedPatternPartner
from levers.learner import DRQNAgent, DRQNetwork
from levers.helpers import generate_binary_patterns


def get_train_patterns(train_patterns_index: int):
    patterns = generate_binary_patterns(3)
    possible_train_patterns = list(combinations(patterns, 4))
    return possible_train_patterns[train_patterns_index]


def run():
    wandb.init()
    config = wandb.config

    # Obtain training patterns
    train_patterns = get_train_patterns(config.exp_train_patterns_index)

    # Set run name
    wandb.run.name = (
        f"train-patterns={train_patterns}-seed={config.exp_train_seed}"
    )

    # Reproducible results
    random.seed(config.exp_train_seed)
    np.random.seed(config.exp_train_seed)
    torch.manual_seed(config.exp_train_seed)

    # Print startup
    print(
        f'{strftime("%Y-%m-%d %H:%M:%S", gmtime())}: {wandb.run.name}',
        flush=True,
    )

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

    # Setup learner
    learner = DRQNAgent(
        q_net=DRQNetwork(
            rnn=nn.LSTM(
                input_size=len(train_envs[0].dummy_obs()),
                hidden_size=config.drqn_rnn_hidden_dim,
                batch_first=True,
            ),
            fnn=nn.Sequential(
                nn.Linear(
                    in_features=config.drqn_rnn_hidden_dim,
                    out_features=config.drqn_fnn_hidden_dim,
                ),
                nn.ReLU(),
                nn.Linear(
                    in_features=config.drqn_fnn_hidden_dim,
                    out_features=train_envs[0].n_actions(),
                ),
            ),
        ),
        capacity=config.drqn_capacity,
        batch_size=config.drqn_batch_size,
        lr=config.drqn_lr,
        tau=config.drqn_tau,
        gamma=config.drqn_gamma,
        len_update_cycle=config.drqn_len_update_cycle,
    )

    # Train DRQN agent
    epsilon = config.drqn_eps_start
    for episode in range(config.drqn_n_episodes):
        if (episode + 1) % 50 == 0:
            print(f"EPISODE: {episode+1}", flush=True)

        # Prepare for logging
        episode_return = 0

        # Train agent on randomly samples training environment
        env = random.sample(train_envs, 1)[0]
        obs = env.reset()
        learner.reset_new_episode(init_obs=obs)
        for _ in range(env.episode_length - 1):
            action = learner.act(obs, epsilon)
            next_obs, reward, done = env.step(action)
            learner.update_trajectory_buffer(action, reward, next_obs, done)
            obs = next_obs
            episode_return += reward

        # Update learner
        learner.flush_trajectory_buffer()
        loss = learner.train()
        epsilon = max(epsilon - config.drqn_eps_diff, config.drqn_eps_min)

        # Log current episode's stats
        stats_dict = {
            "loss": loss if loss is not None else 1,
            "train_return": episode_return,
        }
        wandb.log(stats_dict)

    # Save parameters of final model
    torch.save(
        learner.q_net.state_dict,
        f"models/q-{train_patterns}-{config.exp_train_seed}.pt",
    )
    wandb.save(f"models/q-{train_patterns}-{config.exp_train_seed}.pt")


if __name__ == "__main__":
    print("Starting run...")
    run()
