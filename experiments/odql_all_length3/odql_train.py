from copy import deepcopy
from statistics import stdev
from time import gmtime, strftime
from itertools import combinations
from typing import Dict, List

import wandb
import random
import numpy as np

from evotorch import Problem
from evotorch.logging import StdOutLogger

import torch
import torch.nn as nn

from torch.nn.utils import parameters_to_vector, vector_to_parameters

from levers import IteratedLeverEnvironment
from levers.helpers import n_total_parameters, generate_binary_patterns
from levers.learner import DQNAgent
from levers.partners import FixedPatternPartner
from levers.learner import OpenES
from levers.evaluators import eval_DQNAgent, distribute_param_vec_with_hist_rep


def get_train_patterns(train_patterns_index: int):
    patterns = generate_binary_patterns(3)
    possible_train_patterns = list(combinations(patterns, 4))
    return possible_train_patterns[train_patterns_index]


def get_hist_rep(env: IteratedLeverEnvironment, config):
    return nn.LSTM(
        input_size=len(env.dummy_obs()),
        hidden_size=config.hist_rep_dim,
    )


def get_dqn_learner(env: IteratedLeverEnvironment, config):
    return DQNAgent(
        q_net=nn.Sequential(
            nn.Linear(config.hist_rep_dim, config.dqn_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.dqn_hidden_dim, env.n_actions()),
        ),
        capacity=0,
        batch_size=0,
        lr=config.dqn_lr,
        tau=config.dqn_tau,
        len_update_cycle=config.dqn_len_update_cycle,
        gamma=config.dqn_gamma,
        use_running_memory=config.dqn_use_running_memory,
    )


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

    # Setup environments, history representation, and learner
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

    hist_rep = get_hist_rep(train_envs[0], config)
    learner = get_dqn_learner(train_envs[0], config)

    # ATTENTION: Never use the wandb config in the evaluation function.
    # It does not work.
    epsilon = config.dqn_eps
    eval = lambda param_vec: eval_DQNAgent(
        learner=learner,
        hist_rep=hist_rep,
        envs=train_envs,
        bootstrap_last_step=True,
        train=True,
        epsilon=epsilon,
        param_vec=param_vec,
        distribute_param_vec=distribute_param_vec_with_hist_rep,
    )["return"]

    # Setup problem, searcher, and logger
    n_learner_params = n_total_parameters(learner.q_net)
    n_hist_rep_params = n_total_parameters(hist_rep)

    problem = Problem(
        "max",
        eval,
        solution_length=n_learner_params + n_hist_rep_params,
        initial_bounds=(-1, 1),
        num_actors=24,
    )

    searcher = OpenES(
        problem,
        popsize=config.es_popsize,
        learning_rate=config.es_lr,
        stdev_init=config.es_sigma_init,
        stdev_decay=config.es_sigma_decay,
        stdev_min=config.es_sigma_min,
        mean_init=torch.concat(
            (
                parameters_to_vector(learner.q_net.parameters()),
                parameters_to_vector(hist_rep.parameters()),
            )
        ),
    )

    # Run ES
    for epoch in range(config.es_n_epochs):
        if (epoch + 1) % 50 == 0:
            print(f"EPOCH: {epoch+1}", flush=True)

        # Perform ES step
        searcher.run(1)

        # Eval center
        center_params = searcher.status["center"]
        center_score = eval_DQNAgent(
            learner=learner,
            hist_rep=hist_rep,
            envs=train_envs,
            bootstrap_last_step=True,
            train=True,
            epsilon=epsilon,
            param_vec=center_params,
            distribute_param_vec=distribute_param_vec_with_hist_rep,
        )["return"]

        # Prepare logging
        stats_dict = {
            "mean_eval": searcher.status["mean_eval"],
            "center_train_eval": center_score,
        }
        wandb.log(stats_dict)

    # Distribute parameter vector to networks and save parameters
    distribute_param_vec_with_hist_rep(learner, hist_rep, center_params)
    torch.save(
        learner.q_net.state_dict(),
        f"models/q-{train_patterns}-{config.exp_train_seed}.pt",
    )
    torch.save(
        hist_rep.state_dict(),
        f"models/hist-{train_patterns}-{config.exp_train_seed}.pt",
    )
    wandb.save(f"models/q-{train_patterns}-{config.exp_train_seed}.pt")
    wandb.save(f"models/hist-{train_patterns}-{config.exp_train_seed}.pt")

    # Kill ray workers
    problem.kill_actors()


if __name__ == "__main__":
    print("Starting run...")
    run()
