from itertools import combinations
from datetime import datetime
from typing import List
from os import path

import configargparse
import numpy as np
import random

import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector

from levers import IteratedLeverEnvironment
from levers.partners import FixedPatternPartner
from levers.learner import DRQNAgent, DRQNetwork
from levers.helpers import generate_binary_patterns, train_drqn_agent


def get_parser(default_config_files: List[str]):
    p = configargparse.ArgParser(default_config_files=default_config_files)

    # Reproducibility
    p.add_argument("--seed", type=int)

    # Environment
    p.add_argument("--payoffs", action="append", type=float)
    p.add_argument("--n_iterations", type=int)

    # Experiment
    p.add_argument("--n_train_evals", type=int)
    p.add_argument("--train_id_start", type=int)
    p.add_argument("--n_episodes", type=int)
    p.add_argument("--epsilon", type=float)

    # DRQN Training
    p.add_argument("--rnn_hidden_dim", type=int)
    p.add_argument("--fnn_hidden_dim", type=int)
    p.add_argument("--capacity", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--tau", type=float)
    p.add_argument("--gamma", type=float)
    p.add_argument("--len_update_cycle", type=int)

    # Saving
    p.add_argument("--save", action="store_true")

    return p


def run_experiment(opt):
    # Ensure reproducibility
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    patterns = generate_binary_patterns(3)
    for ppid, partner_patterns in enumerate(combinations(patterns, 4)):
        train_envs = [
            IteratedLeverEnvironment(
                payoffs=opt.payoffs,
                n_iterations=opt.n_iterations + 1,
                partner=FixedPatternPartner(pattern),
                include_step=False,
                include_payoffs=False,
            )
            for pattern in partner_patterns
        ]
        train_id_end = opt.train_id_start + opt.n_train_evals
        for train_id in range(opt.train_id_start, train_id_end):
            print(
                f"{datetime.now()} {partner_patterns} {ppid} {train_id:02d}",
                end="",
            )

            # When running on cluster, check if output files already exist.
            # If this is the case, skip this iteration.
            if opt.save:
                q_net_path = (
                    f"models/drqn/DRQN-{partner_patterns}-{train_id}.pt"
                )
                train_stats_path = f"train_stats/drqn/DRQN-{partner_patterns}-{train_id}.pickle"
                if path.isfile(q_net_path) and path.isfile(train_stats_path):
                    print(" already exists.")
                    continue
                else:
                    print("")
            else:
                print("")

            # Setup DRQN agent
            learner = DRQNAgent(
                q_net=DRQNetwork(
                    rnn=nn.LSTM(
                        input_size=len(train_envs[0].dummy_obs()),
                        hidden_size=opt.rnn_hidden_dim,
                        batch_first=True,
                    ),
                    fnn=nn.Sequential(
                        nn.Linear(
                            in_features=opt.rnn_hidden_dim,
                            out_features=opt.fnn_hidden_dim,
                        ),
                        nn.ReLU(),
                        nn.Linear(
                            in_features=opt.fnn_hidden_dim,
                            out_features=train_envs[0].n_actions(),
                        ),
                    ),
                ),
                capacity=opt.capacity,
                batch_size=opt.batch_size,
                lr=opt.lr,
                tau=opt.tau,
                gamma=opt.gamma,
                len_update_cycle=opt.len_update_cycle,
            )

            # Train DRQN agent
            train_stats = train_drqn_agent(
                agent=learner,
                envs=train_envs,
                n_episodes=opt.n_episodes,
                epsilon=opt.epsilon,
                bootstrap_last_step=True,
            )

            # Save model
            if opt.save:
                torch.save(learner.q_net.state_dict(), q_net_path)
                torch.save(
                    train_stats,
                    train_stats_path,
                )


if __name__ == "__main__":
    # Load configuration
    default_config_files = [
        path.join(
            "/data/engs-oxfair3/orie4536/online-lever-adaptation/",
            "experiments",
            "all_length3_fpp",
            "defaults.conf",
        ),
        path.join(
            "/data/engs-oxfair3/orie4536/online-lever-adaptation/",
            "experiments",
            "all_length3_fpp",
            "drqn.conf",
        ),
    ]
    p = get_parser(default_config_files)
    opt = p.parse_args()

    print("DRQN Experiment Parameters.")
    print(opt, end="\n\n")

    print("Parameter sources.")
    print(p.format_values())

    run_experiment(opt)
