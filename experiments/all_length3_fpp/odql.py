import os
import random
from itertools import combinations
from datetime import datetime
from typing import List

import configargparse
import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector

from evotorch import Problem
from evotorch.logging import PandasLogger, StdOutLogger

from levers import IteratedLeverEnvironment
from levers.partners import FixedPatternPartner
from levers.learner import DQNAgent, OpenES
from levers.helpers import (
    generate_binary_patterns,
    n_total_parameters,
    eval_DQNAgent,
)


def get_parser(default_config_files: List[str]):
    p = configargparse.ArgParser(default_config_files=default_config_files)

    # Reproducibility
    p.add_argument("--seed", type=int)

    # Environment
    p.add_argument("--payoffs", action="append", type=float)
    p.add_argument("--n_iterations", type=int)

    # Training
    p.add_argument("--n_train_evals", type=int)
    p.add_argument("--train_id_start", type=int)
    p.add_argument("--epsilon", type=float)

    # History representation network
    p.add_argument("--hist_rep_dim", type=int)

    # Learner
    p.add_argument("--dqn_hidden_dim", type=int)
    p.add_argument("--capacity", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--dqn_lr", type=float)
    p.add_argument("--tau", type=float)
    p.add_argument("--gamma", type=float)
    p.add_argument("--len_update_cycle", type=int)

    # Evolution strategies
    p.add_argument("--n_epochs", type=int)
    p.add_argument("--popsize", type=int)
    p.add_argument("--es_lr", type=float)
    p.add_argument("--stdev_init", type=float)
    p.add_argument("--stdev_decay", type=float)
    p.add_argument("--stdev_min", type=float)

    # Saving
    p.add_argument("--save", action="store_true")
    p.add_argument("--log_interval", type=int)

    return p


def run_experiment(opt):
    # For reproducibility
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    patterns = generate_binary_patterns(3)
    for partner_patterns in combinations(patterns, 4):
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
            print(f"{datetime.now()} {partner_patterns} {train_id:02d}")

            hist_rep = nn.LSTM(
                input_size=len(train_envs[0].dummy_obs()),
                hidden_size=opt.hist_rep_dim,
            )

            learner = DQNAgent(
                q_net=nn.Sequential(
                    nn.Linear(opt.hist_rep_dim, opt.dqn_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(opt.dqn_hidden_dim, train_envs[0].n_actions()),
                ),
                capacity=opt.capacity,
                batch_size=opt.batch_size,
                lr=opt.dqn_lr,
                tau=opt.tau,
                len_update_cycle=opt.len_update_cycle,
            )

            # Initialize ES problem and algorithm
            n_learner_params = n_total_parameters(learner.q_net)
            n_hist_rep_params = n_total_parameters(hist_rep)

            problem = Problem(
                "max",
                lambda param_vec: eval_DQNAgent(
                    learner=learner,
                    hist_rep=hist_rep,
                    envs=train_envs,
                    bootstrap_last_step=True,
                    train=True,
                    epsilon=opt.epsilon,
                    param_vec=param_vec,
                ),
                solution_length=n_learner_params + n_hist_rep_params,
                initial_bounds=(-1, 1),
                num_actors="max",
            )

            searcher = OpenES(
                problem,
                popsize=opt.popsize,
                learning_rate=opt.es_lr,
                stdev_init=opt.stdev_init,
                stdev_decay=opt.stdev_decay,
                stdev_min=opt.stdev_min,
                mean_init=torch.concat(
                    (
                        parameters_to_vector(learner.q_net.parameters()),
                        parameters_to_vector(hist_rep.parameters()),
                    )
                ),
            )

            # Attach loggers
            if opt.save:
                pandas_logger = PandasLogger(searcher)
            if opt.log_interval != 0:
                stdout_logger = StdOutLogger(
                    searcher, interval=opt.log_interval
                )

            # Run ES algorithm
            searcher.run(opt.n_epochs)

            # Save models and train stats
            if opt.save:
                train_stats = pandas_logger.to_dataframe()
                torch.save(
                    learner.q_net.state_dict(),
                    f"models/odql/ODQL-{partner_patterns}-{train_id}.pt",
                )
                torch.save(
                    hist_rep.state_dict(),
                    f"models/odql/HIST-{partner_patterns}-{train_id}.pt",
                )
                torch.save(
                    train_stats,
                    f"train_stats/odql/ODQL-{partner_patterns}-{train_id}.pickle",
                )


if __name__ == "__main__":
    # Load configuration
    default_config_files = [
        os.path.join(
            "/data/engs-oxfair3/orie4536/online-lever-adaptation/",
            "experiments",
            "all_length3_fpp",
            "defaults.conf",
        ),
        os.path.join(
            "/data/engs-oxfair3/orie4536/online-lever-adaptation/",
            "experiments",
            "all_length3_fpp",
            "odql.conf",
        ),
    ]
    p = get_parser(default_config_files)
    opt = p.parse_args()

    print("ODQL Experiment parameters.")
    print(opt, end="\n\n")

    print("Parameter sources.")
    print(p.format_values())

    run_experiment(opt)
