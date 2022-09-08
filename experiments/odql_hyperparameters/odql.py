from copy import deepcopy
from statistics import stdev
from time import gmtime, strftime
from itertools import combinations
from typing import Dict, List

from evotorch import Problem
from evotorch.logging import StdOutLogger

from levers import IteratedLeverEnvironment
from levers.helpers.helpers import n_total_parameters
from levers.learner.dqn_agent import DQNAgent
from levers.partners import FixedPatternPartner
from levers.learner import OpenES
from levers.evaluators import eval_DQNAgent, distribute_param_vec_with_hist_rep

import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

import wandb


def get_train_test_splits(patterns, train_size):
    train_test_splits = []
    for train_patterns in combinations(patterns, train_size):
        split = {
            "train": train_patterns,
            "test": [p for p in patterns if p not in train_patterns][0],
        }
        train_test_splits.append(split)
    return train_test_splits


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
        capacity=config.dqn_capacity,
        batch_size=min(config.dqn_batch_size, config.dqn_capacity),
        lr=config.dqn_lr,
        tau=config.dqn_tau,
        len_update_cycle=config.dqn_len_update_cycle,
        gamma=config.dqn_gamma,
        use_running_memory=config.dqn_use_running_memory,
    )


def run_split(
    train_test_splits: List,
    tid: int,
    sid: int,
    results: Dict[str, torch.Tensor],
    config,
):
    # Setup environment, history representation, and learner

    # Training environments
    train_envs = [
        IteratedLeverEnvironment(
            payoffs=[1.0, 1.0],
            n_iterations=100 + 1,
            partner=FixedPatternPartner(pattern),
            include_step=False,
            include_payoffs=False,
        )
        for pattern in train_test_splits[sid]["train"]
    ]

    # Testing environment
    test_env = deepcopy(train_envs[0])
    test_env.partner = FixedPatternPartner(train_test_splits[sid]["test"])

    hist_rep = get_hist_rep(test_env, config)
    learner = get_dqn_learner(test_env, config)

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

    mean_evals = results["mean_evals"]
    center_train_evals = results["center_train_evals"]
    center_test_evals = results["center_test_evals"]
    for epoch in range(config.es_n_epochs):
        if (epoch + 1) % 50 == 0:
            print(f"EPOCH: {epoch+1}", flush=True)

        # Perform ES step
        searcher.run(1)

        # Log ES scores, evaluate center and log scores
        mean_evals[tid, sid, epoch] = searcher.status["mean_eval"]

        for train in [True, False]:
            center_params = searcher.status["center"]
            envs = train_envs if train else [test_env]
            score = eval_DQNAgent(
                learner=learner,
                hist_rep=hist_rep,
                envs=envs,
                bootstrap_last_step=True,
                train=True,
                epsilon=epsilon,
                param_vec=center_params,
                distribute_param_vec=distribute_param_vec_with_hist_rep,
            )["return"]
            if train:
                center_train_evals[tid, sid, epoch] = score
            else:
                center_test_evals[tid, sid, epoch] = score

    # Kill ray workers
    problem.kill_actors()

    # Store results in dic
    results["mean_evals"] = mean_evals
    results["center_train_evals"] = center_train_evals
    results["center_test_evals"] = center_test_evals


def run():
    wandb.init()
    config = wandb.config
    if config.dqn_use_running_memory:
        wandb.run.name = f"running-memory-lr={config.dqn_lr:5.4f}-dqn-eps={config.dqn_eps:2.2f}"
    else:
        wandb.run.name = f"capacity={config.dqn_capacity}-batch-size={config.dqn_batch_size}-lr={config.dqn_lr:5.4f}-dqn-eps={config.dqn_eps:2.2f}"

    # Obtain train-test splits
    patterns = [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
    train_test_splits = get_train_test_splits(patterns, train_size=4)
    n_splits = len(train_test_splits)

    # Prepare tensors of results
    experiment_results_shape = (
        config.n_train_evals,
        n_splits,
        config.es_n_epochs,
    )
    results = {
        "mean_evals": torch.zeros(experiment_results_shape),
        "center_train_evals": torch.zeros(experiment_results_shape),
        "center_test_evals": torch.zeros(experiment_results_shape),
    }
    for tid in range(config.n_train_evals):
        for sid in range(n_splits):
            print(
                f'{strftime("%Y-%m-%d %H:%M:%S", gmtime())}: {tid+1}/{config.n_train_evals} - {sid+1}/{n_splits}',
                flush=True,
            )

            # The results dict is modified in place
            run_split(train_test_splits, tid, sid, results, config)

    # Prepare logging
    mean_evals = results["mean_evals"]
    center_train_evals = results["center_train_evals"]
    center_test_evals = results["center_test_evals"]
    for epoch in range(config.es_n_epochs):
        # Log means
        stats_dict = {
            "mean_eval": mean_evals[:, :, epoch].mean(),
            "center_train_eval": center_train_evals[:, :, epoch].mean(),
            "center_test_eval": center_test_evals[:, :, epoch].mean(),
        }
        # Log splits
        for sid in range(n_splits):
            sid_key = f"mean_eval_split_{sid}"
            stats_dict[sid_key] = mean_evals[:, sid, epoch].mean()

            sid_key = f"center_train_eval_{sid}"
            stats_dict[sid_key] = center_train_evals[:, sid, epoch].mean()

            sid_key = f"center_test_eval_{sid}"
            stats_dict[sid_key] = center_test_evals[:, sid, epoch].mean()
        wandb.log(stats_dict)


if __name__ == "__main__":
    print("Starting run...")
    run()
