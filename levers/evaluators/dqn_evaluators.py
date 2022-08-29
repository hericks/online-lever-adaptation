from typing import Callable, Optional, List
from copy import deepcopy

import numpy as np

import torch
import torch.nn as nn

from torch.nn.utils import vector_to_parameters

from ..helpers import n_total_parameters
from ..learner import DQNAgent, Transition
from .. import IteratedLeverEnvironment


def distribute_param_vec_with_hist_rep(
    learner: DQNAgent, hist_rep: nn.RNNBase, param_vec: torch.Tensor
):
    n_learner = n_total_parameters(learner.q_net)
    vector_to_parameters(param_vec[:n_learner], learner.q_net.parameters())
    vector_to_parameters(param_vec[n_learner:], hist_rep.parameters())


def distribute_param_vec_no_hist_rep(
    learner: DQNAgent, hist_rep: nn.RNNBase, param_vec: torch.Tensor
):
    vector_to_parameters(param_vec, learner.q_net.parameters())


def eval_DQNAgent(
    learner: DQNAgent,
    hist_rep: Optional[nn.RNNBase],
    envs: List[IteratedLeverEnvironment],
    bootstrap_last_step: bool,
    train: bool,
    epsilon: Optional[float],
    param_vec: Optional[torch.Tensor] = None,
    distribute_param_vec: Optional[
        Callable[[DQNAgent, Optional[nn.RNNBase], torch.Tensor], None]
    ] = None,
):
    """
    Evaluates the q-learning DQN-Agent `learner` in the environment `env`
    by rolling out `n_episodes` episodes. Cumulative reward serves as measure
    of fitness.
    """
    # Distribute parameter vector if available
    if param_vec is not None:
        if distribute_param_vec is None:
            raise ValueError(
                "Got parameter vector `param_vec` but no function (`distribute_param_vec`) to distribute it."
            )

        distribute_param_vec(learner, hist_rep, param_vec)

    # Save to reload after each environment evaluation
    if train:
        backup_state_dict = deepcopy(learner.q_net.state_dict())
    else:
        backup_state_dict = None

    def _eval_hist_rep(learner, hist_rep, env, epsilon, train):
        obs = env.reset()
        obs_rep, hidden = hist_rep(obs.unsqueeze(0))
        env_return, env_greedy_return, env_n_greedy_steps = 0, 0, 0
        for step in range(env.episode_length - int(bootstrap_last_step)):
            action, greedy = learner.act(obs_rep.squeeze(0), epsilon=epsilon)
            next_obs, reward, done = env.step(action)
            next_obs_rep, next_hidden = hist_rep(next_obs.unsqueeze(0), hidden)
            if train:
                learner.update_memory(
                    Transition(
                        obs_rep.squeeze(0).detach(),
                        action,
                        next_obs_rep.squeeze(0).detach(),
                        reward,
                        done,
                    )
                )
                learner.train()
            env_return += reward
            env_greedy_return += reward if greedy else 0
            env_n_greedy_steps += 1 if greedy else 0
            obs_rep = next_obs_rep
            hidden = next_hidden
        return env_return, env_greedy_return, env_n_greedy_steps

    def _eval_no_hist_rep(learner, env, epsilon, train):
        obs = env.reset()
        env_return, env_greedy_return, env_n_greedy_steps = 0, 0, 0
        for _ in range(env.episode_length - int(bootstrap_last_step)):
            action, greedy = learner.act(obs, epsilon=epsilon)
            next_obs, reward, done = env.step(action)
            if train:
                learner.update_memory(
                    Transition(
                        obs.detach(),
                        action,
                        next_obs.detach(),
                        reward,
                        done,
                    )
                )
                learner.train()
            env_return += reward
            env_greedy_return += reward if greedy else 0
            env_n_greedy_steps += 1 if greedy else 0
            obs = next_obs
        return env_return, env_greedy_return, env_n_greedy_steps

    eval_stats = {
        "return": 0,
        "greedy_return": 0,
        "n_greedy_steps": 0,
    }
    for env in envs:
        learner.reset(backup_state_dict)
        if hist_rep is not None:
            env_ret, env_greedy_ret, env_n_greedy_steps = _eval_hist_rep(
                learner, hist_rep, env, epsilon, train
            )
        else:
            env_ret, env_greedy_ret, env_n_greedy_steps = _eval_no_hist_rep(
                learner, env, epsilon, train
            )
        eval_stats["return"] += env_ret
        eval_stats["greedy_return"] += env_greedy_ret
        eval_stats["n_greedy_steps"] += env_n_greedy_steps

    return eval_stats


def eval_DQNAgent_xplay(
    env: IteratedLeverEnvironment,
    agent1: DQNAgent,
    agent2: DQNAgent,
    hist_rep1: Optional[nn.RNNBase],
    hist_rep2: Optional[nn.RNNBase],
    bootstrap_last_step: bool,
    train1: bool,
    train2: bool,
    epsilon_schedule1: Optional[Callable],
    epsilon_schedule2: Optional[Callable],
):
    def apply_optional_hist_rep(
        obs: torch.Tensor,
        hidden: Optional[torch.Tensor],
        hist_rep: Optional[nn.RNNBase],
    ):
        if hist_rep is None:
            return obs.unsqueeze(0), None
        else:
            return hist_rep(obs.unsqueeze(0), hidden)

    def apply_optional_epsilon_schedule(
        schedule: Optional[Callable], step: int, episode_length: int
    ):
        if schedule is None:
            return 0
        else:
            return schedule(step, episode_length)

    joint_obs = env.reset()
    agent1.reset()
    agent2.reset()

    obs_rep1, hidden1 = apply_optional_hist_rep(joint_obs[0], None, hist_rep1)
    obs_rep2, hidden2 = apply_optional_hist_rep(joint_obs[1], None, hist_rep2)

    n_steps = env.episode_length - int(bootstrap_last_step)
    eval_stats = {
        "rewards": np.zeros(n_steps),
        "action1": np.zeros(n_steps),
        "action2": np.zeros(n_steps),
        "greedy1": np.full(n_steps, False),
        "greedy2": np.full(n_steps, False),
    }
    for step in range(n_steps):
        epsilon1 = apply_optional_epsilon_schedule(
            epsilon_schedule1,
            step,
            env.episode_length - int(bootstrap_last_step),
        )
        action1, greedy1 = agent1.act(obs_rep1.squeeze(0), epsilon=epsilon1)

        epsilon2 = apply_optional_epsilon_schedule(
            epsilon_schedule2,
            step,
            env.episode_length - int(bootstrap_last_step),
        )
        action2, greedy2 = agent2.act(obs_rep2.squeeze(0), epsilon=epsilon2)
        next_joint_obs, reward, done = env.step([action1, action2])

        next_obs_rep1, next_hidden1 = apply_optional_hist_rep(
            next_joint_obs[0], hidden1, hist_rep1
        )
        next_obs_rep2, next_hidden2 = apply_optional_hist_rep(
            next_joint_obs[1], hidden2, hist_rep2
        )

        # Optionally train agents
        if train1:
            agent1.update_memory(
                Transition(
                    obs_rep1.squeeze(0).detach(),
                    action1,
                    next_obs_rep1.squeeze(0).detach(),
                    reward,
                    done,
                )
            )
            agent1.train()
        if train2:
            agent2.update_memory(
                Transition(
                    obs_rep2.squeeze(0).detach(),
                    action1,
                    next_obs_rep2.squeeze(0).detach(),
                    reward,
                    done,
                )
            )
            agent2.train()

        # Update stats
        eval_stats["rewards"][step] = reward
        eval_stats["action1"][step] = action1
        eval_stats["action2"][step] = action2
        eval_stats["greedy1"][step] = greedy1
        eval_stats["greedy2"][step] = greedy2

        # Prepare next step
        obs_rep1 = next_obs_rep1
        obs_rep2 = next_obs_rep2
        hidden1 = next_hidden1
        hidden2 = next_hidden2

    return eval_stats
