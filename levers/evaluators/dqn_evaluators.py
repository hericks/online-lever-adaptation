from typing import Optional, List
from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn.utils import vector_to_parameters

from ..learner import DQNAgent, Transition
from .. import IteratedLeverEnvironment


def eval_DQNAgent(
    learner: DQNAgent,
    hist_rep: Optional[nn.RNNBase],
    envs: List[IteratedLeverEnvironment],
    bootstrap_last_step: bool,
    train: bool,
    epsilon: Optional[float],
    param_vec: Optional[torch.Tensor] = None,
):
    """
    Evaluates the q-learning DQN-Agent `learner` in the environment `env`
    by rolling out `n_episodes` episodes. Cumulative reward serves as measure
    of fitness.
    """
    if param_vec is not None:
        n_learner_params = sum(p.numel() for p in learner.q_net.parameters())

        # Load learner's state
        vector_to_parameters(
            param_vec[:n_learner_params], learner.q_net.parameters()
        )

        # Load history representation's state
        if hist_rep != None:
            vector_to_parameters(
                param_vec[n_learner_params:], hist_rep.parameters()
            )

    # Save to reload after each environment evaluation
    if train:
        backup_state_dict = deepcopy(learner.q_net.state_dict())
    else:
        backup_state_dict = None

    def _eval_hist_rep(learner, hist_rep, env, epsilon, train):
        obs = env.reset()
        obs_rep, hidden = hist_rep(obs.unsqueeze(0))
        env_return, env_greedy_return, env_n_greedy_steps = 0, 0, 0
        for _ in range(env.episode_length - int(bootstrap_last_step)):
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
