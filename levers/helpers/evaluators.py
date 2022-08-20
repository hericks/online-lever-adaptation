from copy import deepcopy
from typing import List, Optional

import random

import torch
import torch.nn as nn

from torch.nn.utils import vector_to_parameters, parameters_to_vector

from ..environment import IteratedLeverEnvironment
from ..learner import DRQNAgent, DQNAgent, HistoryShaper, Transition


def eval_dqn_learner(
    learner: DQNAgent,
    hist_rep: nn.RNNBase,
    envs: List[IteratedLeverEnvironment],
    epsilon: float,
    bootstrap_last_step: bool,
    param_vec: Optional[torch.Tensor] = None,
):
    """
    Evaluates the q-learning DQN-Agent `learner` in the environment `env`
    by rolling out `n_episodes` episodes. Cumulative reward serves as measure
    of fitness.
    """
    if param_vec != None:
        n_learner_params = sum(p.numel() for p in learner.q_net.parameters())

        # Load learner's state
        vector_to_parameters(param_vec[:n_learner_params], learner.q_net.parameters())

        # Load history representation's state
        vector_to_parameters(param_vec[n_learner_params:], hist_rep.parameters())

    # Save to reload after each environment evaluation
    backup_state_dict = deepcopy(learner.q_net.state_dict())

    # Evaluate learners fitness
    cumulative_reward = 0
    for env in envs:
        # Reset environment
        obs = env.reset()
        obs_rep, hidden = hist_rep(obs.unsqueeze(0))

        # Reset learner
        learner.q_net.load_state_dict(backup_state_dict)
        learner.reset()

        for _ in range(env.episode_length - int(bootstrap_last_step)):
            action, _ = learner.act(obs_rep.squeeze(0), epsilon=epsilon)
            next_obs, reward, done = env.step(action)

            # Give experience to learner and train
            next_obs_rep, next_hidden = hist_rep(next_obs.unsqueeze(0), hidden)
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

            # Update train stats
            cumulative_reward += reward

            # Update next observation -> observation
            obs_rep = next_obs_rep
            hidden = next_hidden

    return cumulative_reward


# def eval_drqn_agent(
#     agent: DRQNAgent,
#     envs: List[IteratedLeverEnvironment],
#     bootstrap_last_step: bool = False,
# ):
#     """
#     Note: All the environments in `env` should have the same episode length.
#     """
#     eval_stats = {
#         "actions": torch.zeros((len(envs), envs[0].episode_length - 1)),
#         "rewards": torch.zeros((len(envs), envs[0].episode_length - 1)),
#     }
#     for env_id, env in enumerate(envs):
#         # Reset environment and DRQN's hidden state
#         obs = env.reset()
#         agent.hidden = None

#         # Step through environment
#         if bootstrap_last_step:
#             for step in range(env.episode_length - 1):
#                 action = agent.act(obs)
#                 next_obs, reward, done = env.step(action)
#                 obs = next_obs

#                 # Fill results
#                 eval_stats["actions"][env_id, step] = action
#                 eval_stats["rewards"][env_id, step] = reward

#                 # This should not happen with the iterated lever environment
#                 if done:
#                     raise NameError("Episode done before reaching bootstrap step")
#         else:
#             raise NotImplementedError()

#     return eval_stats


# def eval_drqn_agents(
#     agents: List[DRQNAgent],
#     envs: List[IteratedLeverEnvironment],
#     bootstrap_last_step: bool = False,
# ):
#     """ """
#     out_shape = (len(agents), len(envs), envs[0].episode_length - 1)
#     eval_stats = {
#         "actions": torch.zeros(out_shape),
#         "rewards": torch.zeros(out_shape),
#     }

#     # Evaluate each agent and save corresponding stats
#     for agent_id, agent in enumerate(agents):
#         agent_results = eval_drqn_agent(agent, envs, bootstrap_last_step)
#         eval_stats["actions"][agent_id, :, :] = agent_results["actions"]
#         eval_stats["rewards"][agent_id, :, :] = agent_results["rewards"]

#     return eval_stats
