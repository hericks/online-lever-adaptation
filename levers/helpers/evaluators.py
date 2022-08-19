from ..environment import IteratedLeverEnvironment
from ..learner import DRQNAgent

from typing import List

import random
import torch


def train_drqn_agent(
    agent: DRQNAgent, 
    envs: List[IteratedLeverEnvironment],
    n_episodes: int, 
    epsilon: float,
    bootstrap_last_step = False,
):
    """
    Trains a DRQN agent in environments uniformly sampled from `envs` for
    `n_episodes` episodes with exploration probability `epsilon`.
    """
    train_stats = {
        'episode': [],
        'loss': [],
        'return': [],
    }
    for episode in range(n_episodes):
        # Sample and reset environment from training environments
        env = random.sample(envs, 1)[0]
        obs = env.reset()
        agent.reset_new_episode(init_obs=obs)

        # Setup stats
        episode_return = 0

        # Step through environment
        for step in range(env.episode_length - int(bootstrap_last_step)):
            action = agent.act(obs, epsilon)
            next_obs, reward, done = env.step(action)
            agent.update_trajectory_buffer(action, reward, next_obs, done)
            obs = next_obs 

            # Update episode stats
            episode_return += reward 

        # Flush experience to replay memory and train learner
        agent.flush_trajectory_buffer()
        loss = agent.train()

        # Fill train stats
        train_stats['episode'].append(episode)
        train_stats['loss'].append(loss)
        train_stats['return'].append(episode_return)

    return train_stats

def eval_drqn_agent(
    agent: DRQNAgent,
    envs: List[IteratedLeverEnvironment],
    bootstrap_last_step: bool = False,
):
    """
    Note: All the environments in `env` should have the same episode length.
    """
    eval_stats = {
        'actions': torch.zeros((len(envs), envs[0].episode_length-1)),
        'rewards': torch.zeros((len(envs), envs[0].episode_length-1)),
    }
    for env_id, env in enumerate(envs):
        # Reset environment and DRQN's hidden state
        obs = env.reset()
        agent.hidden = None

        # Step through environment
        if bootstrap_last_step:
            for step in range(env.episode_length-1):
                action = agent.act(obs)
                next_obs, reward, done = env.step(action)
                obs = next_obs 

                # Fill results
                eval_stats['actions'][env_id, step] = action
                eval_stats['rewards'][env_id, step] = reward

                # This should not happen with the iterated lever environment
                if done:
                    raise NameError('Episode done before reaching bootstrap step')
        else:
            raise NotImplementedError()

    return eval_stats

def eval_drqn_agents(
    agents: List[DRQNAgent],
    envs: List[IteratedLeverEnvironment],
    bootstrap_last_step: bool = False
):
    """
    """
    out_shape = (len(agents), len(envs), envs[0].episode_length-1)
    eval_stats = {
        'actions': torch.zeros(out_shape),
        'rewards': torch.zeros(out_shape),
    }

    # Evaluate each agent and save corresponding stats
    for agent_id, agent in enumerate(agents):
        agent_results = eval_drqn_agent(agent, envs, bootstrap_last_step)
        eval_stats['actions'][agent_id,:,:] = agent_results['actions']
        eval_stats['rewards'][agent_id,:,:] = agent_results['rewards']

    return eval_stats