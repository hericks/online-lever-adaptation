from ..environment import IteratedLeverEnvironment
from ..learner import DRQNAgent

from typing import Union, List

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
    `n_episodes` episodes with exploration probability `epsilon`. Training 
    bootstraps into the last step to simulate 
    """
    for episode in range(n_episodes):
        # Sample and reset environment from training environments
        env = random.sample(envs, 1)[0]
        obs = env.reset()
        agent.reset_trajectory_buffer(init_obs=obs)

        # Step through environment
        if bootstrap_last_step:
            for step in range(env.episode_length-1):
                action = agent.act(obs, epsilon)
                next_obs, reward, done = env.step(action)
                agent.update_trajectory_buffer(action, reward, next_obs, done)
                obs = next_obs 

                # This should not happen with the iterated lever environment
                if done:
                    raise NameError('Episode done before reaching bootstrap step')
        else:
            done = False
            while not done:
                action = agent.act(obs, epsilon)
                next_obs, reward, done = env.step(action)
                agent.update_trajectory_buffer(action, reward, next_obs, done)
                obs = next_obs 

        # Flush experience to replay memory and train learner
        agent.flush_trajectory_buffer()
        agent.train()

def eval_drqn_agent(
    agent: DRQNAgent,
    envs: List[IteratedLeverEnvironment],
    bootstrap_last_step=False,
):
    """
    Note: All the environments in `env` should have the same episode length.
    """
    eval_stats = {
        'actions': torch.zeros((len(envs), envs[0].episode_length-1)),
        'rewards': torch.zeros((len(envs), envs[0].episode_length-1))
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