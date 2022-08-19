import random

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from levers import IteratedLeverEnvironment
from levers.partners import FixedPatternPartner
from levers.learner import DRQNAgent, DRQNetwork



# Reproducibility
seed = 0
random.seed(seed)
torch.manual_seed(seed)

# Misc
torch.set_printoptions(precision=2)

# Initialize environment
env = IteratedLeverEnvironment(
    payoffs=[1., 1.], 
    n_iterations=6,
    partner=FixedPatternPartner([0, 1, 1, 1]),
    include_payoffs=False,
    include_step=False,
)

# Initialize DRQN agent
learner = DRQNAgent(
    q_net=DRQNetwork(
        rnn = nn.LSTM(
            input_size=len(env.dummy_obs()),
            hidden_size=16,
            batch_first=True,
        ),
        fnn = nn.Sequential(
            nn.Linear(in_features=16, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=env.n_actions())
        )
    ),
    capacity=16,
    batch_size=8,
    lr=0.01,
    gamma=1.0,
    len_update_cycle=4,
    tau=0.025,
)

stats = {
    'episode': [],
    'return': [],
    'loss': [],
}
n_episodes = 2000
for episode in range(n_episodes):
    # Reset environment, learner's hidden state, and trajectory buffy
    obs = env.reset()
    learner.reset_new_episode(init_obs=obs)
    episode_return = 0

    # Step through environment
    done = False
    while not done:
        # Obtain action from learner and advance internal hidden state
        eps=0.3*max(0, 1 - 2*(episode + 1) / n_episodes)
        action = learner.act(obs, epsilon=eps)
        # Take step in environment
        next_obs, reward, done = env.step(action)
        # Add experience to learners trajectory buffer
        learner.update_trajectory_buffer(action, reward, next_obs, done)
        episode_return += reward
        # Update next observation -> observation
        obs = next_obs

    # Flush experience to replay memory and train
    learner.flush_trajectory_buffer()
    loss = learner.train()

    # Log episode's stats
    stats['episode'].append(episode)
    stats['return'].append(episode_return)
    stats['loss'].append(loss)

    if (episode+1) % 250 == 0:
        print("Episode: {epi:4d} | Epsilon: {eps:2.2f} | Loss: {loss:6.4f} | Return: {ret:2.2f} {is_optimal}".format(
            epi=episode+1,
            eps=eps,
            loss=loss if loss else -1,
            ret=episode_return,
            is_optimal='*' if episode_return == env.episode_length else '',
        ))

        # Rollout environment to log current q-value estimates
        print('-' * 75)
        with torch.no_grad():
            obs = env.reset()
            learner.reset_new_episode(init_obs=obs)
            done = False
            while not done:
                print(learner.q_net(obs.unsqueeze(0), learner.hidden)[0])
                action = learner.act(obs)
                next_obs, reward, done = env.step(action)
                obs = next_obs
        print('-' * 75)

        # Save loss curve
        plt.plot(stats['episode'], stats['loss'])
        plt.semilogy()
        plt.xlabel('Epoch')
        plt.ylabel('DRQN training loss')
        plt.savefig('test.png')
        plt.close()
