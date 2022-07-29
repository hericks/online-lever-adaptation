# Relative imports outside of package
import sys
from os import path
sys.path.insert(1, path.join(sys.path[0], '../..'))

from levers import IteratedLeverEnvironment
from levers.partners import FixedPatternPartner
from levers.learner import DRQNAgent, DRQNetwork

import random


# Environment settings
payoffs = [1., 1.]
truncated_length = 100
include_step=False
include_payoffs=False

len3_patterns = [
    [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
    [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],]
test_patterns = [
    [0, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1],]
train_patterns = [
    pattern for pattern in len3_patterns if pattern not in test_patterns]

# Learner setting
hidden_size = 4
capacity = 8
batch_size = 4
lr = 0.01
gamma = 0.99
len_update_cycle = 4
tau = 5e-4

# Training settings
num_episodes = 1000
epsilon = 0.3

# Construct list of environments to train and test on
train_envs = [
    IteratedLeverEnvironment(
        payoffs, truncated_length+1, FixedPatternPartner(pattern),
        include_step, include_payoffs)
    for pattern in train_patterns
]
test_envs = [
    IteratedLeverEnvironment(
        payoffs, truncated_length+1, FixedPatternPartner(pattern),
        include_step, include_payoffs)
    for pattern in test_patterns
]

# Construct DRQN agent
learner = DRQNAgent(
    DRQNetwork(
        input_size=len(train_envs[0].dummy_obs()),
        hidden_size=4,
        n_actions=train_envs[0].n_actions()),
    capacity, batch_size, lr, gamma, len_update_cycle, tau
)

# Train agent
for episode in range(num_episodes):
    # Sample reset environment from training environments
    env = random.sample(train_envs, 1)[0]
    obs = env.reset()
    learner.reset_trajectory_buffer(init_obs=obs)
    episode_return = 0

    # Step through environment
    for step in range(truncated_length):
        action = learner.act(obs, epsilon)
        next_obs, reward, done = env.step(action)
        episode_return += reward
        learner.update_trajectory_buffer(action, reward, next_obs, done)
        obs = next_obs 

    # Flush experience to replay memory and train learner
    learner.flush_trajectory_buffer()
    loss = learner.train()

    if (episode + 1) % 50 == 0:
        print('Episode: {episode:4d} | Loss: {loss:2.3f} | Return: {ret:3.0f}'.format(
            episode=episode+1,
            loss=loss if loss else -1.,
            ret=episode_return,
        ))

# Evaluate agent
eval_envs = {'TRAINING': train_envs, 'TEST': test_envs}
for name, envs in eval_envs.items():
    print("-" * 100)
    print(name)
    for env in envs:
        print("Environment:", env.partner.pattern)

        optimal_pattern = ''
        for step in range(truncated_length):
            optimal_pattern += str(env.partner.pattern[step % 3])
        print('Optimal:', optimal_pattern)

        greedy_pattern = ''
        ret = 0
        obs = env.reset()
        learner.reset_trajectory_buffer(init_obs=obs)

        for step in range(truncated_length):
            action = learner.act(obs)
            greedy_pattern += str(action)
            next_obs, reward, done = env.step(action)
            ret += reward
            learner.update_trajectory_buffer(action, reward, next_obs, done)
            obs = next_obs 

        print('Greedy :', greedy_pattern)
        print(f'Reward: {ret:3.0f} ({ret / truncated_length * 100:6.2f}%)\n')