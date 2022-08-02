# Relative imports outside of package
import sys
from os import path
from numpy import squeeze
sys.path.insert(1, path.join(sys.path[0], '../..'))

import torch
import torch.nn as nn

from levers import IteratedLeverEnvironment
from levers.partners import FixedPatternPartner
from levers.learner import HistoryShaper, DQNAgent, OpenES, Transition


# Environment settings
payoffs = [1., 1.]
truncated_length = 150
include_step=False
include_payoffs=False

# History shaper settings
hs_hidden_size = 4

# Learner settings
learner_hidden_size = 4
capacity = 16
batch_size = 8
lr = 0.01
gamma = 0.99
len_update_cycle = 10
# epsilon = 0.3

len3_patterns = [
    [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
    [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],]
test_patterns = [
    [0, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1],]
train_patterns = [
    pattern for pattern in len3_patterns if pattern not in test_patterns]

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

# Initialize history shaper
hist_shaper = HistoryShaper(
    hs_net=nn.LSTM(
        input_size=len(train_envs[0].dummy_obs()),
        hidden_size=hs_hidden_size))

# Initialize DQN agent
learner = DQNAgent(
    q_net=nn.Sequential(
        nn.Linear(hs_hidden_size, learner_hidden_size),
        nn.ReLU(),
        nn.Linear(learner_hidden_size, train_envs[0].n_actions())
    ),
    capacity=capacity,
    batch_size=batch_size,
    lr=lr,
    gamma=gamma,
    len_update_cycle=len_update_cycle
)

epoch = 150
hs_net_name = f'./experiments/online_qlearner_length3/data/hs-net-epoch-{epoch:04d}.pt'
q_net_name = f'./experiments/online_qlearner_length3/data/q-net-epoch-{epoch:04d}.pt'


hist_shaper.net.load_state_dict(torch.load(hs_net_name))
learner.q_net.load_state_dict(torch.load(q_net_name))

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

        learner.reset()

        greedy_pattern = ''
        ret = 0
        obs = env.reset()
        obs_rep, hidden = hist_shaper.net(obs.unsqueeze(0))

        for step in range(truncated_length):
            epsilon = 1 * (1 - 4 * step / truncated_length)
            action, _ = learner.act(obs_rep.squeeze(0), epsilon=epsilon)
            next_obs, reward, done = env.step(action)
            ret += reward
            greedy_pattern += str(action)
            # Compute history representation
            next_obs_rep, next_hidden = hist_shaper.net(
                next_obs.unsqueeze(0), hidden)
            # Give experience to learner and train
            learner.update_memory(
                Transition(
                    obs_rep.squeeze(0).detach(),
                    action, 
                    next_obs_rep.squeeze(0).detach(), 
                    reward, done
                )
            )
            learner.train(done)
            # Update next observation -> observation
            obs = next_obs
            obs_rep = next_obs_rep
            hidden = next_hidden

        print('Greedy :', greedy_pattern)
        print(f'Reward: {ret:3.0f} ({ret / truncated_length * 100:6.2f}%)\n')