import torch
import torch.nn as nn
import torch.nn.functional as F

from levers import IteratedLeverEnvironment
from levers.partners import FixedPatternPartner
from levers.learner import DQNAgent, Transition


class QNetwork(nn.Module):
    """Simple single hidden layer MLP with 4 hidden units. """

    def __init__(self, input_dim, n_actions):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 4)
        self.fc2 = nn.Linear(4, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# Initialize the environment
env = IteratedLeverEnvironment(
    payoffs=[1., 1.], 
    n_iterations=2, 
    partner=FixedPatternPartner([0])
)

# Initialize DQN agent
learner = DQNAgent(
    q_net=QNetwork(input_dim=len(env.dummy_obs()), n_actions=env.n_actions()),
    capacity=8,
    batch_size=4,
    lr=0.005
)

for epoch in range(200):
    # Reset environment
    obs = env.reset()
    done = False

    # Step through environment
    while not done:
        # Obtain action from learner
        action = learner.act(obs, epsilon=0.3)
        # Take stap in environment
        next_obs, reward, done = env.step(action)
        # Give experience to learner and train
        learner.update_memory(Transition(obs, action, next_obs, reward, done))
        learner.train(done)
        # Update next observation -> observation
        obs = next_obs

    # Evaluate learner every 50 episodes
    if (epoch+1) % 50 == 0:
        print(f'--- AFTER EPOCH {epoch+1}')
        observations = [
            torch.tensor([0., 1., 1., 0., 0.]),
            torch.tensor([1., 1., 1., 1., 0.]) 
        ]
        for obs in observations:
            print(f'obs: {obs}, q-values: {learner.q_net(obs)}')