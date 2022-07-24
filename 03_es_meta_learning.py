import torch
import torch.nn as nn
import torch.nn.functional as F

from levers import IteratedLeverEnvironment
from levers.partners import FixedPatternPartner
from levers.learner import OpenES, DQNAgent, Transition


def eval_learner(learner, n_episodes: int = 1):
    # Evaluate learners fitness
    cumulative_reward = 0
    for episode in range(n_episodes):
        # Reset environment
        obs = env.reset()
        done = False

        # Step through environment
        while not done:
            # Obtain action from learner
            action = learner.act(obs)
            # Take step in environment
            next_obs, reward, done = env.step(action)
            cumulative_reward += reward
            # Give experience to learner and train
            # learner.update_memory(Transition(obs, action, next_obs, reward, done))
            # learner.train(done)
            # Update next observation -> observation
            obs = next_obs

    return cumulative_reward


def eval_population(population, n_episodes: int = 1):
    population_fitness = []
    for params in population:
        # Populate learner with proposal params
        learner.reset(params)
        # Evaluate learner and save fitness
        fitness = eval_learner(learner, n_episodes)
        population_fitness.append(fitness)

    return population_fitness


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
    partner=FixedPatternPartner([0, 1])
)

# Initialize DQN agent
learner = DQNAgent(
    q_net=QNetwork(input_dim=len(env.dummy_obs()), n_actions=env.n_actions()),
    capacity=16,
    batch_size=8,
    lr=0.01
)

es_strategy = OpenES(pop_size=30)
es_strategy.reset(learner.q_net.parameters())

n_es_epochs = 10
n_q_learning_episodes = 1

# Perform evolution strategies using Ask-Eval-Tell loop
for es_epoch in range(n_es_epochs):
    # Ask for proposal population
    population = es_strategy.ask()
    # Evaluate population
    population_fitness = eval_population(population, n_q_learning_episodes)
    print(population_fitness)
    # Tell (update)
    mean = es_strategy.tell(population_fitness)