from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from levers import IteratedLeverEnvironment
from levers.learner.history_shaper import HistoryShaper
from levers.partners import FixedPatternPartner
from levers.learner import OpenES, DQNAgent, Transition


def eval_learner(
    learner: DQNAgent,
    env: IteratedLeverEnvironment,
    hist_shaper: HistoryShaper,
    n_episodes: int = 1
):
    """
    Evaluates the q-learning DQN-Agent `learner` in the environment `env`
    by rolling out `n_episodes` episodes. Cumulative reward serves as measure
    of fitness.
    """
    # Evaluate learners fitness
    cumulative_reward = 0
    for episode in range(n_episodes):
        # Reset environment
        obs = env.reset()
        obs_rep, hidden = hist_shaper.net(obs.unsqueeze(0))
        done = False

        # Step through environment
        while not done:
            # Obtain action from learner
            action, _ = learner.act(obs_rep.squeeze(0), epsilon=0.3)
            # Take step in environment
            next_obs, reward, done = env.step(action)
            cumulative_reward += reward
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
            obs_rep = next_obs_rep
            hidden = next_hidden

    return cumulative_reward


def eval_population(
    population: List[DQNAgent],
    env: IteratedLeverEnvironment,
    learner: DQNAgent,
    hist_shaper: HistoryShaper,
    n_episodes: int = 1
):
    """
    Evaluates the list of q-learning DQN-Agents `population` in the environment
    `env` by rolling out `n_episodes` episodes. Cumulative reward serves as
    measure of fitness.
    """
    population_fitness = []
    for member in population:
        # Populate learner and history shaper with proposal params
        learner.reset(member['q_net'])
        hist_shaper.reset(member['hs_net'])
        # Evaluate learner and save fitness
        fitness = eval_learner(learner, env, hist_shaper, n_episodes)
        population_fitness.append(fitness)

    return population_fitness


# Initialize environment
env = IteratedLeverEnvironment(
    payoffs=[1., 1., 1.], 
    n_iterations=6, 
    partner=FixedPatternPartner([0, 1, 2]),
    include_step=False,
    include_payoffs=False,
)

# Initialize history shaper with LSTM net
hs_output_size=4
hist_shaper = HistoryShaper(
    hs_net=nn.LSTM(input_size=len(env.dummy_obs()), hidden_size=hs_output_size)
)

# Initialize DQN agent
learner = DQNAgent(
    q_net=nn.Sequential(
        nn.Linear(hs_output_size, 4),
        nn.ReLU(),
        nn.Linear(4, env.n_actions())
    ),
    capacity=16,
    batch_size=8,
    lr=0.01
)

# Initialize evolution strategy
es_strategy = OpenES(
    pop_size=50, 
    sigma_init=0.1, sigma_decay=0.999, sigma_limit=0.01, 
    optim_lr=0.01,
    optim_maximize=True,
)

# Further settings
n_es_epochs = 150
n_q_learning_episodes = 20
print_every_k = 1

# Reset strategy and perform evolve using Ask-Eval-Tell loop
es_params = {
    'q_net': learner.q_net.parameters(),
    'hs_net': hist_shaper.net.parameters(),
}
es_strategy.reset(es_params)
for es_epoch in range(n_es_epochs):
    # Ask for proposal population
    population = es_strategy.ask()
    # Evaluate population
    population_fitness = eval_population(
        population, env, learner, hist_shaper, n_q_learning_episodes
    )
    # Tell (update mean parameters)
    mean = es_strategy.tell(population_fitness)
    # Log epoch stats
    if (es_epoch + 1) % print_every_k == 0:
        print('ES-EPOCH: {epoch:2d} (sigma={sigma:2.2f}) | REWARD (MIN/MEAN/MAX): {min:2.2f}, {mean:2.2f}, {max:2.2f}'.format(
            epoch=es_epoch+1, 
            sigma=es_strategy.sigma,
            min=min(population_fitness),
            mean=sum(population_fitness) / es_strategy.pop_size,
            max=max(population_fitness),
        ))