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

import itertools


def eval_train_patterns(train_patterns, q_net_out_path, hs_net_out_path):
    """
    Performs single evaluation of given train patterns and saves final model
    to output paths provided via `q_net_out_path` and `hs_net_out_path`.
    NOTE. For now, this function relies on global parameters. 
    """
    # Construct list of environments to train on
    train_envs = [
        IteratedLeverEnvironment(
            payoffs, truncated_length+1, FixedPatternPartner(pattern),
            include_step, include_payoffs)
        for pattern in train_patterns
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
            nn.Linear(learner_hidden_size, train_envs[0].n_actions()),
        ),
        capacity=capacity,
        batch_size=batch_size,
        lr=lr,
        gamma=gamma,
        len_update_cycle=len_update_cycle
    )

    # Initialize evolution strategy
    es = OpenES(
        pop_size=pop_size,
        sigma_init=sigma_init,
        sigma_decay=sigma_decay,
        sigma_limit=sigma_limit,
        optim_lr=optim_lr
    )

    # Reset evolution strategy
    es_params = {
        'q_net': learner.q_net.parameters(),
        'hs_net': hist_shaper.net.parameters(),
    }
    es.reset(es_params)

    # ES train loop
    for es_epoch in range(n_es_epochs):
        # Ask for proposal population
        population = es.ask()

        # Evaluate population
        population_fitness = []
        for member in population:
            learner.reset(member['q_net'])
            hist_shaper.reset(member['hs_net'])
            member_fitness = 0
            for env in train_envs:
                obs = env.reset()
                obs_rep, hidden = hist_shaper.net(obs.unsqueeze(0))
                for _ in range(truncated_length):
                    action, _ = learner.act(obs_rep.squeeze(0), epsilon=epsilon)
                    next_obs, reward, done = env.step(action)
                    member_fitness += reward

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
            # Save current member's fitness
            population_fitness.append(member_fitness)

        # Update mean parameters
        mean = es.tell(population_fitness)

        # Log epoch stats
        if (es_epoch + 1) % 10 == 0:
            print('ES-EPOCH: {epoch:2d} (sigma={sigma:2.2f}) | REWARD (MIN/MEAN/MAX): {min:2.2f}, {mean:2.2f}, {max:2.2f}'.format(
                epoch=es_epoch+1, 
                sigma=es.sigma,
                min=min(population_fitness),
                mean=sum(population_fitness) / es.pop_size,
                max=max(population_fitness),
            ))

    # Load elite member and save model
    learner.reset(es.means_dict['q_net'])
    hist_shaper.reset(es.means_dict['hs_net'])
    torch.save(
        learner.q_net.state_dict(), 
        q_net_out_path)
    torch.save(
        hist_shaper.net.state_dict(), 
        hs_net_out_path)


def batch_eval_train_patterns(train_patterns, n_evals, data_folder):
    """
    Evaluate train patterns `n_evals` times and write resulting networks to
    disk. 
    """
    for eval_id in range(n_evals):
        # Define output locations
        q_net_out_path = path.join(data_folder, f'qnet-pattern={train_patterns}-eval_id={eval_id}.pt')
        hs_net_out_path = path.join(data_folder, f'qnet-pattern={train_patterns}-eval_id={eval_id}.pt')

        if path.isfile(q_net_out_path):
            print(f'-> evaluation {eval_id}: ALREADY EXISTS')
            continue

        print(f'-> evaluation {eval_id}: TRAINING')
        eval_train_patterns(
            train_patterns=train_patterns,
            q_net_out_path=q_net_out_path,
            hs_net_out_path=hs_net_out_path)


# Environment settings
payoffs = [1., 1.]
truncated_length = 100
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
epsilon = 0.3

# ES settings
n_es_epochs = 300
pop_size = 50
sigma_init = 0.1
sigma_decay = 0.99
sigma_limit = 0.01
optim_lr = 0.01

# Experiment settings
n_evaluations = 10
data_folder = path.join(path.dirname(__file__), 'data')

patterns = [
    (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
    (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1),
]

for train_patterns in itertools.combinations(patterns, 4):
    print(f'TRAIN-PATTERN: {train_patterns}')
    batch_eval_train_patterns(train_patterns, n_evaluations, data_folder)