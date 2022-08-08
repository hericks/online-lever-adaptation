# Relative imports outside of package
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

import torch
import torch.nn as nn
import torch.multiprocessing as mp

from levers import IteratedLeverEnvironment
from levers.partners import FixedPatternPartner
from levers.learner import HistoryShaper, DQNAgent, OpenES, Transition

from numpy import squeeze
from datetime import datetime
import itertools
import argparse
import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Online Q-Learner Experiment flags")

    parser.add_argument(
        "--cfg-path", help="path to train experiment config yaml file",
        type=str, default='config.yml')
    parser.add_argument(
        "--patterns-start-idx",
        help="index of the first train patterns to use for training (0-69)",
        type=int,
        default=0)
    parser.add_argument(
        "--n-patterns", 
        help="The index of the train patterns to use for training (>= 1)",
        type=int,
        default=70)
    parser.add_argument(
        "--evals-start-idx",
        help="start index of training evaluations",
        type=int,
        default=0)
    parser.add_argument(
        "--n-evals",
        help="number of training evaluations of training patterns",
        type=int,
        default=1)

    args = parser.parse_args()

    cfg = {}
    if args.cfg_path is not None:
        with open(args.cfg_path, "r") as stream:
            try:
                cfg = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    def update_dct(d, u, overwrite=False):
        import collections.abc
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = update_dct(d.get(k, {}), v, overwrite=overwrite)
            else:
                if k not in d:
                    d[k] = v
                else:
                    d[k] = v if overwrite else d[k]
        return d

    # Update cfg with any default config that isn't present in the config file
    # cfg_default = {}
    # if cfg_default:
    #     cfg = update_dct(cfg, cfg_default, overwrite=False)

    # Overwrite cfg with any argparse items that aren't present
    final_cfg = update_dct(cfg, {k: v for k, v in args.__dict__.items() if v is not None}, overwrite=True)

    # TODO: Validate the config parameters for consistency

    return final_cfg


def eval_train_patterns(train_patterns, eval_id, cfg):
    print('{time}: Starting eval {eval_id} of train patterns: {train_patterns}...'.format(
        time=datetime.now().strftime("%H:%M:%S"),
        eval_id=eval_id,
        train_patterns=train_patterns,
    ))

    # Load parameters from config ##############################################

    # Environment
    payoffs          = [1., 1.]
    truncated_length = cfg.get('env_truncated_length')
    include_step     = cfg.get('env_include_step')
    include_payoffs  = cfg.get('env_include_payoffs')

    # History shaper
    hs_hidden_size   = cfg.get('hs_hidden_size')

    # Learner
    learner_hidden_size = cfg.get('lrn_hidden_size')
    capacity            = cfg.get('lrn_capacity')
    batch_size          = cfg.get('lrn_batch_size')
    lr                  = cfg.get('lrn_lr')
    gamma               = cfg.get('lrn_gamma')
    len_update_cycle    = cfg.get('lrn_len_update_cycle')
    epsilon             = cfg.get('lrn_epsilon')

    # Evolution strategy
    n_es_epochs = cfg.get('es_n_epochs')
    pop_size    = cfg.get('es_pop_size')
    sigma_init  = cfg.get('es_sigma_init')
    sigma_decay = cfg.get('es_sigma_decay')
    sigma_limit = cfg.get('es_sigma_limit')
    optim_lr    = cfg.get('es_lr')

    ############################################################################

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
                for step in range(truncated_length):
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
        if (es_epoch + 1) % cfg['print_freq'] == 0:
            print('{time}: ES-EPOCH: {epoch:2d} (sigma={sigma:2.2f}) | REWARD (MIN/MEAN/MAX): {min:2.2f}, {mean:2.2f}, {max:2.2f}'.format(
                time=datetime.now().strftime("%H:%M:%S"),
                epoch=es_epoch+1, 
                sigma=es.sigma,
                min=min(population_fitness),
                mean=sum(population_fitness) / es.pop_size,
                max=max(population_fitness),
            ))

    # Load elite member and save model
    data_dir = cfg.get('data_dir')

    # Q network
    q_net_filename = 'q-net-pattern={patterns}-eval_id={id:02d}.pt'.format(
        patterns=train_patterns, id=eval_id)
    q_net_out_path = os.path.join(data_dir, q_net_filename)
    learner.reset(es.means_dict['q_net'])
    torch.save(
        learner.q_net.state_dict(), 
        q_net_out_path)

    # History shaper network
    hs_net_filename = 'hs-net-pattern={patterns}-eval_id={id:02d}.pt'.format(
        patterns=train_patterns, id=eval_id)
    hs_net_out_path = os.path.join(data_dir, hs_net_filename)
    hist_shaper.reset(es.means_dict['hs_net'])
    torch.save(
        hist_shaper.net.state_dict(), 
        hs_net_out_path)


if __name__ == "__main__":

    # Change working directory to that of current file
    os.chdir(sys.path[0])

    # Retrieve config
    cfg = parse_args()

    # 
    patterns = [
        (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
        (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1),
    ]
    start_p = cfg['patterns_start_idx']
    n_p = cfg['n_patterns']
    start_eval = cfg['evals_start_idx']
    n_eval = cfg['n_evals']
    task = list(itertools.combinations(patterns, 4))[start_p:(start_p+n_p)]
    print('{time}: TASK: {task}, EVAL-IDs: {eval_ids}'.format(
        time=datetime.now().strftime("%H:%M:%S"),
        task=task,
        eval_ids=list(range(start_eval, start_eval+n_eval)),
    ))

    for train_patterns in task:
        for eval_id in range(start_eval, start_eval+n_eval):
            eval_train_patterns(train_patterns, eval_id, cfg)