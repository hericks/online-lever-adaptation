# Online Lever Adaptation

This repository aims to bundle all the code and experiments for my Master's thesis on online adaptation in multi-agent reinforcement learning (MARL). The thesis is jointly supervised by [Jakob Foerster](https://www.jakobfoerster.com) with support from his research group at [FLAIR](https://foersterlab.com) and [Arnaud Doucet](https://www.stats.ox.ac.uk/~doucet/) from the Department of Statistics at the University of Oxford.

The top-level scripts describe some of the basic functionality and structure of this project.
- **[01_step_through_env.py](https://github.com/hericks/online-lever-adaptation/blob/main/01_step_through_env.py)** shows how to initialize and step through an iterated lever environment with custom parameters and partner policies.
- **[02_q_learning.py](https://github.com/hericks/online-lever-adaptation/blob/main/02_q_learning.py)** combines the environment with a learner of class `DQNAgent` to perform vanilla **[q-learning](https://en.wikipedia.org/wiki/Q-learning)**.
- **[03_es_meta_learning.py](https://github.com/hericks/online-lever-adaptation/blob/main/03_es_meta_learning.py)** exemplifies how the `OpenES` class - which implements the evolution strategies algorithm **[Open-ES](https://arxiv.org/abs/1703.03864)** - can be used to learn initial network weights capable of remembering a fixed partner pattern of length three. 
- **[04_es_learn_history_representations.py](https://github.com/hericks/online-lever-adaptation/blob/main/04_es_learn_history_representations.py)** shows how evolution strategies can be used to learn the parameters of a LSTM giving a history representation suitable for effective q-learning. 
