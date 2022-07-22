# Online Lever Adaptation

This repository aims to bundle all the code and experiments for my Master's thesis on online adaptation in multi-agent reinforcement learning (MARL). The thesis is jointly supervised by [Jakob Foerster](https://www.jakobfoerster.com) with support from his research group at [FLAIR](https://foersterlab.com) and [Arnaud Doucet](https://www.stats.ox.ac.uk/~doucet/) from the Department of Statistics at the University of Oxford.

The top-level scripts describe some of the basic functionality and structure of this project.
- In **[01_step_through_env.py](https://github.com/hericks/online-lever-adaptation/blob/main/01_step_through_env.py)** we show how to initialize an iterated lever environment with custom parameters and partner policies.
- In **[02_q_learning.py](https://github.com/hericks/online-lever-adaptation/blob/main/02_q_learning.py)** we combine the environment with the `DQNAgent` class to perform vanilla [q-learning](https://en.wikipedia.org/wiki/Q-learning).
