# imitation-learning

This is a repository for a 2019 project on imitation learning for Computation & the Brain at Columbia University. This work was done by Jenny Chen, Jacob Austin, and Matthew Bowers.

## oneshot.py

MAML implementation for imitation learning written from scratch, along with fine tuning based on One-Shot Visual Imitation Learning via Meta-Learning, Finn et al. Included a reproduction of the architecture used for visual inputs.

## oneshot-nostate.py

Same as oneshot.py but uses only visual input without direct access to the state information. 

## bipedal\_ds.py

Implementation of a dataset class for loading videos learned by our RL algorithm.

## openai-gym.ipynb

This is a modified version of a script running a DDPG model-free RL algorithm in the OpenAI Gym Bipedal Walker environment. Most of this code was taken from an online implementation.

## vis.py 

This is a visualization script for viewing the hidden layers and gradients of our MAML implementation. This allows us to view neurons that respond to particular inputs and are responsible for adaptation in the MAML step.

## k-armed-bandit.py

`k-armed-bandit.py` compares the average reward for epsilon-greedy k-armed bandit algorithms over some number of iterations. These algorithms included the classic epsilon-greedy algorithm, the UCB algorithm, and the gradient bandits algorithm.

![epsilon-greedy.png](https://raw.githubusercontent.com/ja3067/reinforcement-learning/master/epsilon-greedy.png)

This is related to the DeepMind paper presented by Lilicrap at the Simon's Institute on k-armed bandits and reinforcement learning using LSTMs.

## maml

The `maml` directory contains an implementation of the MAML algorithm from Finn et al. (2017) that uses gradient descent to optimize fitness over many different tasks under a small change in the parameters of a neural network. This implementation currently runs on the sinusoid dataset shown in the original paper, but is currently being adapted to RL tasks.

## inverted-pendulum.py

This script solves the optimal control problem using reinforcement Q-learning in the OpenAI Gym environment. To use, `pip install gym`, then run the script. The manual version uses a simple optimal control scheme instead of reinforcement learning. This is not a deep learning approach, but instead learns a binned Q function. 

## project status

All code in the maml directory was written from scratch as part of our visual imitation learning project. The `inverted-pendulum.py` and `k-armed-bandit.py` code is adapted from online code, but significantly modified. 
