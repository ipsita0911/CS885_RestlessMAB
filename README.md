Final Project for CS885 at University of Waterloo
<!-- <<<<<<< HEAD -->
# Restless Multi-Armed Bandits
<!-- ======= -->
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)  [![Version](https://badge.fury.io/gh/tterb%2FHyde.svg)](https://badge.fury.io/gh/tterb%2FHyde)

The Restless Multi-Armed Bandit Problem (RMABP) is a game between a player and an environment. There are K arms and the state of each arm keeps 
evolving according to an underlying distribution at each timestep of the episode (one full play of the game). For every timestep, 
the player pulls one of the arms and receives a reward. The goal of the game is to maximize the reward received over T time steps. This implementation
 compares the performance of different classical and deep RL techniques for the RMAB problem.
 
 ## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
1. Python
2. Reinforcement Learning methods - epsilon-greedy, softmax, UCB, DQN, PPO

### Installing

#### Downloading source code
Enter the following command in your console to clone the repository.
```bash
$> git clone [https://github.com/UCHI-DB/xsystem-ng.git](https://github.com/ipsita0911/CS885_RestlessMAB.git)
```

#### Run for a particular environment
As described in Report, there can be 4 combination of trend and stability parameters. To get performances of different techniques for a particular
 setting run - 
```bash
$> python run.py [trend] [volatile]
```
This command takes two binary arguments, for different setting of trend and volatility. For example, to run the code for a volatile environment without trend, we run

```bash
$> python run.py False True
```

The following table gives the arguments for different environment settings - 
| Environment Type  | Code | Trend | Volatile |
| ------------- | ------------- | ------------- | ------------- |
| Stable without Trend  | SN | False | False |
| Stable with Trend  | ST | True | False |
| Volatile without Trend  | VN | False | True |
| Volatile with Trend  | VT | True | True |
