import numpy as np
import matplotlib.pyplot as plt 
from utils import numpy_ewma_vectorized

def plot_train(fig_no, reward_hist, optimal_hist, model, trend, volatile):
    linewidth = 2
    plt.figure(fig_no)
    plt.plot(numpy_ewma_vectorized(reward_hist, 30), label='Training Reward', linewidth=linewidth)
    plt.plot(numpy_ewma_vectorized(optimal_hist, 30), label='True Reward', linewidth=linewidth)
    plt.title(f"Training and reward for {model} on environment \n {trend} trend & {volatile} volatility")
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.grid()
    plt.legend(loc="upper right")

def plot_compare_metric(fig_no, eps, sm, ucb, dqn, ppo, metric, optimal = None):
    plt.figure(fig_no)
    plt.plot(eps, label=f'{metric} for EpsilonGreedy')
    plt.plot(sm, label=f'{metric} for Softmax')
    plt.plot(ucb, label=f'{metric} for UCB')
    plt.plot(dqn, label=f'{metric} for DQN')
    plt.plot(ppo, label=f'{metric} for PPO')
    if optimal is not None:
        plt.plot(optimal, label=f'Optimal {metric}')
    plt.grid()
    plt.legend(loc="upper right")

def plot_one_game(fig_no, rewards, all_arm_rewards, model):
    linewidth = 2
    plt.figure(fig_no)
    plt.plot(rewards, label=f'One Game for {model}', linewidth=linewidth)
    for i in range(4):
        plt.plot(all_arm_rewards[:,i], '--', label='Arm '+str(i+1), linewidth=1)
    plt.xlabel("trials")
    plt.ylabel("reward")
    plt.grid()
    plt.legend(loc="lower right")