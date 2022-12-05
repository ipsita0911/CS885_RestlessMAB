import numpy as np
import matplotlib.pyplot as plt 
from utils import numpy_ewma_vectorized_v2

def plot_train(fig_no, reward_hist, optimal_hist, model, trend, volatile):
    linewidth = 2
    plt.figure(fig_no)
    plt.plot(numpy_ewma_vectorized_v2(reward_hist, 20), label='Training Reward', linewidth=linewidth)
    plt.plot(numpy_ewma_vectorized_v2(optimal_hist, 20), label='True Reward', linewidth=linewidth)
    plt.title(f"Training and reward for {model} on environment \n {trend} trend & {volatile} volatility")
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.legend(loc="upper right")
    # plt.savefig('dqn_training_trend_high.eps', format='eps')
    # plt.savefig('dqn_training_trend_high.jpg', dpi=300)

def plot_compare_metric(eps, sm, ucb, dqn, ppo, metric, optimal = None):
    plt.figure(figsize=(12,10))
    plt.plot(eps, label=f'{metric} for EpsilonGreedy')
    plt.plot(sm, label=f'{metric} for Softmax')
    plt.plot(ucb, label=f'{metric} for UCB')
    plt.plot(dqn, label=f'{metric} for DQN')
    plt.plot(ppo, label=f'{metric} for PPO')
    if optimal is not None:
        plt.plot(ppo, label=f'Optimal {metric}')
    plt.legend(loc="upper right")

def plot_one_game(rewards, all_arm_rewards, model):
    linewidth = 4
    plt.figure(figsize=(12, 10))
    plt.rcParams.update({'font.size': 22})
    plt.plot(rewards, label=f'One Game for {model}', linewidth=linewidth)
    for i in range(4):
        plt.plot(all_arm_rewards[:,i], '--', label='Arm '+str(i+1), linewidth=3)
    plt.xlabel("Time Steps")
    plt.ylabel("Reward")
    plt.legend(loc="upper right")