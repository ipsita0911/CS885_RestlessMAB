import sys
from env.rmab import MultiArmedBandit
from classical_RL.UCB import trainUCB
from classical_RL.Softmax import trainSoftmax
from classical_RL.epsilon_greedy import trainEpsilongreedy
from deep_RL.dqn import trainDQN
from deep_RL.ppo import trainPPO
from evaluate_model import *
from plot_everything import *

trend = True
volatile = False
arg1 = sys.argv[1]
arg2 = sys.argv[2]
if arg1 == "False":
    trend = False
if arg2 == "True":
    volatile = True
trendDict = {
    True: "with",
    False: "without"
}
volatileDict = {
    True: "high",
    False: "stable"
}

env_ep = MultiArmedBandit(trend=trend, volatile=volatile)
print("------------------Training Epsilon Greedy------------------")
Q_ep, reward_hist_ep, optimal_hist_ep = trainEpsilongreedy(env_ep, maxep=500)
plot_train(1, reward_hist_ep, optimal_hist_ep, "Decaying Epsilon Greedy", trendDict[trend], volatileDict[volatile])

env_sm = MultiArmedBandit(trend=trend, volatile=volatile)
print("------------------Training Softmax------------------")
Q_sm, reward_hist_sm, optimal_hist_sm = trainSoftmax(env_sm, maxep=500)
plot_train(2, reward_hist_sm, optimal_hist_sm, "SoftMax", trendDict[trend], volatileDict[volatile])

env_ucb = MultiArmedBandit(trend=trend, volatile=volatile)
print("------------------Training UCB------------------")
Q_ucb, reward_hist_ucb, optimal_hist_ucb = trainUCB(env_ucb, maxep=500)
plot_train(3, reward_hist_ucb, optimal_hist_ucb, "UCB", trendDict[trend], volatileDict[volatile])

env_dqn = MultiArmedBandit(trend=trend, volatile=volatile)
print("------------------Training DQN------------------")
model_dqn, reward_hist_dqn, optimal_hist_dqn = trainDQN(env_dqn)
plot_train(4, reward_hist_dqn, optimal_hist_dqn, "DQN", trendDict[trend], volatileDict[volatile])

env_ppo = MultiArmedBandit(trend=trend, volatile=volatile)
print("------------------Training PPO------------------")
model_ppo, reward_hist_ppo, optimal_hist_ppo = trainPPO(env_ppo)
plot_train(5, reward_hist_ppo, optimal_hist_ppo, "PPO", trendDict[trend], volatileDict[volatile])

plt.show()

env = MultiArmedBandit(trend=trend, volatile=volatile)
print("------------------Evaluating Epsilon Greedy------------------")
avg_optimal_rewards, avg_rewards_ep, avg_regret_ep, avg_frac_opt_action_ep, avg_frac_subopt_action_ep = evaluate_avg(env, maxep=100, Q=Q_ep)

env = MultiArmedBandit(trend=trend, volatile=volatile)
print("------------------Evaluating Softmax------------------")
avg_optimal_rewards, avg_rewards_sm, avg_regret_sm, avg_frac_opt_action_sm, avg_frac_subopt_action_sm = evaluate_avg(env, maxep=100, Q=Q_sm)

env = MultiArmedBandit(trend=trend, volatile=volatile)
print("------------------Evaluating UCB------------------")
avg_optimal_rewards, avg_rewards_ucb, avg_regret_ucb, avg_frac_opt_action_ucb, avg_frac_subopt_action_ucb = evaluate_avg(env, maxep=100, Q=Q_ucb)

env = MultiArmedBandit(trend=trend, volatile=volatile)
print("------------------Evaluating DQN------------------")
avg_optimal_rewards, avg_rewards_dqn, avg_regret_dqn, avg_frac_opt_action_dqn, avg_frac_subopt_action_dqn = evaluate_avg(env, maxep=100, model=model_dqn)

env = MultiArmedBandit(trend=trend, volatile=volatile)
print("------------------Evaluating PPO------------------")
avg_optimal_rewards, avg_rewards_ppo, avg_regret_ppo, avg_frac_opt_action_ppo, avg_frac_subopt_action_ppo = evaluate_avg(env, maxep=100, model=model_ppo)

plot_compare_metric(1, avg_regret_ep, avg_regret_sm, avg_regret_ucb, avg_regret_dqn, avg_regret_ppo,"Regret")
plot_compare_metric(2, avg_frac_opt_action_ep, avg_frac_opt_action_sm, avg_frac_opt_action_ucb, avg_frac_opt_action_dqn, avg_frac_opt_action_ppo, "Fraction of optimal Action")
plot_compare_metric(3, avg_frac_subopt_action_ep, avg_frac_subopt_action_sm, avg_frac_subopt_action_ucb, avg_frac_subopt_action_dqn, avg_frac_subopt_action_ppo, "Fraction of sub-optimal Action")
plot_compare_metric(4, numpy_ewma_vectorized(avg_rewards_ep, 20), numpy_ewma_vectorized(avg_rewards_sm, 20), numpy_ewma_vectorized(avg_rewards_ucb, 20), numpy_ewma_vectorized(avg_rewards_dqn, 20), numpy_ewma_vectorized(avg_rewards_ppo,20), "Rewards", optimal=numpy_ewma_vectorized(avg_optimal_rewards,20))
plt.show()

env = MultiArmedBandit(trend=trend, volatile=volatile)
print("----------------------One Game Run-------------------")
rewards_ep, all_arm_rewards_ep = evaluate_one_game(env, Q=Q_ep)
plot_one_game(1, rewards_ep, all_arm_rewards_ep, "EpsilonGreedy")

env = MultiArmedBandit(trend=trend, volatile=volatile)
rewards_sm, all_arm_rewards_sm = evaluate_one_game(env, Q=Q_sm)
plot_one_game(2, rewards_sm, all_arm_rewards_sm, "Softmax")

env = MultiArmedBandit(trend=trend, volatile=volatile)
rewards_ucb, all_arm_rewards_ucb = evaluate_one_game(env, Q=Q_ucb)
plot_one_game(3, rewards_ucb, all_arm_rewards_ucb, "UCB")

env = MultiArmedBandit(trend=trend, volatile=volatile)
rewards_dqn, all_arm_rewards_dqn = evaluate_one_game(env, model=model_dqn)
plot_one_game(4, rewards_dqn, all_arm_rewards_dqn, "DQN")

env = MultiArmedBandit(trend=trend, volatile=volatile)
rewards_ppo, all_arm_rewards_ppo = evaluate_one_game(env, model=model_ppo)
plot_one_game(5, rewards_ppo, all_arm_rewards_ppo, "PPO")

plt.show()