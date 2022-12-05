from stable_baselines3 import DQN, PPO
import numpy as np

def trainPPO(env):
    gamma = 0
    learning_rate = 0.001
    buffer_size = 1000000
    batch_size = 32
    seed = 0
    episodes = 200
    policy_kwargs = dict(net_arch=[dict(pi=[400,300], vf=[400,300])])
    model_ppo = PPO("MlpPolicy", env, verbose=0, policy_kwargs=policy_kwargs, gamma = gamma, learning_rate=learning_rate,
                batch_size=batch_size, seed = seed)
    model_ppo.learn(total_timesteps=200*episodes)
    # model_ppo.save("stable_trend_dqn")
    return model_ppo, np.array(env.reward_hist), np.array(env.optimal_hist)