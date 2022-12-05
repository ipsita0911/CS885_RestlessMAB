from stable_baselines3 import DQN
import numpy as np

def trainDQN(env):
    gamma = 0
    exploration_fraction = 0.2
    exploration_initial_eps = 1.0
    exploration_final_eps = 0.05
    learning_rate = 0.01
    buffer_size = 1000000
    learning_starts = 10000
    batch_size = 32
    seed = 0
    verbose = 0
    episodes = 200
    policy_kwargs = dict(net_arch=[400,300])
    model_dqn = DQN('MlpPolicy', env, verbose=verbose, exploration_initial_eps=exploration_initial_eps, exploration_final_eps=exploration_final_eps, 
                exploration_fraction=exploration_fraction, gamma=gamma, learning_rate=learning_rate, 
                buffer_size=buffer_size, learning_starts=learning_starts, batch_size=batch_size, 
                seed = seed, target_update_interval=250, create_eval_env=True, policy_kwargs = policy_kwargs)
    # Train the agent
    model_dqn.learn(total_timesteps=int(200*episodes))
    # Save the agent
    # model_dqn.save("dqn")
    return model_dqn, np.array(env.reward_hist), np.array(env.optimal_hist)
    
