from tqdm import tqdm
import numpy as np

MAXSTEPS = 200
ARMS = 4

def evaluate_avg(env, maxep, Q = None, model = None):
    assert ((Q is not None) + (model is not None)) == 1
    env.reset()
    rewards = np.zeros((maxep, MAXSTEPS))
    optimal_reward = np.zeros((maxep, MAXSTEPS))
    regret = np.zeros((maxep, MAXSTEPS))
    frac_opt_action = np.zeros((maxep, MAXSTEPS))
    frac_subopt_action = np.zeros((maxep, MAXSTEPS))
    all_arm_rewards = np.zeros((maxep, MAXSTEPS, ARMS))

    for e in tqdm(range(maxep)):
        env.seed(1111+e)
        obs = env.reset()
        done = False
        while not done:
            timestep = env.timestep
            if Q is not None:
                a = np.argmax(Q[timestep])
            else:
                a, _ = model.predict(obs)
            obs, rewards[e][timestep], done, info = env.step(a)
            all_arm_rewards[e][timestep][:] = env.rewards
            optimal_reward[e][timestep] = env.optimal_reward
            maxx = np.max(all_arm_rewards[e][timestep][:])
            minn = np.min(all_arm_rewards[e][timestep][:])

            if timestep == 0:
                frac_opt_action[e][timestep] = (rewards[e][timestep] == maxx)
                frac_subopt_action[e][timestep] = (rewards[e][timestep] - minn)/(maxx - minn)
                regret[e][timestep] = optimal_reward[e][timestep] - rewards[e][timestep]
            else:
                frac_opt_action[e][timestep] = (frac_opt_action[e][timestep-1]*timestep +(rewards[e][timestep] == maxx))/(timestep+1)
                frac_subopt_action[e][timestep] = (frac_subopt_action[e][timestep-1]*timestep + (rewards[e][timestep] - minn)/(maxx - minn))/(timestep+1)
                regret[e][timestep] = regret[e][timestep-1] + (maxx - rewards[e][timestep])

    avg_rewards = np.mean(rewards, axis = 0)
    avg_optimal_rewards = np.mean(optimal_reward, axis = 0)
    avg_regret = np.mean(regret, axis = 0)
    avg_frac_opt_action = np.mean(frac_opt_action, axis = 0)
    avg_frac_subopt_action = np.mean(frac_subopt_action, axis = 0)

    return avg_optimal_rewards, avg_rewards, avg_regret, avg_frac_opt_action, avg_frac_subopt_action


def evaluate_one_game(env, seed = 1111, Q = None, model = None):
    assert ((Q is not None) + (model is not None)) == 1
    env.reset()
    rewards = np.zeros(MAXSTEPS)
    all_arm_rewards = np.zeros((MAXSTEPS, ARMS))

    env.seed(seed)
    obs = env.reset()
    done = False

    while not done:
        timestep = env.timestep
        if Q is not None:
            a = np.argmax(Q[timestep])
        else:
            a, _ = model.predict(obs)
        obs, rewards[timestep], done, info = env.step(a)
        all_arm_rewards[timestep][:] = env.rewards

    return rewards, all_arm_rewards    