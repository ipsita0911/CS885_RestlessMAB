import numpy as np
import gym 

MAXTIMESTEP = 200

def trainSoftmax(env, maxep = 1000):
    env.reset()
    e = 0
    tou = 5
    decay = 0.1
    ARMS = env.action_space.n
    Q = np.zeros((MAXTIMESTEP, ARMS))
    N = np.zeros((MAXTIMESTEP , ARMS))
    while(e < maxep):
        env.seed(1111+e)
        env.reset()
        done = False
        while not done:
            timestep = env.timestep
            probs = np.exp(Q[timestep]/tou)/np.sum(np.exp(Q[timestep]/tou))
            a = np.random.choice(ARMS, 1, p = probs)
            _, R, done, _ = env.step(a[0])
            N[timestep][a] += 1
            Q[timestep][a] = Q[timestep][a] + (R - Q[timestep][a])/N[timestep][a]
        # if e < 50:
        #     tou -= decay
        e += 1
    return Q, np.array(env.reward_hist), np.array(env.optimal_hist)