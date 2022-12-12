import numpy as np
import gym

MAXTIMESTEP = 200

def trainUCB(env, maxep = 1000):
    env.reset()
    e = 0
    c = 20
    ARMS = env.action_space.n
    Q = np.zeros((MAXTIMESTEP, ARMS))
    N = np.zeros((MAXTIMESTEP , ARMS))
    while(e < maxep):
        env.seed(1111+e)
        env.reset()
        done = False
        while not done:
            timestep = env.timestep
            if e < ARMS :
                a = e
            else:
                U = c * np.sqrt(np.log(e)/N[timestep])
                a = np.argmax(Q[timestep] + U)
            _, R, done, _ = env.step(a)
            N[timestep][a] += 1
            Q[timestep][a] = Q[timestep][a] + (R - Q[timestep][a])/N[timestep][a]
        e += 1
    return Q, np.array(env.reward_hist), np.array(env.optimal_hist)