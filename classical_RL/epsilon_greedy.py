import numpy as np
import gym

MAXTIMESTEP = 200

def trainEpsilongreedy(env, maxep = 500, epsilon = 0.1, decay = 0, isdecaylinear = True):
    env.reset()
    e = 0
    a = 0
    ARMS = env.action_space.n
    Q = np.zeros((MAXTIMESTEP, ARMS))
    N = np.zeros((MAXTIMESTEP , ARMS))

    while(e < maxep):
        env.seed(1111+e)
        env.reset()
        done = False
        while not done:
            timestep = env.timestep
            if np.random.uniform(0, 1) >= epsilon:
                a = np.argmax(Q[timestep])
            else:
                a = np.random.randint(0, ARMS)
            _, R, done,_ = env.step(a)
            if isdecaylinear:
                epsilon -= decay
            else:
                epsilon *= decay
            N[timestep][a] += 1
            Q[timestep][a] = Q[timestep][a] + (R - Q[timestep][a])/N[timestep][a]
        e += 1
    return Q, np.array(env.reward_hist), np.array(env.optimal_hist)

def decay(env, maxep = 500):
    env.reset()
    e = 0
    a = 0
    ARMS = env.action_space.n
    Q = np.zeros((MAXTIMESTEP, ARMS))
    N = np.zeros((MAXTIMESTEP , ARMS))
    
    while(e < maxep):
        env.seed(1111+e)
        env.reset()
        done = False
        while not done:
            timestep = env.timestep
            epsilon = 1/(timestep + 1.0)
            if np.random.uniform(0, 1) >= epsilon:
                a = np.argmax(Q[timestep])
            else:
                a = np.random.randint(0, ARMS)
            _, R, done,_ = env.step(a)
            N[timestep][a] += 1
            Q[timestep][a] = Q[timestep][a] + (R - Q[timestep][a])/N[timestep][a]
        e += 1
    return Q, np.array(env.reward_hist), np.array(env.optimal_hist)