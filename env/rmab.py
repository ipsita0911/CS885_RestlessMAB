import numpy as np
import gym
from gym import spaces
import random
import os

ARMS = 4
INITIAL_MU = np.array([-60, -20, 20, 60])
DECAY_PARAM = 0.9836
TREND = np.array([1, 1, -1, -1])
REWARD_VARIANCE = 4
STABLE_INNOVATION_VARIANCE = 4
VOLATILE_INNOVATION_VARIANCE = 16
TIMESTEPS = 200
MEMOMY_LEN = 8
MEMORY_DECAY = 0.9


# Four armed bandit
class MultiArmedBandit(gym.Env):
    def __init__(self, trend = False, volatile = False, seed = 1):
        super(MultiArmedBandit, self).__init__()
        self.action_space = spaces.Discrete(ARMS,)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape = (ARMS, MEMOMY_LEN), dtype=np.float32)
        self.decay = DECAY_PARAM
        self.memory_decay = MEMORY_DECAY

        self.mu = INITIAL_MU

        self.volatile = volatile
        self.innovation_var = STABLE_INNOVATION_VARIANCE
        self.reward_var = REWARD_VARIANCE

        self.rewards = INITIAL_MU

        self.ep_opti_hist = []
        self.optimal_hist = []

        self.ep_reward_hist = []
        self.reward_hist = []

        self.state = np.zeros((ARMS, MEMOMY_LEN))

        self.is_trend = trend

        if trend:
            self.trend = TREND
        else:
            self.trend = np.zeros(ARMS)

        self.timestep = 0

    def seed(self, seednum = 1):
        random.seed(seednum)
        np.random.seed(seednum)
        os.environ['PYTHONHASHSEED'] = str(seednum)

    def step(self, action):
        assert action in list(range(0, ARMS))
        self.rewards = np.random.normal(self.mu, self.reward_var)
        curr_reward = self.rewards[action]
        self.optimal_arm = np.argmax(self.rewards)
        self.optimal_reward = np.max(self.rewards)

        temp = self.state[action].tolist()
        temp.pop(0)
        temp.append(curr_reward)
        self.state[action] = np.array(temp)

        self.innovation_var = STABLE_INNOVATION_VARIANCE
        if (self.volatile & (((self.timestep > 50) & (self.timestep <= 100)) | ((self.timestep > 150) & (self.timestep <= 200))) ):
            self.innovation_var = VOLATILE_INNOVATION_VARIANCE

        new_mu = self.decay * self.mu + self.trend + np.random.normal(0, self.innovation_var, size=(ARMS))
        self.mu = new_mu

        self.timestep += 1

        done = False
        self.ep_reward_hist.append(curr_reward)
        self.ep_opti_hist.append(self.optimal_reward)

        if self.timestep == TIMESTEPS:
            done = True
            self.reward_hist.append(np.sum(self.ep_reward_hist))
            self.optimal_hist.append(np.sum(self.ep_opti_hist))
            self.ep_reward_hist = []
            self.ep_opti_hist = []
            self.timestep = 0
        
        return self.state, curr_reward, done, {}

    def reset(self):
        self.mu = INITIAL_MU
        self.rewards = INITIAL_MU
        #-----------Non numbered Arms------------------
        # index = np.arange(ARMS)
        # random.shuffle(index)
        # self.mu = self.mu[index]
        # self.trend = self.trend[index]
        #-----------Random Shuffling between given sets--------------
        # random.shuffle(self.mu)
        # random.shuffle(self.trend)
        #-----------Randomly Chosing mu and trend value-------------
        # self.mu = np.random.normal(0, 16, size=(ARMS))
        # self.trend = np.random.normal(0, 1, size=(ARMS))

        self.innovation_var = STABLE_INNOVATION_VARIANCE

        self.timestep = 0
        self.state = np.zeros((ARMS, MEMOMY_LEN))
                
        self.ep_opti_hist = []
        # self.optimal_hist = []

        self.ep_reward_hist = []
        # self.reward_hist = []

        return self.state
