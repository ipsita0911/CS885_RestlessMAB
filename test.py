import numpy as np
import matplotlib.pyplot as plt 
from env.rmab import MultiArmedBandit

env1 = MultiArmedBandit(trend=False, volatile=False)
env2 = MultiArmedBandit(trend=True, volatile=False)
env3 = MultiArmedBandit(trend=False, volatile=True)
env4 = MultiArmedBandit(trend=True, volatile=True)

n_trials = 7
rew1 = np.zeros((n_trials, 200, 4))
rew2 = np.zeros((n_trials, 200, 4))
rew3 = np.zeros((n_trials, 200, 4))
rew4 = np.zeros((n_trials, 200, 4))
for k in range(n_trials):
    env1.seed(1111+k)
    env2.seed(1111+k)
    env3.seed(1111+k)
    env4.seed(1111+k)
    for j in range(4):
        for i in range(200):
            _,rew1[k][i][j],_,_ = env1.step(j)
            _,rew2[k][i][j],_,_ = env2.step(j)
            _,rew3[k][i][j],_,_ = env3.step(j)
            _,rew4[k][i][j],_,_ = env4.step(j)
        env1.reset()
        env2.reset()
        env3.reset()
        env4.reset()

linewidth = 1

plt.subplot(2,2,1)
plt.plot(np.average(rew1, axis=0), label = ['Arm 1', 'Arm 2', 'Arm 3', 'Arm 4'], linewidth=linewidth)
plt.title("Stable and without Trend")
plt.ylabel("value")
plt.ylim([-100, 100])
plt.legend()

plt.subplot(2,2,2)
plt.plot(np.average(rew2, axis=0), label =  ['Arm 1', 'Arm 2', 'Arm 3', 'Arm 4'], linewidth=linewidth)
plt.title("Stable and with Trend")
plt.ylabel("value")
plt.ylim([-100, 100])
plt.legend()

plt.subplot(2,2,3)
plt.plot(np.average(rew3, axis=0), label = ['Arm 1', 'Arm 2', 'Arm 3', 'Arm 4'], linewidth=linewidth)
plt.title("Volatile and without Trend")
plt.xlabel("trial")
plt.ylabel("value")
plt.ylim([-100, 100])
plt.legend()

plt.subplot(2,2,4)
plt.plot(np.average(rew4, axis=0), label =  ['Arm 1', 'Arm 2', 'Arm 3', 'Arm 4'], linewidth=linewidth)
plt.title("Volatile and with Trend")
plt.xlabel("trial")
plt.ylabel("value")
plt.ylim([-100, 100])
plt.legend()

plt.show()
