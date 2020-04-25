import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from GLM.utils import *
from compare_opt import CompareOpt

sns.set()

"""
Run processed data from improv data sharing @ Box.
"""

y = np.loadtxt('end_spikes.txt')
s_raw = np.loadtxt('stim_data.txt')

N, M = y.shape
ds = len(np.unique(s_raw[:, 0]))

params = {  # For both data generation and fitting.
        'N': N,
        'N_lim': N,
        'M': M,
        'M_lim': M,
        'dh': 10,
        'dt': 1,
        'ds': ds,
        'λ1': 1,
        'λ2': 0.0
}

optimizers = [
    {'name': 'nesterov', 'step_size': 10, 'mass': 0.99, 'offline': True}
]

neuron_start = (y != 0).argmax(axis=1)  # First non-zero
indicator = np.zeros((N, M))
for i in range(N):
    indicator[i, neuron_start[i]:] = 1

s = np.zeros((ds, M))
for i in range(M):
    s[int(s_raw[i, 0]), i] = 1


c = CompareOpt(params, y, s)

lls = c.run(optimizers, theta=gen_rand_theta(params), resume=True,  use_gpu=True, save_theta=1000,
            save_grad=None, iters_offline=15000, indicator=indicator, hamming_thr=0.1, rpf=2, verbose=1000)

w = np.asarray(c.theta['nesterov_offline'][-1]['w'])
k = np.asarray(c.theta['nesterov_offline'][-1]['k'])
h = np.asarray(c.theta['nesterov_offline'][-1]['h'])
b = np.asarray(c.theta['nesterov_offline'][-1]['b'])

fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
im = ax.imshow(np.abs(h), vmax=0.5*np.max(np.abs(h)), vmin=0)
ax.grid(0)
fig.colorbar(im)
plt.show()

with open('theta_real.pk', 'wb') as f:
    pickle.dump({k: np.asarray(v, dtype=np.float64) for k, v in c.theta['nesterov_offline'][-1].items()}, f)

with open('params_real.pk', 'wb') as f:
    pickle.dump(params, f)
