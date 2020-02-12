import time

import seaborn as sns

sns.set()

import matplotlib.pyplot as plt
import numpy as np
import pickle

from glm_jax import GLMJax

"""
Create a log-likelihood graph over time using different optimizers.

"""

params = {'dh': 10, 'ds': 8, 'dt': 0.1, 'n': 0, 'N_lim': 200, 'M_lim': 3000}

optimizers = [
    {'name': 'sgd', 'step_size': 1e-5},
    # {'name': 'sgd', 'step_size': 1e-4, 'offline': True},
    # {'name': 'momentum', 'step_size': 1e-5, 'mass': 0.9},
    # {'name': 'momentum', 'step_size': 1e-5, 'mass': 0.9, 'offline': True},
    # {'name': 'nesterov', 'step_size': 1e-5, 'mass': 0.9},
    # {'name': 'adam', 'step_size': 1e-4},
    # {'name': 'adam', 'step_size': 1e-4, 'offline': True},
    # {'name': 'adagrad', 'step_size': 1e-5, 'momentum': 0.9},
    # {'name': 'rmsprop', 'step_size': 4e-5},
    # {'name': 'rmsprop_momentum', 'step_size': 1e-5},
    # {'name': 'rmsprop_momentum', 'step_size': 1e-5, 'offline': True},
    # {'name': 'sm3', 'step_size': 1e-5},
]

def conv_ys(y, s):
    """ Assuming that M = 100. """
    y_all = np.zeros((y[-1].shape[0], len(y)), dtype=np.float32)

    first_hundred = y[98]
    y_all[:first_hundred.shape[0], :98+1] = y[98]

    for t in range(99, len(y)):
        y_curr = y[t][:, -1]
        n_neu = y_curr.shape[0]
        y_all[:n_neu, t] = y_curr

    s_all = np.zeros((s[0].shape[0], len(y)))

    first_hundred = s[98]
    s_all[:first_hundred.shape[0], :98+1] = s[98]

    for t in range(99, len(y)):
        s_curr = s[t][:, -1]
        n_s = s_curr.shape[0]
        y_all[:n_s, t] = s_curr

    return y_all, s_all


with open('tbif_batch_for_analysis.pk', 'rb') as f:
    y, s = pickle.load(f)

y_all, s_all = conv_ys(y, s)

rpf = 1
ts = dict()
lls = np.zeros((rpf * len(y), len(optimizers)))

for i, opt in enumerate(optimizers):
    offline = opt.get('offline', False)
    if 'offline' in opt:
        del opt['offline']

    params['M_lim'] = 3000 if offline else 100

    model = GLMJax(params, optimizer=opt)
    t0 = time.time()

    for t in range(len(y)):
        for rep in range(rpf):
            if not offline:
                lls[t * rpf + rep, i] = model.fit(y[t], s[t])
            else:
                lls[t * rpf + rep, i] = model.fit(y_all, s_all)

        if t % 100 == 0:
            print(f"{opt['name']}, step: {t}")

    if offline:
        opt['offline'] = True

    ts[opt['name']] = time.time() - t0

# Plot
fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
colors = ['r', 'g', 'b', 'k']
for i in range(lls.shape[1]):
    if 'offline' in optimizers[i]:
        args = (lls[10:, i], colors[i//2]+'--')
    else:
        args = (lls[10:, i], colors[i//2]+'-')
    ax.plot(*args, linewidth=1, label=", ".join([f'{k}: {v}' for k, v in optimizers[i].items()])[6:], alpha=0.8)

ax.set_xlabel('Iterations')
ax.set_ylabel('-log-likelihood')
ax.set_title('Optimizers Comparison')
plt.legend()
plt.savefig('optimizer_comp.png')
plt.show()
