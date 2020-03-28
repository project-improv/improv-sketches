import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from compare_opt import CompareOpt
from synthetic import offline_decay, gen_theta

from data_gen_network import DataGenerator

"""
Generate multiple samples of data, fit model, and save θ.
"""

sns.set()

params = {  # For both data generation and fitting.
    'N': 20,
    'M': 5000,
    'dh': 2,
    'dt': 1,
    'ds': 1,
    'λ1': 0.4,
    'λ2': 0.05
}

params['M_lim'] = params['M']
params['N_lim'] = params['N']

params_θ = {
    'seed': 0,
    'p_inh': 0.5,
    'p_rand': 0.,
    'base': 0.4,
    'connectedness': 3,
}

gen = DataGenerator(params=params, params_θ=params_θ)

gnd_p = np.sum(np.abs(gen.theta['w']) > 0)
gnd_n = params['N_lim']**2 - gnd_p

optimizers = [{'name': 'adam', 'step_size': offline_decay(5e-3, 1e4, 0.1), 'offline': True}]


def sample(params, rep):
    θws = np.zeros((rep, params['N_lim'], params['N_lim']))
    s = np.zeros((params['ds'], params['M']), dtype=np.float32)
    n_spikes = np.zeros((rep, params['N_lim']))
    for i in range(rep):
        # print(f'Repeat {i}/{rep}.')
        r, y = gen.gen_spikes(params=params, seed=10 * i)
        c = CompareOpt(params, y, s)
        c.run(optimizers, theta=gen_theta(params), gnd_data=gen.theta, use_gpu=True, save_theta=1000,
              iters_offline=10000, hamming_thr=0.1, verbose=False)
        θws[i, ...] = c.theta['adam_offline'][-1]['w']
        n_spikes[i, :] = np.sum(y, axis=1)
    return θws, n_spikes


def vary(to_vary, space):
    out = dict()
    for i, s in enumerate(space):
        print(f'{to_vary}={s}')
        p = params.copy()
        p[to_vary] = s
        p['M_lim'] = p['M']
        out[s] = sample(p, 8)
    return out

sp = [500, 1000, 2000, 5000, 10000, 20000]

raw = vary('M', sp)

out = raw[5000][0]

sd = np.sqrt(np.var(out, axis=0))
mean = np.abs(np.mean(out, axis=0))
mae = np.mean(np.abs(out - gen.theta['w']), axis=0)
mask = np.abs(np.mean(out, axis=0)) > 0.1 * np.max(out)

print('CV median:', np.median((sd/mean)[mask]))

fig, ax = plt.subplots(figsize=(10, 4), dpi=200, ncols=2)
ai = ax[0].imshow(sd/0.1, vmin=0, vmax=1.0)
ax[0].grid(False)
ax[0].set_title(f'θ_w Normalized SD, median={np.median((sd/mean)[mask]): .3f}, clipped to >0.1 max')
fig.colorbar(ai, ax=ax[0])


percent = mae / np.max(np.abs(gen.theta['w']))
ai = ax[1].imshow(percent, vmin=0, vmax=1.)
ax[1].grid(False)
ax[1].set_title(f'θ_w Normalized Error (to max entry), mean={np.mean(mae): .3f}')
fig.colorbar(ai, ax=ax[1])
fig.tight_layout()

plt.show()

# Calculate SD
def process(th):
    sd = np.sqrt(np.var(th, axis=0))
    mean = np.abs(np.mean(th, axis=0))
    mae = np.mean(np.abs(th - gen.theta['w']), axis=0)
    mask = np.abs(np.mean(th, axis=0)) > 0.1 * np.max(th)
    return np.median((sd / mean)[mask]), np.mean(mae)

to_plot = [process(x[0])[0] for x in raw.values()]

import pickle
with open(f'n{params["N"]}_b0.5.pk', 'wb') as f:
   pickle.dump(raw, f)