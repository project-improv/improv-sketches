import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

from compare_opt import CompareOpt

sns.set()

"""
Run GLMJax using synthetic data from `data_gen.py`.
"""

params_raw = pickle.loads(Path('params_dict.pickle').read_bytes())
window_size = 10000
params = {
    'N_lim': params_raw['numNeurons'],
    'M_lim': window_size,
    'dh': params_raw['hist_dim'],
    'dt': 1,  # params_raw['dt'],
    'ds': 1,
    'λ1': 0.02,
    'λ2': 0.05
}

print('Loading data.')
gnd = pickle.loads(Path('theta_dict.pickle').read_bytes())
print('Number of non-zero values in gnd θ_w:', np.sum(np.abs(gnd['w']) > 0))

y = np.loadtxt('data_sample.txt').astype(np.float32)  # [:, :10000]
s = np.zeros((params['ds'], y.shape[1]), dtype=np.float32)
c = CompareOpt(params, y, s)


def gen_theta(p):
    np.random.seed(0)
    w = 1 / 20 * np.random.random((p['N_lim'], p['N_lim']))
    k = np.zeros((p['N_lim'], p['ds']))
    b = np.random.random((p['N_lim'], 1))
    h = 1 / 20 * np.random.random((p['N_lim'], p['dh']))
    return {'h': h, 'w': w, 'b': b, 'k': k}


θ_init = gen_theta(params)


def online_decay(step_size, decay_steps, decay_rate):
    def schedule(i):
        return (i < window_size) * i / window_size * step_size \
               + (i >= window_size) * step_size * decay_rate ** ((i - window_size) / decay_steps)

    return schedule


def offline_decay(step_size, decay_steps, decay_rate):
    def schedule(i):
        return step_size * decay_rate ** (i / decay_steps)

    return schedule


optimizers = [
    # {'name': 'adam', 'step_size': online_decay(2e-2, 2e4, 0.1)},
    {'name': 'adam', 'step_size': offline_decay(1e-2, 2e4, 0.1), 'offline': True},
]

lls = c.run(optimizers, theta=gen_theta(params), resume=True, gnd_data=gnd, use_gpu=True, save_theta=1000,
            save_grad=None, iters_offline=5000)

# %% Plot θ
fig, ax = plt.subplots(figsize=(8, 12), nrows=3, ncols=2, dpi=200)
ax = ax.reshape(ax.size)


def gen_plot(i, data, title):
    scale = np.max(np.abs(data))
    g = ax[i].imshow(data, vmin=-scale, vmax=scale)
    ax[i].grid(0)
    ax[i].set_title(title)
    fig.colorbar(g, ax=ax[i])

    g.set_cmap('bwr')


opt = 'adam_offline'

gen_plot(0, gnd['w'], 'Ground Truth w')
gen_plot(1, c.theta[opt][-1]['w'], 'Fitted w')

gen_plot(2, gnd['h'], 'Ground Truth h')
gen_plot(3, c.theta[opt][-1]['h'][:, ::-1], 'Fitted h')

gen_plot(4, gnd['b'][:, np.newaxis], 'Ground Truth b')
gen_plot(5, c.theta[opt][-1]['b'], 'Fitted b')
ax[4].get_xaxis().set_visible(False)
ax[5].get_xaxis().set_visible(False)

plt.tight_layout()
plt.show()

# %% Animation of Weights

# from matplotlib.animation import FuncAnimation
# Path('nesterov.gif').unlink(missing_ok=True)
# pos = []
# neg = []
# thr = 0.
#
# ham_base = (np.abs(gnd['w'].reshape(gnd['w'].size)) > thr * np.max(np.abs(gnd['w']))).astype(np.int)
# for th in c.theta['adam']:
#     arr = (np.abs(th['w']).reshape(th['w'].size) > thr*np.max(np.abs(th['w']))).astype(np.int) - ham_base
#     pos.append(np.sum(arr == 1))
#     neg.append(np.sum(arr == -1))
#
# fig, ax = plt.subplots(dpi=300)
#
#
# ax.semilogy(pos, label='False Positive (in fit, not gnd)')
# ax.semilogy(neg, label='False Negative (not in fit, in gnd)')
# ax.set_xlabel('Iterations (x1e3)')
# ax.set_title(f'FP/FN Fitted Model (N=100, 404 non-zero weights, 2e5 iters, Adam, Thr={thr}')
# fig.set_tight_layout(True)
#
#
# def update(i):
#     ax.clear()
#     pos = []
#     neg = []
#     thr = i*0.05
#     ham_base = (np.abs(gnd['w'].reshape(gnd['w'].size)) > thr * np.max(np.abs(gnd['w']))).astype(np.int)
#     for th in c.theta['adam']:
#         arr = (np.abs(th['w']).reshape(th['w'].size) > thr * np.max(np.abs(th['w']))).astype(np.int) - ham_base
#         pos.append(np.sum(arr == 1))
#         neg.append(np.sum(arr == -1))
#
#     ax.semilogy(pos, label='False Positive (in fit, not gnd)')
#     ax.semilogy(neg, label='False Negative (not in fit, in gnd)')
#     ax.set_title(f'FP/FN Fitted Model (N=100, 404 non-zero weights, 2e5 iters, Adam, Thr={thr:0.2f}')
#     ax.axis([0,8,1,10000])
#     ax.legend()
#
#     return ax
#
# anim = FuncAnimation(fig, update, frames=np.arange(8))
# anim.save('nesterov.gif', dpi=200, writer='imagemagick', fps=2)
#
# plt.show()


# %% Animation of Weights

# from matplotlib.animation import FuncAnimation
#
# opt = 'adam'
#
# print(list(c.theta.keys()))
# fig, ax = plt.subplots()
# fig.set_tight_layout(True)
#
# scale = np.max(np.abs(c.theta[opt][0]['w']))
# x = ax.imshow(c.theta[opt][0]['w'], vmin=-scale, vmax=scale)
# x.set_cmap('bwr')
# colorbar = fig.colorbar(x)
#
# # Query the figure's on-screen size and DPI. Note that when saving the figure to
# # a file, we need to provide a DPI for that separately.
# print('fig size: {0} DPI, size in inches {1}'.format(
#     fig.get_dpi(), fig.get_size_inches()))
#
# # Plot a scatter that persists (isn't redrawn) and the initial line.
#
# Path('nesterov.gif').unlink(missing_ok=True)
# def update(i):
#     global colorbar
#     colorbar.remove()
#     label = 'timestep {0}'.format(i)
#     print(label)
#     # Update the line and the axes (with a new xlabel). Return a tuple of
#     # "artists" that have to be redrawn for this frame.
#     scale = np.max(np.abs(c.theta[opt][i]['w']))
#     x = ax.imshow(c.theta[opt][i]['w'], vmin=-scale, vmax=scale)
#     x.set_cmap('bwr')
#     ax.grid(False)
#     colorbar = fig.colorbar(x)
#     return ax
#
# anim = FuncAnimation(fig, update, frames=np.arange(0, len(c.theta[opt]), 1))
# anim.save('nesterov.gif', dpi=150, writer='imagemagick')
# plt.show()
