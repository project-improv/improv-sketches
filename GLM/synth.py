import pickle

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from compare_opt import CompareOpt
from GLM.utils import *

sns.set()

"""
Run GLMJax using synthetic data from `data_gen.py`.
"""

params_raw = pickle.loads(Path('params_dict.pickle').read_bytes())
window_size = 2000
params = {
    'N_lim': params_raw['numNeurons'],
    'M_lim': window_size,
    'dh': params_raw['hist_dim'],
    'dt': params_raw['dt'],
    'ds': 1,
    'λ1': 0.4,
    'λ2': 0.02
}

print('Loading data.')
gnd = pickle.loads(Path('theta_dict.pickle').read_bytes())
gnd_p = np.sum(np.abs(gnd['w']) > 0)
gnd_n = params['N_lim']**2 - gnd_p
print('Number of non-zero values in gnd θ_w:', gnd_p)

y = np.loadtxt('data_sample.txt').astype(np.float32)[:, :window_size]
s = np.zeros((params['ds'], y.shape[1]), dtype=np.float32)


θ_init = gen_rand_theta(params)

optimizers = [
    #{'name': 'adam', 'step_size': online_decay(2e-3, 2e4, 0.1)},
    {'name': 'adam', 'step_size': offline_decay(5e-3, 1e4, 0.1), 'offline': True},
]

# indicator = np.ones(y.shape)
# indicator[:, :10000] = 0.
# y[:, :10000] = 0.

def regularization_path(r: np.ndarray):
    out = np.zeros((len(r), params['N_lim']**2))
    for i, v in enumerate(r):
        p = params.copy()
        p['λ1'] = v
        print(f"{p['λ1']}=")
        c = CompareOpt(p, y, s)
        c.run(optimizers, theta=gen_rand_theta(params), resume=True, gnd_data=gnd, use_gpu=True, save_theta=1000,
              save_grad=None, iters_offline=5000, indicator=indicator)
        out[i, :] = c.theta['adam_offline'][-1]['w'].reshape(p['N_lim']**2)

        plt.imshow(np.abs(c.theta['adam_offline'][-1]['w']))
        plt.show()

    return out

def plot_rp(lambdas, theta):
    from matplotlib import rc, rcParams
    rc('text', usetex=True)
    rcParams['font.family'] = 'serif'

    fig, ax = plt.subplots(dpi=300, figsize=(5, 4))
    for i, x in enumerate(gnd['w'].reshape(gnd['w'].size)):
        ax.scatter([1e-4], gnd['w'].reshape(gnd['w'].size)[i], )
    ax.semilogx(lambdas, theta)
    ax.set_title(f"$\\ell_1$ Regularization Path. $N={params['N_lim']}$, ${window_size}$ Steps", loc='left')
    ax.set_xlabel('$\\lambda$')

    plt.tight_layout()
    plt.savefig('regularization_path.png')

    plt.show()




# if __name__ == '__main__':
#     λs = np.logspace(-4, -1, 30)
#     th = regularization_path(λs)
#     plot_rp(λs, th)


if __name__ == '__main__':

    c = CompareOpt(params, y, s)
    lls = c.run(optimizers, theta=gen_theta(params), resume=True, gnd_data=gnd, use_gpu=True, save_theta=1000,
            save_grad=None, iters_offline=10000, indicator=None, hamming_thr=0.1, rpf=2)

    # def roc(λs):
    #     results = dict()
    #     for w in [1000, 5000, 10000, 50000, 100000]:
    #         print(w)
    #         window_size = w
    #         y_ = y[:, :w]
    #         s_ = s[:, :w]
    #         results[w] = np.zeros((len(λs), 2))
    #         for i, λ in enumerate(λs):
    #             params = {
    #                 'N_lim': params_raw['numNeurons'],
    #                 'M_lim': window_size,
    #                 'dh': params_raw['hist_dim'],
    #                 'dt': params_raw['dt'],
    #                 'ds': 1,
    #                 'λ1': λ,
    #                 'λ2': 0
    #             }
    #
    #             c = CompareOpt(params, y_, s_)
    #             c.run(optimizers, theta=gen_theta(params), resume=True, gnd_data=gnd, use_gpu=True, save_theta=1000,
    #                         save_grad=None, iters_offline=10000, indicator=None, hamming_thr=0.1, rpf=1)
    #             results[w][i, :] = c.hamming['adam_offline'][-1]
    #     return results
    #
    # results = roc(np.linspace(0.005, 1, 10))  # return FP/FN

    # Precision = True Positives / (True Positives + False Positives)
    # Recall = True Positives / (True Positives + False Negatives)


    # tp = gnd_p - np.array([r[:, 1] for r in results.values()])
    # fp = np.array([r[:, 0] for r in results.values()])
    #
    # precision = tp / (tp + fp)
    # recall = tp / (tp + np.array([r[:, 1] for r in results.values()]))
    #
    # fig, ax = plt.subplots(dpi=300)
    # for i, k in enumerate(results.keys()):
    #     ax.plot(recall[i,:], precision[i,:], label=f'{k} points')
    # ax.set_xlabel('Recall')
    # ax.set_ylabel('Precision')
    # fig.legend(loc=7)
    # ax.set_xlim(0,1)
    # ax.set_ylim(0,1)
    # ax.set_title('Offline, 50 neurons, 0.1 hamming threshold, 118 non-zero')
    # plt.savefig('precision_recall.png')
    # plt.show()


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


    opt = f"{optimizers[0]['name']}_offline" if optimizers[0]['offline'] else optimizers[0]['name']

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
