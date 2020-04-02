import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from GLM.utils import *
from compare_opt import CompareOpt
from data_gen_network import DataGenerator

"""
Generate multiple samples of data, fit model, and save θ.
"""

sns.set()


def vary(to_vary, space, params, gnd, rep=8, opt=None, rpf=1, iters=5000):
    """
    Fit the model with parameter `to_vary` being varied in `space` to obtain a sampling distribution of fitted θ.
    :return: A dict with {param_value: 3D-array} of θ_w.
    """

    if opt is None:
        opt = [{'name': 'nesterov', 'step_size': offline_decay(10, 1e3, 0.1), 'mass': 0.99, 'offline': True}]
    assert len(opt) == 1

    def sample(p):
        """ Sample data and fit model. """
        θ = {name: np.zeros((rep, *arr.shape)) for name, arr in gen_rand_theta(params).items()}
        s = np.zeros((p['ds'], p['M']), dtype=np.float32)  # Zero for now.
        # n_spikes = np.zeros((rep, p['N_lim']))
        gen = DataGenerator(params, theta=gnd)

        for i in range(rep):
            print(f'Repeat {i + 1}/{rep}.')
            r, y = gen.gen_spikes(params=p, seed=10 * i)
            if not np.isfinite(np.mean(r)):
                raise Exception('Generator blew up.')
            c = CompareOpt(p, y, s)
            c.run(opt, theta=gen_rand_theta(params), gnd_data=gnd, use_gpu=True, save_theta=500,
                  iters_offline=iters, hamming_thr=0.1, verbose=100, rpf=rpf)
            for name, arr in c.theta[f"{opt[0]['name']}_offline"][-1].items():
                θ[name][i, ...] = arr
            # n_spikes[i, :] = np.sum(y, axis=1)
        return θ

    out = dict()
    for i, s in enumerate(reversed(space)):
        print(f'{to_vary}={s}')
        p_ = params.copy()
        p_[to_vary] = s
        p_['M_lim'] = p_['M']
        p_['N_lim'] = p_['N']
        out[s] = sample(p_)
    return out


def plot_bias_var(θs, gnd, name='w'):
    sd = np.sqrt(np.var(θs[name], axis=0))
    mean = np.abs(np.mean(θs[name], axis=0))
    mae = np.mean(np.abs(θs[name] - gnd[name]), axis=0)
    mask = np.abs(np.mean(θs[name], axis=0)) > 0.1 * np.max(θs[name])

    fig, ax = plt.subplots(figsize=(10, 4), dpi=200, ncols=2)
    ai = ax[0].imshow(sd / 0.1, vmin=0, vmax=1.0)
    ax[0].grid(False)
    ax[0].set_title(f'θ_w Normalized SD, median={np.median((sd / mean)[mask]): .3f}, clipped to >0.1 max')
    fig.colorbar(ai, ax=ax[0])

    percent = mae / np.max(np.abs(gnd['w']))
    ai = ax[1].imshow(percent, vmin=0, vmax=1.)
    ax[1].grid(False)
    ax[1].set_title(f'θ_w Normalized Error (to max entry), mean={np.mean(mae): .3f}')
    fig.colorbar(ai, ax=ax[1])
    fig.tight_layout()

    plt.show()

def calc_median_cv(th):
    mask = np.zeros(th.shape[1:], dtype=np.bool)
    N = mask.shape[1]

    for i in range(N):
        if i == 0:
            mask[i, i + 1] = 1
        elif i == N - 1:
            mask[i, i - 1] = 1
        else:
            mask[i, i - 1] = 1
            mask[i, i + 1] = 1

    sd = np.sqrt(np.var(th, axis=0))
    mean = 0.05 #* np.abs(np.mean(th, axis=0))
    # mask = np.abs(np.mean(th, axis=0)) > 0.1 * np.max(th)
    return np.mean((sd / mean))

def gen_sparse_params_θ(N, p=0.05):
    return {
        'seed': 0,
        'p_inh': 0.6,
        'p_rand': 0.,
        'base': 0.,
        'connectedness': int(p * N) + 1,
        'rand_w': False,
        'max_w': 0.05
    }


def plot_hamming(ax, func, from_run, sp, varyN=False, norm=False):
    trim = 0
    for i, u in enumerate(sp):
        x = []
        y = []
        ci = []

        if varyN:
            p = params_base.copy()
            p['N'] = p['N_lim'] = u
            gen = DataGenerator(p, gen_sparse_params_θ(u))
        else:
            gen = DataGenerator(params_base, gen_sparse_params_θ(params_base['N']))

        for λ, θs in from_run[u].items():
            x.append(λ)
            results = [func((gen.theta['w'], θs['w'][i, ...])) for i in range(len(θs['w']))]
            y.append(np.mean(results))
            ci.append(1.96 * np.sqrt(np.var(results)))

        # for x, y in processed.items():
        x = np.array(x)[trim:]
        if norm:
            y = np.array(y)[trim:] / p['N']**2
            ci = np.array(ci)[trim:] / p['N']**2
        else:
            y = np.array(y)[trim:]
            ci = np.array(ci)[trim:]
        ax.plot(x, y, f'C{i}', label=u)
        idx_min = np.argmin(y)
        ax.plot(x[idx_min], y[idx_min], f'*C{i}')
        ax.fill_between(x, y - ci, y + ci, alpha=0.3)

def gen_plot(*args, xlabel=None, **kwargs):
    fig, axs = plt.subplots(ncols=3, figsize=(12, 4), dpi=300)
    axs.flatten()
    funcs = [
        lambda t: np.sum(calc_hamming(t[0], t[1]) == 1),
        lambda t: np.sum(calc_hamming(t[0], t[1]) == -1),
        lambda t: np.sum(np.abs(calc_hamming(t[0], t[1]))),
    ]

    [plot_hamming(axs[i], funcs[i], *args, **kwargs) for i in range(len(funcs))]

    for ax in axs:
        ax.set_xlabel(xlabel)

    axs[0].set_ylabel('FP')
    axs[1].set_ylabel('FN')
    axs[2].set_ylabel('FP+FN')
    return fig, axs


if __name__ == '__main__':
    """ Sample vary N and M. """

    params_base = {  # For both data generation and fitting.
        'N': 40,
        'M': 10000,
        'dh': 2,
        'dt': 1,
        'ds': 1,
        'λ1': 2.4,
        'λ2': 0.0
    }

    params_base['M_lim'] = params_base['M']
    params_base['N_lim'] = params_base['N']

    # gnd_p = np.sum(np.abs(gen.theta['w']) > 0)
    # gnd_n = params['N_lim'] ** 2 - gnd_p

    N_sp = [40, 80, 160]
    M_sp = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    #
    out = dict()
    for N in reversed(N_sp):
        p = params_base.copy()
        p['N_lim'] = p['N'] = N
        gen = DataGenerator(params=p, params_θ=gen_sparse_params_θ(N))
        out[N] = vary('M', M_sp, p, gen.theta, rep=5, rpf=10, iters=1000)

        with open(f'N{p["N"]}M.pk', 'wb') as f:
            pickle.dump(out[N], f)
#%%
    gen_plot(out, N_sp, varyN=True, xlabel='M', norm=True)
    plt.suptitle('FP/FN vs Data Length. Normalized to N^2. Vary N. λ=2.4, 5% sparsity.')
    plt.legend()
    plt.tight_layout()
    plt.savefig('length_N_fpfn.png')
    plt.show()

    #%% Data Length / MSE plot.
    fig, ax = plt.subplots(dpi=300)
    for i, N in enumerate(reversed(N_sp)):
        cv = np.zeros(len(M_sp))
        for j, M in enumerate(M_sp):
            cv[j] = calc_median_cv(out[N][M]['w'])
        ax.semilogx(M_sp, cv, f'C{i}', label=f'{N=}', alpha=0.7)

    ax.set_title(
        'Standardized SD θ_w (n=5)\n Watt-Strogatz, 2 connections/neuron, no randomness, 60% inhibitory, fixed θw=0.05')
    ax.set_ylabel('Coefficient of Variation')
    ax.set_xlabel('Data Length')
    fig.legend(loc='center right')
    plt.tight_layout()
    plt.savefig('SD.png')
    plt.show()
