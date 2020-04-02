import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from GLM.utils import *
from data_gen_network import DataGenerator
from sampling_base import vary, gen_sparse_params_θ, gen_plot

sns.set()

params_base = {  # For both data generation and fitting.
    'N': 40,
    'M': 20000,
    'dh': 2,
    'dt': 1,
    'ds': 1,
    'λ1': 1.,
    'λ2': 0.0
}

params_base['M_lim'] = params_base['M']
params_base['N_lim'] = params_base['N']

params_θ = {
    'seed': 0,
    'p_inh': 0.6,
    'p_rand': 0.,
    'base': 0.,
    'connectedness': 3,
    'rand_w': False,
    'max_w': 0.05
}


def run_M(M_sp, λ_sp):
    out = dict()
    for M in reversed(M_sp):
        p = params_base.copy()
        p['M_lim'] = p['M'] = M
        gen = DataGenerator(params=p, params_θ=params_θ)
        out[M] = vary('λ1', λ_sp, p, gen.theta, rep=4)

        with open(f'M{p["M"]}λ.pk', 'wb') as f:
            pickle.dump(out[M], f)
    return out


def run_N(N_sp, λ_sp):
    out = dict()
    for N in reversed(N_sp):
        p = params_base.copy()
        p['N_lim'] = p['N'] = N
        gen = DataGenerator(params=p, params_θ=gen_sparse_params_θ(p['N']))
        out[N] = vary('λ1', λ_sp, p, gen.theta, rep=4, rpf=10, iters=600)

        with open(f'N{p["N"]}λ.pk', 'wb') as f:
            pickle.dump(out[N], f)
    return out


if __name__ == '__main__':
    # %% Vary M
    M_sp = [1000, 2000, 5000, 10000, 20000]
    λ_sp = np.linspace(1, 3, 20)
    out = run_M(M_sp, λ_sp)
    gen_plot(out, M_sp, xlabel='λ')
    plt.suptitle('L1 Regularization vs FP/FN. Vary M. N=40. 2 connections/neuron.')
    plt.legend()
    plt.tight_layout()
    plt.savefig('reg_M_fpfn.png')
    plt.show()

    # %% Vary N
    N_sp = [40, 80, 160]
    λ_sp = np.linspace(1, 4, 20)
    out = run_N(N_sp, λ_sp)
    gen_plot(out, N_sp, varyN=True, xlabel='λ')
    plt.suptitle('L1 Regularization vs FP/FN. M=20000. Vary N. 2 connections/neuron.')
    plt.legend()
    plt.tight_layout()
    plt.savefig('reg_N_fpfn.png')
    plt.show()
