#%%
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from GLM.utils import *
from data_gen_network import DataGenerator
from sampling_base import vary, gen_hamming_wrapper, run_params

sns.set()


def gen_sparse_params_θ(params_θ, N, p=0.05):
    if int(p * N) + 1 != p * N + 1:
        raise ValueError('Number of connections per neuron not integer.')

    p = params_θ.copy()
    p['connectedness'] = int(p * N) + 1
    return p

# def run_N(N_sp, λ_sp):
#     out = dict()
#     for N in reversed(N_sp):
#         p = params_base.copy()
#         p['N_lim'] = p['N'] = N
#         gen = DataGenerator(params=p, params_θ=gen_sparse_params_θ(p['N']))
#         out[N] = vary('λ1', λ_sp, p, gen.theta, rep=4, rpf=10, iters=600)
#
#         with open(f'N{p["N"]}λ.pk', 'wb') as f:
#             pickle.dump(out[N], f)
#     return out


if __name__ == '__main__':
    params_base = {  # For both data generation and fitting.
        'N': 40,
        'M': 5000,
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

    λ_sp = np.linspace(1, 4, 15)

    def run_and_plot(name, sp, params_base, params_θ):
        θ_fitted, θ_gnd = run_params(name, sp, 'λ1', λ_sp, params_base, params_θ, sample_theta_gnd=True, rep=5)
        gen_hamming_wrapper(θ_fitted, θ_gnd, sp, xlabel='λ', save_name=f'λ_vary{name}.png',
                            title=f"L1 Regularization. Vary {name}.")

    #%% Vary M
    M_sp = [1000, 2000, 5000, 10000, 20000]
    run_and_plot('M', M_sp, params_base, params_θ)


    # # %% Vary N
    # N_sp = [40, 80, 160]
    # params = params_base.copy()
    # params['M_lim'] = params['M'] = 10000
    # # TODO
    # run_and_plot('N', N_sp, params, params_θ)


    #%% Vary sparsity
    params = params_base.copy()
    params['M_lim'] = params['M'] = 20000
    params['N_lim'] = params['N'] = 160
    connectedness_sp = list(range(3, 13, 3))
    run_and_plot('connectedness', connectedness_sp, params, params_θ)


    #%% Vary randomness
    params = params_base.copy()
    params['M_lim'] = params['M'] = 10000
    params['N_lim'] = params['N'] = 80
    r_sp = [0.01, 0.05, 0.1, 0.2]
    run_and_plot('p_rand', r_sp, params, params_θ)


    #%% Vary max weight
    w_sp = [0.01, 0.05, 0.07, 0.1]
    run_and_plot('max_w', w_sp, params_base, params_θ)


    #%% Vary base
    b_sp = [-0.2, 0, 0.2]
    run_and_plot('base', b_sp, params_base, params_θ)