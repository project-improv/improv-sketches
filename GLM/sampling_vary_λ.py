#%%
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from GLM.utils import *
from data_gen_network import DataGenerator
from sampling_base import vary, gen_sparse_params_θ, gen_hamming_plot

sns.set()

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

def run_sparsity(c_sp, λ_sp):
    out = dict()
    for c in reversed(c_sp):
        print(f'{c=}')
        p = params_θ.copy()
        p['connectedness'] = c
        gen = DataGenerator(params=params_base, params_θ=p)
        out[c] = vary('λ1', λ_sp, params_base, gen.theta, rep=4, rpf=10, iters=500)

        with open(f'connectedness{c}λ.pk', 'wb') as f:
            pickle.dump(out[c], f)
    return out

def run_params_θ(p1, sp1, p2, sp2, params_θ, rep=4, varyθ=False):
    out = dict()
    th = dict()
    for u in reversed(sp1):
        print(f'{u=}')
        p = params_θ.copy()
        p[p1] = u
        if varyθ:
            p['seed'] = int(100 * abs(u))
        gen = DataGenerator(params=params_base, params_θ=p)
        out[u] = vary(p2, sp2, params_base, gen.theta, rep=rep, rpf=10, iters=500)
        th[u] = gen.theta
        with open(f'{p1}{u}{p2}.pk', 'wb') as f:
            pickle.dump((out[u], th), f)
    return out, th

def run_params(p1, sp1, p2, sp2, params, params_θ, sample_theta_gnd=False, save=True, rep=8, **vary_kwargs):
    """ p2 must be in params. """

    θ_fitted_dict = dict()
    θ_gnd_dict = dict()

    if p1 in params:
        vary_p = True
    elif p1 in params_θ:
        vary_p = False
    else:
        raise ValueError('Unknown parameter to vary!')

    for u in reversed(sp1):  # Start with largest to detect any failure.
        print(f'{u=}')
        p = params.copy()
        pθ = params_θ.copy()

        if vary_p:
            p[p1] = u
        else:
            pθ[p1] = u

        gen = DataGenerator(params=p, params_θ=pθ)  # Generate theta
        if sample_theta_gnd:
            θ_gnd = [gen.gen_new_theta() for _ in range(rep)]
        else:
            θ_gnd = gen.theta

        θ_fitted_dict[u] = vary(p2, sp2, p, θ_gnd, rep=rep, **vary_kwargs)
        θ_gnd_dict[u] = θ_gnd

        if save:
            with open(f'{p1}{u}{p2}.pk', 'wb') as f:
                pickle.dump((θ_fitted_dict[u], θ_gnd_dict), f)

    return θ_fitted_dict, θ_gnd_dict

def gen_hamming_wrapper(*args, title=None, save_name=None, **kwargs):
    gen_hamming_plot(*args, **kwargs)
    plt.suptitle(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_name)
    plt.show()




if __name__ == '__main__':
    # # Vary M
    # M_sp = [1000, 2000, 5000, 10000, 20000]
    # λ_sp = np.linspace(1, 3, 20)
    # out = run_M(M_sp, λ_sp)
    # gen_hamming_plot(out, M_sp, xlabel='λ')
    # plt.suptitle('L1 Regularization vs FP/FN. Vary M. N=40. 2 connections/neuron.')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('reg_M_fpfn.png')
    # plt.show()
    #
    # # %% Vary N
    # N_sp = [40, 80, 160]
    # λ_sp = np.linspace(1, 4, 20)
    # out = run_N(N_sp, λ_sp)
    # gen_hamming_plot(out, N_sp, varyN=True, xlabel='λ')
    # plt.suptitle('L1 Regularization vs FP/FN. M=20000. Vary N. 2 connections/neuron.')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('reg_N_fpfn.png')
    # plt.show()

    #%% Vary sparsity
    #
    # params_base = {  # For both data generation and fitting.
    #     'N': 160,
    #     'M': 20000,
    #     'dh': 2,
    #     'dt': 1,
    #     'ds': 1,
    #     'λ1': 1.,
    #     'λ2': 0.0
    # }
    #
    # params_θ = {
    #     'seed': 0,
    #     'p_inh': 0.6,
    #     'p_rand': 0.,
    #     'base': 0.,
    #     'connectedness': 9,
    #     'rand_w': False,
    #     'max_w': 0.05
    # }
    #
    # connectedness_sp = range(3, 13, 3)
    # λ_sp = np.linspace(1, 5, 20)
    # out = run_sparsity(connectedness_sp, λ_sp)
    #
    # def change_connectedness(params, c):
    #     p = params.copy()
    #     p['connectedness'] = c
    #     return p
    # gens = {c: DataGenerator(params_base, change_connectedness(params_θ, c)) for c in range(3, 13, 3)}
    #
    # gen_hamming_plot(out, range(3, 13, 3), params_base, gen=gens, xlabel='λ')
    # plt.suptitle('L1 Regularization vs Sparsity. N=160. M=20000. Vary connections/neuron.')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('reg_sparsity_fpfn.png')
    # plt.show()

    #%% Vary randomness
    params_base = {  # For both data generation and fitting.
        'N': 80,
        'M': 10000,
        'dh': 2,
        'dt': 1,
        'ds': 1,
        'λ1': 1.,
        'λ2': 0.0
    }

    params_θ = {
        'seed': 0,
        'p_inh': 0.6,
        'p_rand': 0.,
        'base': 0.,
        'connectedness': 5,
        'rand_w': False,
        'max_w': 0.05
    }
    # }
    #
    # r_sp = [0.01, 0.05, 0.1, 0.2]
    # λ_sp = np.linspace(1, 4, 15)
    # out, th = run_params_θ('p_rand', r_sp, 'λ1', λ_sp, varyθ=True)
    #
    # gen_hamming_plot(out, r_sp, params_base, th=th, xlabel='λ')
    #
    #
    # plt.suptitle('L1 Regularization vs %random connections. M=10000. 5 connections/neuron.')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('prand_λ_fpfn.png')
    # plt.show()
#%%
    # r_sp = [0.01, 0.05, 0.07, 0.1]
    # λ_sp = np.linspace(1, 4, 15)
    # out, th = run_params_θ('max_w', r_sp, 'λ1', λ_sp, varyθ=True)
    #
    # gen_hamming_plot(out, r_sp, params_base, th=th, xlabel='λ')
    #
    # plt.suptitle('L1 Regularization vs weight. M=10000. N=80. 5 connections/neuron.')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('maxw_λ_fpfn.png')
    # plt.show()

# #%%
#     params_θ = {
#         'seed': 0,
#         'p_inh': 0.6,
#         'p_rand': 0.,
#         'base': 0.,
#         'connectedness': 5,
#         'rand_w': True,
#         'max_w': 0.05
#     }
#
#     r_sp = [0.05, 0.07, 0.1, 0.15]
#     λ_sp = np.linspace(1, 4, 15)
#     out, th = run_params_θ('max_w', r_sp, 'λ1', λ_sp, params_θ, varyθ=True, rep=6)
#
#     gen_hamming_plot(out, r_sp, params_base, th=th, xlabel='λ')
#
#     plt.suptitle('L1 Regularization vs random weight. Vary max weight. M=10000. N=80. 5 connections/neuron.')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig('maxw_rand_λ_fpfn.png')
#     plt.show()

#%%

    params_base = {  # For both data generation and fitting.
        'N': 40,
        'M': 5000,
        'dh': 2,
        'dt': 1,
        'ds': 1,
        'λ1': 2.4,
        'λ2': 0.0
    }

    params_θ = {
        'seed': 0,
        'p_inh': 0.6,
        'p_rand': 0.,
        'base': 0.,
        'connectedness': 5,
        'rand_w': False,
        'max_w': 0.05
    }

    b_sp = [-0.2, 0., 0.2]
    λ_sp = np.linspace(1, 4, 15)
    out, th = run_params('base', b_sp, 'λ1', λ_sp, params_base, params_θ, sample_theta_gnd=True, rep=5)

    gen_hamming_plot(out, th, b_sp, xlabel='λ')

    plt.suptitle('L1 Regularization vs θb. Vary θb. M=5000. N=40. 5 connections/neuron.')
    plt.legend()
    plt.tight_layout()
    plt.savefig('θb_rand_λ_fpfn.png')
    plt.show()