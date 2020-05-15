from GLM.utils import *
from GLM.synthetic.sampling_base import gen_hamming_figure, run_params


if __name__ == '__main__':
    params_base = {  # For both data generation and fitting.
        'N': 40,
        'M': 5000,
        'dh': 2,
        'dt': 1,
        'ds': 8,
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
        θ_fitted, θ_gnd = run_params(name, sp, 'λ1', λ_sp, params_base, params_θ, rep=5)
        gen_hamming_figure(θ_fitted, θ_gnd, sp, xlabel='λ', save_name=f'λ_vary{name}.png',
                           title=f"L1 Regularization. Vary {name}.")

    #%% Vary M
    # run_and_plot('M', [1000, 2000, 5000, 10000, 20000], params_base, params_θ)
    #
    #
    # # %% Vary N
    params = params_base.copy()
    params['M_lim'] = params['M'] = 10000
    run_and_plot('N', [40, 80, 160], params, params_θ)
    #
    #
    # #%% Vary sparsity
    # params = params_base.copy()
    # params['M_lim'] = params['M'] = 20000
    # params['N_lim'] = params['N'] = 160
    # run_and_plot('connectedness', list(range(3, 13, 3)), params, params_θ)
    #
    #
    # #%% Vary randomness
    # params = params_base.copy()
    # params['M_lim'] = params['M'] = 10000
    # params['N_lim'] = params['N'] = 80
    # run_and_plot('p_rand', [0.01, 0.05, 0.1, 0.2], params, params_θ)
    #
    #
    # #%% Vary max weight
    # w_sp = [0.01, 0.05, 0.07, 0.1]
    # run_and_plot('max_w', w_sp, params_base, params_θ)
    #
    #
    # #%% Vary base
    # b_sp = [-0.2, 0, 0.2]
    # run_and_plot('base', b_sp, params_base, params_θ)
