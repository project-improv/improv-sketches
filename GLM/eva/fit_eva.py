import pickle

from GLM.utils import *
from GLM.model.compare_opt import CompareOpt

"""
Run processed data from improv data sharing @ Box.

Plot data from `batch` or `onacid` using the variable `name`.
Make sure to run scripts in `process_caiman` first.

40,000 iterations using the Nesterov momentum optimizer.

"""

for name in ['batch', 'onacid']:
    with open(f'GLM/eva/{name}_S.pk', 'rb') as f:
        y = pickle.load(f)

    s_raw = np.loadtxt('GLM/eva/stim_data.txt')

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
            'λ1': 1.5,
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
                save_grad=None, iters_offline=8000, indicator=indicator, hamming_thr=0.1, rpf=5, verbose=1000)

    w = np.asarray(c.theta['nesterov_offline'][-1]['w'])
    k = np.asarray(c.theta['nesterov_offline'][-1]['k'])
    h = np.asarray(c.theta['nesterov_offline'][-1]['h'])
    b = np.asarray(c.theta['nesterov_offline'][-1]['b'])

    with open(f'GLM/eva/theta_real_{name}.pk', 'wb') as f:
        pickle.dump({k: np.asarray(v, dtype=np.float64) for k, v in c.theta['nesterov_offline'][-1].items()}, f)

    with open('params_real.pk', 'wb') as f:
        pickle.dump(params, f)
