import matplotlib.pyplot as plt
import seaborn as sns
import skopt
import time
from numpy import ndarray
from sklearn.metrics import mean_squared_error

from data_gen_network import DataGenerator
from jax.numpy import DeviceArray
from glm_jax import GLMJaxSynthetic
from GLM.utils import *
from typing import Dict, List

"""
Create a log-likelihood graph over time using different optimizers.

"""

try:  # kernprof
    profile
except NameError:
    def profile(x):
        return x

sns.set()


class CompareOpt:
    """
    Wrapper class of GLMJaxSynthetic for comparing optimizers and collecting fitting progress.
    """
    def __init__(self, params, y, s, checkpoint=100):
        self.y, self.s = y, s
        self.params = params

        self.models: Dict[str: GLMJaxSynthetic] = dict()
        self.optimizers: Dict[str: str] = dict()

        # Snapshots of gradient and θ at each checkpoint.
        self.grad: Dict[str: List[DeviceArray]] = dict()
        self.theta: Dict[str: List[Dict[DeviceArray]]] = dict()

        # Snapshots of metrics.
        self.ll: Dict[str: ndarray] = dict()
        self.mse: Dict[str: ndarray] = dict()
        self.hamming: Dict[str: ndarray] = dict()

        self.curr_N = 0
        self.total_M = self.y.shape[1]

        self.names_θ = ['b', 'h', 'w']
        self.checkpoint = checkpoint

        self.N_idx = np.zeros(self.total_M, dtype=np.int)
        for i in range(self.total_M):  # Number of neurons at each time step.
            self.N_idx[i] = np.argmin(np.cumsum(self.y[:, i][::-1])[::-1])

    @profile
    def run(self, optimizers, theta=None, resume=False, save_grad=None, save_theta=None, use_gpu=False, gnd_data=None,
            iters_offline=None, hamming_thr=0.2, indicator=None, rpf=1, verbose=0):
        """
        optimizers: a list of dict.
        theta: dict of weights.
        resume: continue training.
        save_grad: save gradient at every 1000 steps to self.grad.
        save_theta: save θ at every 1000 steps to self.theta.
        use_gpu: use GPU
        gnd_data: (y, s) ground truth
        iters_offline: if offline, number of repeats
        hamming_thr: threshold for hamming distance (proportion of max abs).
        """

        if gnd_data:  # Ground truth exists. Pre-calculate hamming threshold.
            gnd_for_hamming = np.abs(gnd_data['w']) > hamming_thr * np.max(np.abs(gnd_data['w'])).astype(np.int)

        for opt in optimizers:
            offline = opt.get('offline', False)

            iters = self.total_M if not offline else iters_offline
            name = opt['name'] if not offline else f"{opt['name']}_offline"
            n_checkpoints = int(iters / self.checkpoint)

            if resume and name in self.models:  # Continue training.
                model: GLMJaxSynthetic = self.models[name]
                self.ll[name] = ll = np.hstack([self.ll[name], np.zeros(n_checkpoints)])
                self.mse[name] = mse = np.vstack([self.mse[name], np.zeros((n_checkpoints, len(self.names_θ)))])
                self.hamming[name] = hamming = np.vstack([self.hamming[name], np.zeros((n_checkpoints, 2))])

            else:
                opt_ = {k: v for k, v in opt.items() if k != 'offline'}
                self.models[name] = model = GLMJaxSynthetic(self.params, optimizer=opt_, use_gpu=use_gpu, theta=theta,
                                                            data=(self.y, self.s), offline=offline, rpf=rpf)
                self.grad[name] = list()
                self.theta[name] = list()
                self.ll[name] = ll = np.zeros(n_checkpoints)
                self.mse[name] = mse = np.zeros((n_checkpoints, len(self.names_θ)))
                self.hamming[name] = hamming = np.zeros((n_checkpoints, 2))

            if verbose:
                print('Offline:', offline)
                print('Total iterations: ', iters)

            for i in range(iters):
                if i == 1:  # Avoid JIT
                    t0 = time.time()

                if save_grad and (i == 1 or i % save_grad == 0 or i == iters - 1):
                    self.grad[name].append(model.get_grad(y, s))

                if save_theta and (i == 1 or i % save_theta == 0 or i == iters - 1):
                    self.theta[name].append(model.θ)

                if i % self.checkpoint == 0 or i == iters - 1:
                    idx = int(model.iter / self.checkpoint)

                    if gnd_data:
                        for j, name_θ in enumerate(self.names_θ):
                            mse[idx, j] = mean_squared_error(model.θ[name_θ], gnd_data[name_θ])
                        ham = calc_hamming(gnd_data['w'], model.θ['w'], thr=hamming_thr)
                        hamming[idx, 0] = np.sum(ham == 1)  # FP
                        hamming[idx, 1] = np.sum(ham == -1)  # FN

                    ll[idx] = model.fit(return_ll=True, indicator=indicator)

                    if i > self.checkpoint and ll[idx] > 1e5:
                        raise Exception(f'Blew up at {i}.')

                    if verbose and (i % verbose == 0 or i == iters - 1):
                        print(f"{opt['name']}, step: {i:5.0f}, w_norm: {mse[idx, 2]:8.5e}, hamming FP/FN: {hamming[idx, 0], hamming[idx, 1]}, |θw|: {np.sum(np.abs(model.θ['w'])):8.5e}, ll:{ll[idx]:8.5f}")

                else:
                    model.fit(return_ll=False, indicator=indicator)  # Don't calculate LL. ~2x faster.

            if verbose:
                print(f"{opt['name']}: {1e3 * (time.time() - t0) / iters:.03f} ms/step, rpf={rpf}")

            self.optimizers = optimizers.copy()

        return self.ll

    def hyper_opt(self, name, space, n_calls=30, seed=0):
        func_run = self.run

        @skopt.utils.use_named_args(space)
        def _opt_func(**hyperp):
            opt = {'name': name, **hyperp}
            lls = func_run([opt])
            if not np.isnan(lls[-1]) and np.isfinite(lls[-1]) and np.max(lls[100:]) <= 5:
                return lls[-1, 0]
            else:
                return 1e5

        return skopt.gp_minimize(_opt_func, space, n_calls=n_calls, random_state=seed, noise=1e-10)


    def plot_opts(self, omit=3):
        fig, ax = plt.subplots(dpi=300)
        for i, (name, ll) in enumerate(self.ll.items()):
            x = np.arange(len(ll))[omit:] * self.checkpoint
            ax.plot(x, ll[omit:], label=f"{name}_stepsize: {self.optimizers[i]['step_size']}")
        ax.set_xlabel('Iteration')
        ax.set_ylabel('-log likelihood')
        ax.set_title('')
        plt.legend()
        plt.savefig('compare_opts.png')
        plt.show()


if __name__ == '__main__':
    params = {  # For both data generation and fitting.
        'N': 50,
        'M': 20000,
        'dh': 4,
        'dt': 1,
        'ds': 1,
        'λ1': 1.,
        'λ2': 0.05
    }

    params['M_lim'] = params['M']
    params['N_lim'] = params['N']

    params_θ = {
        'seed': 0,
        'p_inh': 0.5,
        'p_rand': 0.,
        'base': 0.,
        'connectedness': 4,
        'rand_w': False
    }

    opts = [
        {'name': 'sgd', 'step_size': 10, 'offline': True},
        {'name': 'nesterov', 'step_size': 10, 'offline': True, 'mass': 0.9},
        {'name': 'adam', 'step_size': 1e-3, 'offline': True},
        {'name': 'adagrad', 'step_size': 1, 'offline': True},
        {'name': 'rmsprop', 'step_size': 1e-3, 'offline': True},
        {'name': 'rmsprop_momentum', 'step_size': 1e-3, 'offline': True},
    ]

    gen = DataGenerator(params=params, params_θ=params_θ)

    s = np.zeros((params['ds'], params['M']), dtype=np.float32)
    r, y = gen.gen_spikes(params=params, seed=0)
    c = CompareOpt(params, y, s)
    c.run(opts, theta=gen_rand_theta(params), gnd_data=gen.theta, use_gpu=True, save_theta=10000,
          iters_offline=10000, hamming_thr=0.1, verbose=True)

    c.plot_opts()
