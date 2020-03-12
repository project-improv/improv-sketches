import pickle

import numpy as np
import skopt
import time
from pathlib import Path
from sklearn.metrics import mean_absolute_error

from glm_jax import GLMJax

"""
Create a log-likelihood graph over time using different optimizers.

"""

try:  # kernprof
    profile
except NameError:
    def profile(x): return x


class CompareOpt:
    def __init__(self, params, S, stim):
        self.S, self.stim = S, stim
        self.params = params

        self.grad = dict()
        self.theta = dict()
        self.model = dict()

        self.curr_N = 0
        self.total_M = self.S.shape[1]
        self.N_idx = np.zeros(self.total_M, dtype=np.int)
        self.lls = dict()
        self.maes = dict()
        self.hamming = dict()
        self.theta_names = ['b', 'h', 'w']

        for i in range(self.total_M):  # Number of neurons at each time step.
            self.N_idx[i] = np.argmin(np.cumsum(self.S[:, i][::-1])[::-1])

    @profile
    def run(self, optimizers, theta=None, resume=False, save_grad=False, save_theta=False, use_gpu=False, gnd_data=None,
            iters_offline=None, checkpoint=100, hamming_thr=0.2):
        """
        optimizers: a list of dict.
        theta: dict of weights.
        resume: continue training.
        save_grad: save gradient at every 1000 steps to self.grad.
        save_theta: save θ at every 1000 steps to self.theta.
        use_gpu: use GPU
        gnd_data: (y, s) ground truth
        iters_offline: if offline, number of repeats
        checkpoint: save LL, MAE, hamming every _ steps
        hamming_thr: threshold for hamming distance (proportion of max abs).
        """

        for opt in optimizers:
            name = opt['name']
            offline = opt.get('offline', False)
            iters = self.total_M if not offline else iters_offline


            if resume and name in self.model:
                model = self.model[name]
                self.lls[name] = ll = np.hstack([self.lls[name], np.zeros(int(iters / checkpoint))])
                self.maes[name] = maes = np.vstack([self.mses[name], np.zeros((int(iters / checkpoint), len(self.theta_names)))])
                self.hamming[name] = hamming = np.vstack([self.hamming[name], np.zeros((int(iters / checkpoint), 2))])

            else:
                opt_ = {k: v for k, v in opt.items() if k != 'offline'}
                offline_data = (self.S, self.stim) if offline else None
                self.model[name] = model = GLMJax(self.params, optimizer=opt_, use_gpu=use_gpu, theta=theta, offline_data=offline_data)
                self.grad[name] = list()
                self.theta[name] = list()
                self.lls[name] = ll = np.zeros(int(iters / checkpoint))
                self.maes[name] = maes = np.zeros((int(iters / checkpoint), len(self.theta_names)))
                self.hamming[name] = hamming = np.zeros((int(iters / checkpoint), 2))

            if gnd_data:
                gnd_for_hamming = np.abs(gnd_data['w']) > hamming_thr * np.max(np.abs(gnd_data['w'])).astype(np.int)

            print('Total iterations: ', iters)

            for t in range(iters):
                if t == 1:  # Avoid JIT
                    t0 = time.time()

                y, s = self.generate_step(t) if not offline else (None, None)  # Offline case is handled in GLMJax.
                if save_grad and (t == 1 or t % 1000 == 0):
                    self.grad[opt['name']].append(model.get_grad(y, s).copy())

                if save_theta and (t == 1 or t % 1000 == 0):
                    self.theta[opt['name']].append(model.θ.copy())

                if t % checkpoint == 0:
                    idx = int(model.iter/checkpoint)

                    if gnd_data:
                        for j, name in enumerate(self.theta_names):
                            maes[idx, j] = mean_absolute_error(model.θ[name], gnd_data[name])
                        binarized = (np.abs(model.θ['w']) > hamming_thr * np.max(np.abs(model.θ['w']))).astype(np.int)
                        res = binarized - gnd_for_hamming
                        hamming[idx, 0] = np.sum(res == 1) # FP
                        hamming[idx, 1] = np.sum(res == -1)  # FN

                    ll[idx] = model.fit(y, s, return_ll=True)

                    if t > 50 and ll[idx] > 1e5:
                        raise Exception(f'Blew up at {t}.')

                    if t % int(iters / 20) == 0:
                        print(
                            f"{opt['name']}, step: {checkpoint*idx:5.0f}, w_norm: {maes[idx, 2]:8.5e}, hamming: {hamming[idx,0], hamming[idx,1]}, |θw|: {np.sum(np.abs(model.θ['w'])):8.5e}, ll:{ll[idx]:8.5f}")

                else:
                    model.fit(y, s, return_ll=False)

            print(f"{opt['name']}: {1e3 * (time.time() - t0) / iters:.03f} ms/step")

        return self.lls

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

    def generate_step(self, i):
        m = self.params['M_lim']

        self.curr_N = self.N_idx[i] if self.N_idx[i] > self.curr_N else self.curr_N

        if i < m:
            y_step = self.S[:self.curr_N + 1, :i + 1]
            stim_step = self.stim[:, :i + 1]
        else:
            y_step = self.S[:self.curr_N + 1, i - m:i]
            stim_step = self.stim[:, i - m:i]
        return y_step, stim_step


if __name__ == '__main__':

    def online_sch(M, step_size, decay_steps, decay_rate):
        def schedule(i):
            if i < M:
                return step_size * i / M
            else:
                return step_size  # * decay_rate ** (i / decay_steps)

        """ Learning rate schedule for JAX. Linear increase over frame number. """
        return schedule


    params = {'dh': 2, 'ds': 8, 'dt': 0.1, 'n': 0, 'N_lim': 100, 'M_lim': 100}
    optimizers = [
        # {'name': 'sgd', 'step_size': 1e-6},
        # {'name': 'nesterov', 'step_size': online_sch(100, 1e-3, 5000, 0.1), 'mass': 0.9},
        # {'name': 'adam', 'step_size': online_sch(100, 1e-4, 5000, 0.1)},#, 'b1': 0.9, 'b2': 0.999},
        {'name': 'sgd', 'step_size': 1e-4, 'offline': True},
    ]

    # S, stim = pickle.loads(Path('../twot.pk').read_bytes())
    # lls = c.run(optimizers)  #, save_grad=True, save_theta=True)
    gnd = pickle.loads(Path('theta_dict.pickle').read_bytes())

    N = params['N_lim']
    dh = params['dh']

    np.random.seed(0)
    theta_flat = np.random.random(np.sum([x.size for x in gnd.values()]))

    θ = {
        'w': 1 / N * theta_flat[:N * N].reshape((N, N)),
        'h': 1 / N * theta_flat[N * N:N * (N + dh)].reshape((N, dh)),
        'b': 1 / N * theta_flat[-N:],
        'k': np.zeros((params['N_lim'], params['ds']))
    }

    S = np.loadtxt('data_sample.txt').astype(np.float32)
    stim = np.zeros((params['ds'], S.shape[1]), dtype=np.float32)

    c = CompareOpt(params, S, stim)
    lls = c.run(optimizers, theta=θ, gnd_data=gnd, use_gpu=True)
    # %%
    import matplotlib.pyplot as plt

    plt.plot(c.lls[optimizers[0]['name']])
    plt.show()

    plt.plot(c.mses[optimizers[0]['name']][:, 2])
    plt.show()

    # space = [
    #     skopt.space.Real(1e-6, 1e-4, name='step_size', prior='uniform'),
    # ]
    # x = c.hyper_opt('sgd', space, n_calls=15)

