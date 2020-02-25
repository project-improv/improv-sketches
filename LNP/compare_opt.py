import pickle
import time
from pathlib import Path

import numpy as np
import skopt

from glm_jax import GLMJax
from sklearn.metrics import mean_squared_error

"""
Create a log-likelihood graph over time using different optimizers.

"""


class CompareOpt:
    def __init__(self, params, S, stim):
        self.S, self.stim = S, stim
        self.params = params

        self.grad = dict()
        self.theta = dict()

        self.curr_N = 0
        self.total_M = self.S.shape[1]
        self.N_idx = np.zeros(self.total_M, dtype=np.int)
        self.lls = dict()
        self.mses = dict()
        self.theta_names = ['b', 'h', 'w']

        for i in range(self.total_M):  # Number of neurons at each time step.
            self.N_idx[i] = np.argmin(np.cumsum(self.S[:, i][::-1])[::-1])

    def conv_ys(self, y, s):
        """ Assuming that M = 100. """
        y_all = np.zeros((y[-1].shape[0], len(y)), dtype=np.float32)

        first_hundred = y[98]
        y_all[:first_hundred.shape[0], :98 + 1] = y[98]

        for t in range(99, len(y)):
            y_curr = y[t][:, -1]
            n_neu = y_curr.shape[0]
            y_all[:n_neu, t] = y_curr

        s_all = np.zeros((s[0].shape[0], len(y)))

        first_hundred = s[98]
        s_all[:first_hundred.shape[0], :98 + 1] = s[98]

        for t in range(99, len(y)):
            s_curr = s[t][:, -1]
            n_s = s_curr.shape[0]
            y_all[:n_s, t] = s_curr

        return y_all, s_all

    def run(self, optimizers, theta=None, save_grad=False, save_theta=False, use_gpu=False, gnd=None):
        """
        optimizers: a list of dict.
        save_grad: save the gradient of every step to self.grad.
        save_theta: save θ of every step to self.theta.
        """
        rpf = 1

        for opt in optimizers:
            ll = np.zeros((rpf * self.total_M))
            mses = np.zeros((rpf * self.total_M, len(self.theta_names)))
            offline = opt.get('offline', False)
            if 'offline' in opt:
                del opt['offline']

            self.grad[opt['name']] = list()
            self.theta[opt['name']] = list()

            self.params['M_lim'] = self.total_M if offline else 100

            model = GLMJax(self.params, optimizer=opt, use_gpu=use_gpu, θ=theta, zeros=False)
            t0 = time.time()

            for t in range(1, self.total_M):
                for rep in range(rpf):
                    # y = self.y[t] if not offline else self.y_all
                    # s = self.s[t] if not offline else self.s_all
                    if offline:
                        y, s = self.S, self.stim[:, :self.S.shape[1]]
                    else:
                        y, s = self.generate_step(t)

                    if save_grad:
                        self.grad[opt['name']].append(model.get_grad(y, s).copy())

                    if save_theta:
                        self.theta[opt['name']].append(model.θ.copy())

                    ll[t * rpf + rep] = model.fit(y, s)

                    if t > 50 and ll[t * rpf + rep] > 1e5:
                        raise Exception(f'Blew up at {t}.')

                    if gnd:
                        for j, name in enumerate(self.theta_names):
                            mses[t, j] = mean_squared_error(model.θ[name], gnd[name])

                if t % 1000 == 0:
                    print(f"{opt['name']}, step: {t}, b_norm: {mses[t, 0]}, ll:{ll[t * rpf]}")

            if offline:
                opt['offline'] = True
                self.lls[f"{opt['name']}_offline"] = ll
                self.mses[f"{opt['name']}_offline"] = mses
            else:
                self.lls[opt['name']] = ll
                self.mses[opt['name']] = mses

            print(f"{opt['name']}: {(time.time() - t0) / self.total_M:02f} s/step")

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
            y_step = self.S[:self.curr_N, :i]
            stim_step = self.stim[:, :i]
        else:
            y_step = self.S[:self.curr_N, i-m:i]
            stim_step = self.stim[:, i-m:i]

        return y_step, stim_step


if __name__ == '__main__':
    params = {'dh': 2, 'ds': 8, 'dt': 0.1, 'n': 0, 'N_lim': 10, 'M_lim': 100}
    optimizers = [
        {'name': 'sgd', 'step_size': 2e-5},
        # {'name': 'sgd', 'step_size':1e-5, 'offline': True},
    ]

    # S, stim = pickle.loads(Path('../twot.pk').read_bytes())
    # lls = c.run(optimizers)  #, save_grad=True, save_theta=True)
    gnd = pickle.loads(Path('theta_dict.pickle').read_bytes())


    N = params['N_lim']
    dh = params['dh']

    np.random.seed(0)
    theta_flat = np.random.random(np.sum([x.size for x in gnd.values()]))

    θ = {
        'w': theta_flat[:N * N].reshape((N, N)),
        'h': theta_flat[N * N:N * (N + dh)].reshape((N, dh)),
        'b': theta_flat[-N:],
        'k': np.zeros((params['N_lim'], params['ds']))
    }

    S = np.loadtxt('data_sample.txt').astype(np.float32)
    stim = np.zeros((params['ds'], S.shape[1]), dtype=np.float32)


    c = CompareOpt(params, S, stim)
    lls = c.run(optimizers, theta=θ, gnd=gnd, use_gpu=True)
#%%
    import matplotlib.pyplot as plt
    plt.plot(c.lls['sgd'])
    plt.show()

    plt.plot(c.mses['b'])

    # space = [
    #     skopt.space.Real(1e-6, 1e-4, name='step_size', prior='uniform'),
    # ]
    # x = c.hyper_opt('sgd', space, n_calls=15)
