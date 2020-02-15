import pickle
import time
from pathlib import Path

import numpy as np
import skopt

from glm_jax import GLMJax

"""
Create a log-likelihood graph over time using different optimizers.

"""


class CompareOpt:
    def __init__(self, params, file='tbif_batch_for_analysis.pk'):
        self.y, self.s = pickle.loads(Path(file).read_bytes())
        self.params = params
        self.y_all, self.s_all = self.conv_ys(self.y, self.s)

        self.grad = dict()
        self.theta = dict()

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

    def run(self, optimizers, save_grad=False, save_theta=False):
        """
        optimizers: a list of dict.
        save_grad: save the gradient of every step to self.grad.
        save_theta: save θ of every step to self.theta.
        """
        rpf = 1
        lls = np.zeros((rpf * len(self.y), len(optimizers)))

        for i, opt in enumerate(optimizers):
            offline = opt.get('offline', False)
            if 'offline' in opt:
                del opt['offline']

            self.grad[opt['name']] = list()
            self.theta[opt['name']] = list()

            self.params['M_lim'] = 3000 if offline else 100

            model = GLMJax(self.params, optimizer=opt)
            t0 = time.time()

            for t in range(len(self.y)):
                for rep in range(rpf):
                    y = self.y[t] if not offline else self.y_all
                    s = self.s[t] if not offline else self.s_all

                    if save_grad:
                        self.grad[opt['name']].append(model.get_grad(self.y[t], self.s[t]).copy())

                    if save_theta:
                        self.theta[opt['name']].append(model.θ.copy())

                    lls[t * rpf + rep, i] = model.fit(y, s)

                if t % 100 == 0:
                    print(f"{opt['name']}, step: {t}")

            if offline:
                opt['offline'] = True

            print(f"{opt['name']}: {(time.time() - t0) / len(self.y):02f} s/step")

        return lls

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


if __name__ == '__main__':
    params = {'dh': 10, 'ds': 8, 'dt': 0.5, 'n': 0, 'N_lim': 200, 'M_lim': 3000}
    optimizers = [{'name': 'sgd', 'step_size': 1e-5}]

    c = CompareOpt(params=params, file='tbif_batch_for_analysis.pk')
    # lls = c.run(optimizers, save_grad=True, save_theta=True)

    space = [
        skopt.space.Real(1e-6, 1e-4, name='step_size', prior='uniform'),
    ]
    x = c.hyper_opt('sgd', space, n_calls=15)
