# -*- coding: utf-8 -*-

from functools import partial
from importlib import import_module
from typing import Dict, Tuple

import jax.numpy as np
import numpy as onp
from jax import devices, jit, random, grad, value_and_grad
from jax.config import config
from jax.experimental.optimizers import OptimizerState
from jax.interpreters.xla import DeviceArray

try:  # kernprof
    profile
except NameError:
    def profile(x):
        return x


class GLMJax:
    def __init__(self, p: Dict, theta=None, optimizer=None, use_gpu=False, rpf=1):
        """
        A JAX implementation of simGLM.

        Assuming that all parameters are fixed except N and M.
        - N can be increased forever (at a cost).
        - M can only be increased to M_lim.

        There is a substantial cost to increasing N. It is a better idea to keep N reasonably high at initialization.

        :param p: Dictionary of parameters. λ is L1 sparsity.
        :type p: dict
        :param theta: Dictionary of ndarray weights. Must conform to parameters in p.
        :type theta: dict
        :param optimizer: Dictionary of optimizer name and hyperparameters from jax.experimental.optimizers.
            Ex. {'name': 'sgd', 'step_size': 1e-4}
        :type optimizer: dict
        :param use_gpu: Use GPU
        :type use_gpu: bool
        :param rpf: "Round per frame". Number of times to repeat `fit` using the same data.
        :type rpf: int
        """

        self.use_gpu = use_gpu
        platform = 'gpu' if use_gpu else 'cpu'  # Restart interpreter when switching platform.
        config.update('jax_platform_name', platform)
        print('Using', devices()[0])

        if not self.use_gpu:  # JAX defaults to single precision.
            config.update("jax_enable_x64", True)

        # p check
        if not all(k in p for k in ['ds', 'dh', 'dt', 'N_lim', 'M_lim']):
            raise ValueError('Parameter incomplete!')
        self.params = p

        # θ check
        if theta is None:
            raise ValueError('θ not specified.')
        else:
            assert theta['w'].shape == (p['N_lim'], p['N_lim'])
            assert theta['h'].shape == (p['N_lim'], p['dh'])
            assert theta['k'].shape == (p['N_lim'], p['ds'])
            assert (theta['b'].shape == (p['N_lim'],)) or (theta['b'].shape == (p['N_lim'], 1))

            if len(theta['b'].shape) == 1:  # Array needs to be 2D.
                theta['b'] = np.reshape(theta['b'], (p['N_lim'], 1))

            self._θ = {k: np.asarray(v) for k, v in theta.items()}  # Transfer to device.

        # Optimizer
        if optimizer is None:
            raise ValueError('Optimizer not named.')
        opt_func = getattr(import_module('jax.experimental.optimizers'), optimizer['name'])
        optimizer = {k: v for k, v in optimizer.items() if k != 'name'}
        self.opt_init, self.opt_update, self.get_params = [jit(func) for func in opt_func(**optimizer)]
        self._θ: OptimizerState = self.opt_init(self._θ)

        self.rpf = rpf
        self.ones = onp.ones((self.params['N_lim'], self.params['M_lim']))

        self.current_N = 0
        self.current_M = 0
        self.iter = 0

    def ll(self, y, s, indicator=None):
        return self._ll(self.theta, self.params, *self._check_arrays(y, s, indicator))

    @profile
    def fit(self, y, s, return_ll=False, indicator=None):
        """
        Fit model. Returning log-likelihood is ~2 times slower.
        """
        if return_ll:
            self._θ, self.iter, ll = GLMJax._fit_ll(self._θ, self.params, self.opt_update, self.get_params,
                                                    self.iter, *self._check_arrays(y, s, indicator))
            self.iter += 1
            return ll
        else:
            self._θ, self.iter = GLMJax._fit(self._θ, self.params, self.rpf, self.opt_update, self.get_params,
                                             self.iter, *self._check_arrays(y, s, indicator))
            self.iter += 1

    @staticmethod
    @partial(jit, static_argnums=(1, 2, 3))
    def _fit_ll(θ: Dict, p: Dict, opt_update, get_params, iter, m, n, y, s, indicator):
        ll, Δ = value_and_grad(GLMJax._ll)(get_params(θ), p, m, n, y, s, indicator)
        θ = opt_update(iter, Δ, θ)
        return θ, ll

    @staticmethod
    @partial(jit, static_argnums=(1, 2, 3, 4))
    def _fit(θ: Dict, p: Dict, rpf, opt_update, get_params, iter, m, n, y, s, indicator):
        for i in range(rpf):
            Δ = grad(GLMJax._ll)(get_params(θ), p, m, n, y, s, indicator)
            θ = opt_update(iter, Δ, θ)
        return θ

    def grad(self, y, s, indicator=None) -> DeviceArray:
        return grad(self._ll)(self.theta, self.params, *self._check_arrays(y, s, indicator)).copy()

    def predict(self, y, s, indicator=None):
        y, s, indicator = self._check_arrays(y, s, indicator)[2:]
        linear = GLMJax._run_linear(self.theta, self.params, y, s)
        log_r̂ = linear[0] + linear[1] + linear[2] + linear[3] + linear[4]  # Broadcast.
        return np.exp(log_r̂) * indicator

    def linear_contributions(self, y, s, indicator=None):
        y, s, indicator = self._check_arrays(y, s, indicator)[2:]
        linear = GLMJax._run_linear(self.theta, self.params, y, s)
        if indicator is not None:
            return linear[0], *[u*indicator for u in linear[1:4]], linear[4]
        return linear

    @profile
    def _check_arrays(self, y, s, indicator=None) -> Tuple[onp.ndarray]:
        """
        Check validity of input arrays and pad y and s to (N_lim, M_lim) and (ds, M_lim), respectively.
        Indicator matrix discerns true zeros from padded ones.
        :return current_M, current_N, y, s, indicator
        """
        assert y.shape[1] == s.shape[1]
        assert s.shape[0] == self.params['ds']

        self.current_N, self.current_M = y.shape

        N_lim = self.params['N_lim']
        M_lim = self.params['M_lim']

        while y.shape[0] > N_lim:
            self._increase_θ_size()
            N_lim = self.params['N_lim']

        if indicator is None:
            indicator = onp.ones(y.shape)

        if y.shape != (N_lim, M_lim):
            y_ = onp.zeros((N_lim, M_lim), dtype=onp.float32)
            s_ = onp.zeros((self.params['ds'], M_lim), dtype=onp.float32)
            indicator_ = onp.zeros((N_lim, M_lim), dtype=onp.float32)

            y_[:y.shape[0], :y.shape[1]] = y
            s_[:s.shape[0], :s.shape[1]] = s

            if indicator is not None:
                indicator_[:y.shape[0], :y.shape[1]] = indicator
            else:
                indicator_[:y.shape[0], :y.shape[1]] = 1.
            y, s, indicator = y_, s_, indicator_

        if y.shape[1] > M_lim:
            raise ValueError('Data are too wide (M exceeds M_lim).')

        return self.current_M, self.current_N, y, s, indicator

    def _increase_θ_size(self) -> None:
        """
        Doubles θ capacity for N in response to increasing N.

        """
        N_lim = self.params['N_lim']
        print(f"Increasing neuron limit to {2 * N_lim}.")
        self._θ = self.theta

        self._θ['w'] = onp.concatenate((self._θ['w'], onp.zeros((N_lim, N_lim))), axis=1)
        self._θ['w'] = onp.concatenate((self._θ['w'], onp.zeros((N_lim, 2 * N_lim))), axis=0)

        self._θ['h'] = onp.concatenate((self._θ['h'], onp.zeros((N_lim, self.params['dh']))), axis=0)
        self._θ['b'] = onp.concatenate((self._θ['b'], onp.zeros((N_lim, 1))), axis=0)
        self._θ['k'] = onp.concatenate((self._θ['k'], onp.zeros((N_lim, self.params['ds']))), axis=0)

        self.params['N_lim'] = 2 * N_lim
        self._θ = self.opt_init(self._θ)

    # Compiled Functions
    # A compiled function needs to know the shape of its inputs.
    # Changing array size will trigger a recompilation.
    # See https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#%F0%9F%94%AA-Control-Flow

    @staticmethod
    @partial(jit, static_argnums=(1,))
    def _run_linear(θ: Dict, p: Dict, y, s) -> Tuple:
        """
        Return log rates from the model. That is, the linear part of the model.
        """
        cal_stim = θ["k"] @ s
        cal_hist = GLMJax._convolve(p, y, θ["h"])
        cal_weight = θ["w"] @ y
        # Necessary padding since history convolution shrinks M.
        cal_weight = np.hstack((np.zeros((p['N_lim'], p['dh'])), cal_weight[:, p['dh'] - 1:p['M_lim'] - 1]))

        return θ["b"], cal_weight, cal_hist, cal_stim, np.log(p['dt'])


    @staticmethod
    @partial(jit, static_argnums=(1,))
    def _ll(θ: Dict, p: Dict, m, n, y, s, indicator) -> DeviceArray:
        """
        Return negative log-likelihood of data given model.
        ℓ1 and ℓ2 regularizations are specified in params.
        """
        linear = GLMJax._run_linear(θ, p, y, s)
        log_r̂ = linear[0] + linear[1] + linear[2] + linear[3] + linear[4]  # Broadcast.

        r̂ = np.exp(log_r̂)
        r̂ *= indicator

        l1 = p['λ1'] * np.sum(np.abs(θ["w"])) / (np.sqrt(m) * n**2)
        l2 = p['λ2'] * np.sum(θ["w"] ** 2) / (2 * np.sqrt(m) * n**2)

        return (np.sum(r̂) - np.sum(y * log_r̂)) / (m * n ** 2) + l1 + l2

    @staticmethod
    @partial(jit, static_argnums=(0,))
    def _convolve(p: Dict, y, θ_h) -> DeviceArray:
        """
        Sliding window convolution for history terms.
        """
        cvd = np.zeros((y.shape[0], y.shape[1] - p['dh']))
        for i in np.arange(p['dh']):
            w_col = np.reshape(θ_h[:, i], (p['N_lim'], 1))
            cvd += w_col * y[:, i:p['M_lim'] - (p['dh'] - i)]
        return np.hstack((np.zeros((p['N_lim'], p['dh'])), cvd))

    @property
    def theta(self) -> Dict:
        return self.get_params(self._θ)

    @property
    def weights(self) -> onp.ndarray:
        return onp.asarray(self.theta['w'][:self.current_N, :self.current_N])

    def __repr__(self):
        return f'simGLM({self.params}, θ={self.theta}, gpu={self.use_gpu})'

    def __str__(self):
        return f'simGLM Iteration: {self.iter}, \n Parameters: {self.params})'


class GLMJaxSynthetic(GLMJax):
    def __init__(self, *args, data=None, offline=False, **kwargs):

        """
        GLMJax with data handling. Data are given beforehand to the constructor.

        If offline:
            If M_lim == y.shape[1]: The entire (y, s) is passed into the fit function.
            If M_lim > y.shape[1]: A random slice of width M_lim is used. See `self.rand`.

        If not offline:
            Data of width `self.iter` are used until `self.iter` > M_lim.
            Then, a sliding window of width M_lim is used instead.
        """

        super().__init__(*args, **kwargs)

        self.y, self.s = data
        self.offline = offline
        if self.offline:
            self.rand = onp.zeros(0)  # Shuffle, batch training.
            self.current_M = self.params['M_lim']
            self.current_N = self.params['N_lim']

        assert self.y.shape[1] == self.s.shape[1]
        assert self.y.shape[1] >= self.current_M

    @profile
    def fit(self, return_ll=False, indicator=None):
        if self.offline:
            if self.iter % 10000 == 0:
                self.rand = onp.random.randint(low=0, high=self.y.shape[1] - self.params['M_lim'] + 1, size=10000)

            i = self.rand[self.iter % 10000]
            args = (self.params['M_lim'], self.params['N_lim'],
                    self.y[:, i:i + self.params['M_lim']],
                    self.s[:, i:i + self.params['M_lim']], self.ones)

        else:
            if self.iter < self.params['M_lim']:  # Increasing width.
                y_step = self.y[:, :self.iter + 1]
                stim_step = self.s[:, :self.iter + 1]
                args = self._check_arrays(y_step, stim_step, indicator=indicator)
            else:  # Sliding window.
                if indicator is None:
                    indicator = self.ones
                y_step = self.y[:, self.iter - self.params['M_lim']: self.iter]
                stim_step = self.s[:, self.iter - self.params['M_lim']: self.iter]
                args = (self.params['M_lim'], self.params['N_lim'], y_step, stim_step, indicator)

        if return_ll:
            self._θ, ll = self._fit_ll(self._θ, self.params, self.opt_update, self.get_params, self.iter, *args)
            self.iter += 1
            return ll
        else:
            self._θ = self._fit(self._θ, self.params, self.rpf, self.opt_update, self.get_params, self.iter, *args)
            self.iter += 1


if __name__ == '__main__':  # Test
    key = random.PRNGKey(42)

    N = 2
    M = 100
    dh = 2
    ds = 8
    p = {'N': N, 'M': M, 'dh': dh, 'ds': ds, 'dt': 0.1, 'n': 0, 'N_lim': N, 'M_lim': M}

    w = random.normal(key, shape=(N, N)) * 0.001
    h = random.normal(key, shape=(N, dh)) * 0.001
    k = random.normal(key, shape=(N, ds)) * 0.001
    b = random.normal(key, shape=(N, 1)) * 0.001

    theta = {'h': np.flip(h, axis=1), 'w': w, 'b': b, 'k': k}
    model = GLMJax(p, theta)

    sN = 8  #
    data = onp.random.randn(sN, 2)  # onp.zeros((8, 50))
    stim = onp.random.randn(ds, 2)
    print(model.ll(data, stim))
