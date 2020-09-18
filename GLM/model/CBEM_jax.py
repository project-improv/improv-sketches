# -*- coding: utf-8 -*-

from functools import partial
from importlib import import_module
from typing import Dict, Tuple

import jax.numpy as np
import numpy as onp
from jax import devices, jit, random, grad, value_and_grad
import jax
from jax.config import config
from jax.experimental.optimizers import OptimizerState
from jax.experimental import loops
from jax.interpreters.xla import DeviceArray
import matplotlib.pyplot as plt
import pickle

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
            assert theta['ke'].shape == (p['N_lim'], p['ds'])
            assert theta['ki'].shape == (p['N_lim'], p['ds'])
            assert (theta['be'].shape == (p['N_lim'],)) or (theta['be'].shape == (p['N_lim'], 1))
            assert (theta['bi'].shape == (p['N_lim'],)) or (theta['bi'].shape == (p['N_lim'], 1))

            if len(theta['be'].shape) == 1:  # Array needs to be 2D.
                theta['be'] = np.reshape(theta['be'], (p['N_lim'], 1))

            if len(theta['bi'].shape) == 1:  # Array needs to be 2D.
                theta['bi'] = np.reshape(theta['bi'], (p['N_lim'], 1))

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

        self.V = np.ones((60,1))*-60
        self.y = np.zeros((N, 1))

    @profile
    def _check_arrays(self, y, s, indicator=None) -> Tuple[onp.ndarray]:
        """
        Check validity of input arrays and pad y and s to (N_lim, M_lim) and (ds, M_lim), respectively.
        Indicator matrix discerns true zeros from padded ones.
        :return current_M, current_N, y, s, indicator
        """
        #assert y.shape[1] == s.shape[1]
        #assert s.shape[0] == self.params['ds']

        self.current_N, self.current_M = y.shape

        N_lim = self.params['N_lim']
        M_lim = self.params['M_lim']

        while y.shape[0] > N_lim:
            self._increase_θ_size()
            N_lim = self.params['N_lim']

        indicator = onp.zeros((N_lim, y.shape[1]), dtype=onp.float32)

        return self.current_M, self.current_N, y, s, indicator

    def ll(self, y, s, return_r= None,indicator=None):
        return self._ll(self.theta, self.params, *self._check_arrays(y, s, indicator))

    @profile
    def fit(self, y, s, return_ll=False, indicator=None):
        """
        Fit model. Returning log-likelihood is ~2 times slower.
        """
        if return_ll:
            self._θ, ll = GLMJax._fit_ll(self._θ, self.params, self.opt_update, self.get_params,
                                                    self.iter, *self._check_arrays(y, s, indicator))
            self.iter += 1
            return ll
        else:
            self._θ = GLMJax._fit(self._θ, self.params, self.rpf, self.opt_update, self.get_params,
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

    def _increase_θ_size(self) -> None:
        """
        Doubles θ capacity for N in response to increasing N.

        """
        N_lim = self.params['N_lim']
        print(f"Increasing neuron limit to {2 * N_lim}.")
        self._θ = self.theta

        self._θ['h'] = onp.concatenate((self._θ['h'], onp.zeros((N_lim, self.params['dh']))), axis=0)
        self._θ['be'] = onp.concatenate((self._θ['b'], onp.zeros((N_lim, 1))), axis=0)
        self._θ['ke'] = onp.concatenate((self._θ['k'], onp.zeros((N_lim, self.params['ds']))), axis=0)
        self._θ['bi'] = onp.concatenate((self._θ['b'], onp.zeros((N_lim, 1))), axis=0)
        self._θ['ki'] = onp.concatenate((self._θ['k'], onp.zeros((N_lim, self.params['ds']))), axis=0)

        self.params['N_lim'] = 2 * N_lim
        self._θ = self.opt_init(self._θ)

    # Compiled Functions
    # A compiled function needs to know the shape of its inputs.
    # Changing array size will trigger a recompilation.
    # See https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#%F0%9F%94%AA-Control-Flow

    @staticmethod
    @partial(jit, static_argnums=(1,))
    def _ll(θ: Dict, p: Dict, m, n, y, s, indicator) -> DeviceArray:
        """
        Return negative log-likelihood of data given model.
        ℓ1 and ℓ2 regularizations are specified in params.
        """

        El = -60
        Ee = 0
        Ei = -80
        gl = 0.5
        a = 0.45
        b = 53*a
        c = 90

        ke= θ['ke']
        ki= θ['ki']
        be= θ['be']
        bi= θ['bi']

        stim_exc = ke @ s 
        stim_inh = ki @ s

        ge = np.log(1+ np.exp(stim_exc + be))
        gi = np.log(1+ np.exp(stim_inh + bi))

        gtot = gl + ge +gi
        Itot = gl*El + ge*Ee + gi*Ei

        def V_loop(y, gtot, Itot):

            with loops.Scope() as s:
                s.r= np.zeros(y.shape)

                for t in range(s.r.shape[1]):
                    for _ in s.cond_range(t==0):
                        Vnow = np.ones(p['N_lim'])*-60
                        V= np.ones(p['N_lim'])*-60
                        cal_hist = 0

                    for _ in s.cond_range(t!=1):
                        Vnow = np.multiply(np.exp(-p['dt']*gtot[:,t]),(V-Itot[:,t]/gtot[:,t])+Itot[:,t]/gtot[:,t])
                        cal_hist = -10*y[:,t-1]

                        V = Vnow+cal_hist

                    s.r = jax.ops.index_update(s.r, jax.ops.index[:,t], c*np.log(1+np.exp(a*V+b)))

                return s.r 

        r = V_loop(y, gtot, Itot)

        '''
        for t in range(p['M_lim']):

            if t == 0:

                Vnow = np.ones(p['N_lim'])*-60
                cal_hist = 0

            else:
                cal_hist = -10*y[:,t-1]

                Vnow = np.multiply(np.exp(-p['dt']*gtot[:,t]),(V[:,t-1]-Itot[:,t]/gtot[:,t])+Itot[:,t]/gtot[:,t])

            V= jax.ops.index_update(V, jax.ops.index[:,t], Vnow+cal_hist)

            rnow = c*np.log(1+np.exp(a*V[:,t]+b))

            r = jax.ops.index_update(r, jax.ops.index[:,t], rnow)
        '''

        return -np.mean(np.sum(y*np.log(1-np.exp(-r*p['dt']))-(1-y)*r*p['dt'], axis=1))

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

def MSE(x, y):

    return (np.square(x - y)).mean(axis=None)


if __name__ == '__main__':  # Test
    key = random.PRNGKey(42)

    N = 10
    M = 50
    dh = 2
    ds = 8
    p = {'N': N, 'M': M, 'dh': dh, 'ds': ds, 'dt': 0.001, 'n': 0, 'N_lim': N, 'M_lim': M, 'λ1':4, 'λ2':0.0}

    ke = onp.random.rand(N, ds)
    ki = onp.random.rand(N, ds)

    be = random.normal(key, shape=(N, 1)) * 0.01
    bi = random.normal(key, shape=(N, 1)) * 0.01

    theta = {'be': be, 'ke': ke, 'ki': ki, 'bi': bi}
    model = GLMJax(p, theta, optimizer={'name': 'adam', 'step_size': 1e-3})

    y= onp.loadtxt('data_sample.txt')
    s= onp.loadtxt('stim_sample.txt')

    MSEke= np.zeros(200)
    MSEbe= np.zeros(200)
    MSEki= np.zeros(200)
    MSEbi= np.zeros(200)


    indicator= None

    with open('theta_dict.pickle', 'rb') as f:
        ground_theta= pickle.load(f)

    window= 10

    for i in range(200):

        print(i)

        model.fit(y, s, return_ll=False, indicator=onp.ones(y.shape))

        MSEke= jax.ops.index_update(MSEke, i, MSE(model.theta['ke'], ground_theta['ke']))
        MSEbe= jax.ops.index_update(MSEbe, i, MSE(model.theta['be'], ground_theta['be']))
        MSEki= jax.ops.index_update(MSEki, i, MSE(model.theta['ki'], ground_theta['ki']))
        MSEbi= jax.ops.index_update(MSEbi, i, MSE(model.theta['bi'], ground_theta['bi']))
    
    llfin = model.ll(y,s, return_r= True)

    r_ground= onp.loadtxt('rates_sample.txt')

    indicator= onp.ones(y.shape)

    log_r= lin[0] + lin[1] + lin[2] + lin[3] + lin[4]
    log_r *= indicator
    r= np.exp(log_r)
    r *= indicator
    
    fig, (ax1, ax2) = plt.subplots(2)
    u1 = ax1.imshow(r[:,:])
    ax1.grid(0)
    u2 = ax2.imshow(r_ground[:,:])
    ax2.grid(0)
    fig.colorbar(u1)
    ax1.set_xlabel('time steps')
    ax2.set_xlabel('time steps')
    ax1.set_ylabel('neurons')
    ax2.set_ylabel('neurons')
    ax1.set_title('Ground truth, single cos')
    ax2.set_title('Model fit, single cos')
    plt.show()
    
    onp.savetxt('rates_model.txt', r)

    mse= MSE(r, r_ground)

    print('MSE loss= ' +str(mse))



