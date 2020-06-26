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
from jax.interpreters.xla import DeviceArray
import matplotlib.pyplot as plt
import pickle
import time

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
            assert theta['k'].shape == (p['N_lim'], 3)
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

    def predict(self, y, s, indicator=None):
        y, s, indicator = self._check_arrays(y, s, indicator)[2:]
        linear = GLMJax._run_linear(self.theta, self.params, y, s)
        log_r̂ = linear[0] + linear[1] + linear[2] + linear[3] + linear[4]  # Broadcast.
        return (np.exp(log_r̂) * indicator)[:self.current_N, :self.current_M]

    def residual(self, y, s, indicator=None):
        return y - self.predict(y, s, indicator)

    def linear_contributions(self, y, s, indicator=None):
        y, s, indicator = self._check_arrays(y, s, indicator)[2:]
        linear = GLMJax._run_linear(self.theta, self.params, y, s)
        return (linear[0][:self.current_N], *[(u*indicator)[:self.current_N, :self.current_M] for u in linear[1:4]], linear[4])

    @profile
    def _check_arrays(self, y, s, indicator=None) -> Tuple[onp.ndarray]:
        """
        Check validity of input arrays and pad y and s to (N_lim, M_lim) and (ds, M_lim), respectively.
        Indicator matrix discerns true zeros from padded ones.
        :return current_M, current_N, y, s, indicator
        """

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
            s_ = onp.zeros((M_lim), dtype=onp.float32)
            indicator_ = onp.zeros((N_lim, M_lim), dtype=onp.float32)

            y_[:y.shape[0], :y.shape[1]] = y
            s_[:s.shape[0]] = s

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
        #self._θ['k'] = onp.concatenate((self._θ['k'], onp.zeros((N_lim, self.params['ds']))), axis=0)

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

        a= np.reshape(θ["k"][:,0], (p['N_lim'],1))
        b= np.reshape(θ["k"][:,1], (p['N_lim'],1))
        c= np.reshape(θ["k"][:,2], (p['N_lim'],1))

        A= a @ np.ones((1, p['N_lim'])) 
        B= b @ np.ones((1, p['M_lim']))
        C= c @ np.ones((1, p['M_lim']))

        t= θ["t"]

        gauss= A @ np.exp(-np.divide(np.square(s*(np.pi/4)-B), 2*(C+0.001)**2))

        cal_stim= np.zeros((p['N_lim'],p['M_lim']))

        for m in range(p['M_lim']):
            
            if m in [0, 1, 2, 3, 4]:
                ex= np.zeros((1,m+1))
                ex= jax.ops.index_update(ex, jax.ops.index[:, 0:m+1], [[t[0]*np.exp(-np.square(i-m+t[1])/(2*(t[2]+0.01)**2)) for i in range(m+1)]])
                mult= np.multiply(gauss[:,0:m+1], ex)
                sm= mult @ np.ones((m+1,1))
                cal_stim= jax.ops.index_update(cal_stim, jax.ops.index[:,0:m+1], sm)

            else:
                ex= np.zeros((1,5))
                ex= jax.ops.index_update(ex, jax.ops.index[:, 0:5], [[t[0]*np.exp(-np.square(i-m+t[1])/(2*(t[2]+0.01)**2)) for i in range(m-4, m+1)]])
                mult= np.multiply(gauss[:,m-4:m+1], ex)
                sm= mult @ np.ones((5,1))
                cal_stim= jax.ops.index_update(cal_stim, jax.ops.index[:, m-4:m+1], sm)

        '''

        for n in range(p['N_lim']):
            
            gauss= a[n] * np.exp(-(np.square(s*(np.pi/4)-b[n])/ (2*(c[n]+0.001)**2)))
            gauss1= 0.3 * np.exp(-(np.square(s*(np.pi/4)-np.pi)/ (2*(0.1+0.001)**2)))

            for m in range(p['M_lim']):
                
                ex= np.zeros(p['M_lim'])
                ex=jax.ops.index_update(ex, jax.ops.index[0:m], [np.exp(-np.abs(i-m)) for i in range(m)])

                cal= jax.ops.index_update(cal, jax.ops.index[n,m], np.sum(np.multiply(gauss1, ex)))
                
                if m in [0, 1, 2, 3, 4]:
                    ex= np.zeros(m)
                    ex= jax.ops.index_update(ex, jax.ops.index[0:m], [np.exp(-np.abs(i-m)) for i in range(m)])
                    cal_stim= jax.ops.index_update(cal_stim, jax.ops.index[n,m], np.sum(np.multiply(gauss[0:m], ex)))

                else: 
                    ex= np.zeros(5)
                    ex= jax.ops.index_update(ex, jax.ops.index[0:5], [np.exp(-np.abs(i-m)) for i in range(m-5, m)])
                    cal_stim= jax.ops.index_update(cal_stim, jax.ops.index[n,m], np.sum(np.multiply(gauss[m-5:m], ex)))
        '''   

        cal_hist = GLMJax._convolve(p, y, θ["h"])
        cal_weight = (θ["w"] * (np.eye(p['N_lim']) == 0)) @ y
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

        log_r̂ *= indicator
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

def MSE(x, y):

    return (np.square(x - y)).mean(axis=None)


if __name__ == '__main__':  # Test
    key = random.PRNGKey(42)

    N = 10
    M = 50
    dh = 2
    ds = 8
    p = {'N': N, 'M': M, 'dh': dh, 'ds': ds, 'dt': 1, 'n': 0, 'N_lim': N, 'M_lim': M, 'λ1':4, 'λ2':0.0}

    w = random.normal(key, shape=(N, N)) * 0.001
    h = random.normal(key, shape=(N, dh)) * 0.001
    k = np.zeros((N,3))
    k= jax.ops.index_update(k, jax.ops.index[:, 0], onp.random.rand(N))
    k= jax.ops.index_update(k, jax.ops.index[:, 1], onp.random.rand(N))
    k= jax.ops.index_update(k, jax.ops.index[:, 2], onp.random.rand(N))
    t= onp.random.rand(3)
    b = random.normal(key, shape=(N, 1)) * 0.001
    

    theta = {'h': np.flip(h, axis=1), 'w': w, 'b': b, 'k': k, 't': t}
    model = GLMJax(p, theta, optimizer={'name': 'adam', 'step_size': 1e-3})

    y= onp.loadtxt('data_sample.txt')
    s= onp.loadtxt('stim_info_sample.txt')
    s= np.reshape(s, (50,1)).transpose()

    ll= np.zeros(4000)

    MSEk= np.zeros(4000)
    MSEb= np.zeros(4000)
    MSEw= np.zeros(4000)
    MSEh= np.zeros(4000)
    MSEt= np.zeros(4000)

    indicator= None

    with open('theta_dict.pickle', 'rb') as f:
        ground_theta= pickle.load(f)


    for i in range(4000):
        model.fit(y, s, return_ll=False, indicator=onp.ones(y.shape))

        MSEk= jax.ops.index_update(MSEk, i, MSE(model.theta['k'], ground_theta['k']))
        MSEb= jax.ops.index_update(MSEb, i, MSE(model.theta['b'], ground_theta['b']))
        MSEw= jax.ops.index_update(MSEw, i, MSE(model.theta['w'], ground_theta['w']))
        MSEh= jax.ops.index_update(MSEh, i, MSE(model.theta['h'], ground_theta['h']))
        MSEt= jax.ops.index_update(MSEt, i, MSE(model.theta['t'], ground_theta['t']))

        ll= jax.ops.index_update(ll, i, model.ll(y, s))

    
    fig, axs= plt.subplots(3, 2)
    fig.suptitle('MSE for weights vs #iterations, ADAM, lr=1e-3', fontsize=12)
    axs[0][0].plot(MSEk)
    axs[0][0].set_title('MSEk')
    axs[0][1].plot(MSEb)
    axs[0][1].set_title('MSEb')
    axs[1][0].plot(MSEw)
    axs[1][0].set_title('MSEw')
    axs[1][1].plot(MSEh)
    axs[1][1].set_title('MSEh')
    axs[2][0].plot(MSEt)
    axs[2][0].set_title('MSEt')
    plt.show()
    

    plt.plot(ll)
    plt.title('Guassian fit with exp decay, lr=1e-3, adam')
    plt.xlabel('# iterations')
    plt.ylabel('- log likelihood')
    plt.show()
    
    llfin = model.ll(y,s)

    r_ground= onp.loadtxt('rates_sample.txt')

    indicator= onp.ones(y.shape)

    lin= model._run_linear(model.theta, model.params, y, s)
    log_r= lin[0] + lin[1] + lin[2] + lin[3] + lin[4]
    log_r *= indicator
    r= np.exp(log_r)
    r *= indicator

    print(model.theta['k'])
    print(model.theta['b'])
    print(model.theta['w'])
    print(model.theta['h'])
    
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
    ax1.set_title('Model fit, Guassian with temporal decay')
    ax2.set_title('Ground truth, Gaussian with temporal decay')
    plt.show()
    
    onp.savetxt('rates_model.txt', r)

    mse= MSE(r, r_ground)

    print('MSE loss= ' +str(mse))


    #sN = 8  #
    #data = onp.random.randn(sN, 2)  # onp.zeros((8, 50))
    #stim = onp.random.randn(ds, 2)
    #print(model.ll(data, stim))


