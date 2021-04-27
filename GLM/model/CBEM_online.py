# -*- coding: utf-8 -*-

from functools import partial
from importlib import import_module
from typing import Dict, Tuple

import jax.numpy as np
import numpy as onp
from numpy import matlib as mb
import scipy
from jax import devices, jit, random, grad, value_and_grad
import jax
from jax.config import config
from jax.experimental.optimizers import OptimizerState
from jax.interpreters.xla import DeviceArray
import matplotlib.pyplot as plt
import pickle
from jax.experimental import loops

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

        self.Qstim, self.Qbasstim, self.ihtstim = self.makeCosBasis(10, 1, np.asarray([0,150]), 0.02)
        self.Qspike, self.Qbasspike, self.ihtspike = self.makeCosBasis(7, 1, np.asarray([2,90]), 0.02)

        self.Qbasspike = jax.ops.index_update(self.Qbasspike, jax.ops.index[0:2,:], 0)

        refract = np.zeros((100, 2))
        refract = jax.ops.index_update(refract, jax.ops.index[0:2,:], 1)

        self.Qbasspike = np.hstack((refract, self.Qbasspike))

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

    def makeCosBasis(self, nb, dt, endpoints, b):
        
        yrange = np.log(endpoints+b+10**-20)
        db = np.diff(yrange)/(nb-1)
        ctrs = np.arange(yrange[0], yrange[1]+0.9*db, db)
        mxt = 100
        iht = np.expand_dims(np.arange(0,mxt,dt),axis=1)
        nt = iht.shape[0]

        nliniht = np.log(iht+b+10**-20)

        x = mb.repmat(nliniht, 1, nb)
        c = mb.repmat(ctrs, nt, 1)

        ihbasis = self.ff(x, c, db)

        return scipy.linalg.orth(ihbasis), ihbasis, iht

    def ff(self, x, c, dc):
    
        radians= (x-c)*np.pi/dc/2

        mini = np.minimum(np.pi, radians)
        maxi = np.maximum(-np.pi, mini)

        ret= (np.cos(maxi)+1)/2

        return ret

    def ll(self, y, s, indicator=None):
        return self._ll(self.theta, self.params, *self._check_arrays(y, s, indicator))

    @profile
    def fit(self, y, s, Vin, ycurr, return_ll=False, indicator=None):
        """
        Fit model. Returning log-likelihood is ~2 times slower.
        """
        if return_ll:
            self._θ, ll = GLMJax._fit_ll(self._θ, self.params, self.opt_update, self.get_params,
                                                    self.iter, *self._check_arrays(y, s, indicator))
            self.iter += 1
            return ll
        else:
            self._θ, Vret, y = GLMJax._fit(self._θ, self.params, self.rpf, self.opt_update, self.get_params,
                                             self.iter, Vin, ycurr, self.Qbasspike, self.Qbasstim, *self._check_arrays(y, s, indicator))
            self.iter += 1
            return Vret, y

    @staticmethod
    @partial(jit, static_argnums=(1, 2, 3))
    def _fit_ll(θ: Dict, p: Dict, opt_update, get_params, iter, Vin, ycurr, m, n, y, s, indicator):
        ll, Δ = value_and_grad(GLMJax._ll)(get_params(θ), p, m, n, y, s, indicator)
        θ = opt_update(iter, Δ, θ)
        return θ, ll

    @staticmethod
    @partial(jit, static_argnums=(1, 2, 3, 4))
    def _fit(θ: Dict, p: Dict, rpf, opt_update, get_params, iter, Vin, ycurr, Qspike, Qstim, m, n, y, s, indicator):
        for i in range(rpf):

            Δ = grad(GLMJax._ll)(get_params(θ), p, m, n, y, s, Vin, ycurr, Qspike, Qstim, indicator)
            θ = opt_update(iter, Δ, θ)

        theta = get_params(θ)
        El = -60
        Ee = 0
        Ei = -80
        gl = 0.5

        spikefilt = theta['ps'] @ np.transpose(Qspike)

        yuse = y[:,0:100]
        cal_hist = np.sum(np.multiply(yuse, np.fliplr(spikefilt)), axis=1)

        stimE = ke @ np.transpose(Qstim) #NX150
        stimI = ki @ np.transpose(Qstim)

        suse = np.reshape(s[0:100], (1, 100))

        stimbaseE = np.sum(np.multiply(suse, np.fliplr(stimE)), axis=1) + theta['be'] #convolve produces M size array
        stimbaseI = np.sum(np.multiply(suse, np.fliplr(stimI)), axis=1) + theta['bi']

        ge = np.log(1+ np.exp(stimbaseE))
        gi = np.log(1+ np.exp(stimbaseI))

        gtot = gl + ge +gi
        Itot = gl*El + ge*Ee + gi*Ei

        Vret = np.multiply(np.exp(-p['dt']*gtot[:,0]),(Vin-Itot[:,0]/gtot[:,0]))+Itot[:,0]/gtot[:,0] 
        Vret = Vret+cal_hist

        return θ, Vret, y[:, 0]

    def grad(self, y, s, indicator=None) -> DeviceArray:
        return grad(self._ll)(self.theta, self.params, *self._check_arrays(y, s, indicator)).copy()

    def _increase_θ_size(self) -> None:
        """
        Doubles θ capacity for N in response to increasing N.

        """
        N_lim = self.params['N_lim']
        print(f"Increasing neuron limit to {2 * N_lim}.")
        self._θ = self.theta

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
    def _ll(θ: Dict, p: Dict, m, n, y, s, V, ycurr, Qspike, Qstim, indicator) -> DeviceArray:
        """
        Return negative log-likelihood of data given model.
        ℓ1 and ℓ2 regularizations are specified in params.
        """

        El = -60
        Ee = 0
        Ei = -80
        gl = np.ones((y.shape[0],1))*0.5
        a = 0.45
        b = 53*a
        c = 90

        ke= θ['ke']
        ki= θ['ki']
        be= θ['be']
        bi= θ['bi']
        ps= θ['ps']

        spikefilt = ps @ np.transpose(Qspike)

        stimE = ke @ np.transpose(Qstim) #NX150
        stimI = ki @ np.transpose(Qstim)

        def V_loop(y, V, s):

            with loops.Scope() as sc:
                sc.r= np.zeros((y.shape[0], 10))

                for t in range(sc.r.shape[1]):

                    for _ in sc.cond_range(t==0):
                        Vnow= V
                        cal_hist = np.sum(np.multiply(y[:,t:t+100], np.fliplr(spikefilt)), axis=1)

                    for _ in sc.cond_range(t!=0):

                        stimbaseE = np.sum(np.multiply(np.reshape(s[t:t+100], (1,100)), np.fliplr(stimE)), axis=1) 
                        stimbaseI = np.sum(np.multiply(np.reshape(s[t:t+100], (1,100)), np.fliplr(stimI)), axis=1) 

                        ge = np.log(1+ np.exp(stimbaseE +be))
                        gi = np.log(1+ np.exp(stimbaseI +bi))

                        gtot = gl + ge + gi 
                        Itot = gl*El + ge*Ee + gi*Ei

                        Vnow = np.multiply(np.exp(-p['dt']*gtot[:,t]), (V-Itot[:,t]/gtot[:,t]))+Itot[:,t]/gtot[:,t]
                        cal_hist = np.sum(np.multiply(y[:,t:t+100], np.fliplr(spikefilt)), axis=1)

                    V = Vnow+cal_hist

                    sc.r = jax.ops.index_update(sc.r, jax.ops.index[:,t], c*np.log(1+np.exp(a*V+b)))

                return sc.r 

        r = V_loop(y, V, s)

        return -np.mean(np.sum(y[:,-10:]*np.log(1-np.exp(-r*p['dt'])+0.000001)-(1-y[:,-10:])*r*p['dt'], axis=1))

    def ll_step(self, y, s, p, Vin):
        
        El = -60
        Ee = 0
        Ei = -80
        gl = 0.5
        a = 0.45
        b = 53*a
        c = 90

        θ = self.theta

        ke= θ['ke']
        ki= θ['ki']
        be= θ['be']
        bi= θ['bi']
        ps= θ['ps']

        r= np.zeros((y.shape[0], 10))
        V= np.zeros((y.shape[0], 10))

        spikefilt = ps @ np.transpose(self.Qbasspike)

        stimE = ke @ np.transpose(self.Qbasstim) #NX150
        stimI = ki @ np.transpose(self.Qbasstim)

        for t in range(10):

            if t==0:
                Vnow= Vin
                cal_hist = np.sum(np.multiply(y[:,t:t+100], np.fliplr(spikefilt)), axis=1)

            else:

                stimbaseE = np.sum(np.multiply(np.reshape(s[t:t+100], (1,100)), np.fliplr(stimE)), axis=1)
                stimbaseI = np.sum(np.multiply(np.reshape(s[t:t+100], (1,100)), np.fliplr(stimI)), axis=1) 

                ge = np.log(1+ np.exp(stimbaseE + be))
                gi = np.log(1+ np.exp(stimbaseI + bi))

                gtot = gl + ge + gi 
                Itot = gl*El + ge*Ee + gi*Ei


                Vnow = np.multiply(np.exp(-p['dt']*gtot[:,t]), (Vcurr-Itot[:,t]/gtot[:,t]))+Itot[:,t]/gtot[:,t]
                cal_hist = np.sum(np.multiply(y[:,t:t+100], np.fliplr(spikefilt)), axis=1)

            Vcurr = Vnow +cal_hist

            V = jax.ops.index_update(V, jax.ops.index[:,t], Vcurr)
            
            r = jax.ops.index_update(r, jax.ops.index[:,t], c*np.log(1+np.exp(a*Vcurr+b)))
        return -np.mean(np.sum(y[:, -10:]*np.log(1-np.exp(-r*p['dt'])+0.00001)-(1-y[:, -10:])*r*p['dt'], axis=1)), r, V

    def ll_full(self, y, s, p):
        
        El = -60
        Ee = 0
        Ei = -80
        gl = 0.5
        a = 0.45
        b = 53*a
        c = 90

        θ = self.theta

        ke= θ['ke']
        ki= θ['ki']
        be= θ['be']
        bi= θ['bi']
        ps= θ['ps']

        r= np.zeros((p['N_lim'], p['M_lim']))
        V= np.zeros((p['N_lim'], p['M_lim']))

        spikefilt = ps @ np.transpose(self.Qbasspike)

        stimE = ke @ np.transpose(self.Qbasstim) #NX150
        stimI = ki @ np.transpose(self.Qbasstim)

        Vin= np.ones(p['N_lim'])*-60

        for t in range(p['M_lim']):

            if t<100:
                Vnow = Vin
                cal_hist = 0

            else:

                stimbaseE = np.sum(np.multiply(np.reshape(s[t-100:t], (1,100)), np.fliplr(stimE)), axis=1) + be
                stimbaseI = np.sum(np.multiply(np.reshape(s[t-100:t], (1,100)), np.fliplr(stimI)), axis=1) + bi

                ge = np.log(1+ np.exp(stimbaseE + be))
                gi = np.log(1+ np.exp(stimbaseI + bi))

                gtot = gl + ge +gi
                Itot = gl*El + ge*Ee + gi*Ei

                Vnow = np.multiply(np.exp(-p['dt']*gtot[:,t]),(V[:,t-1]-Itot[:,t]/gtot[:,t]))+Itot[:,t]/gtot[:,t]
                cal_hist = np.sum(np.multiply(y[:,t-100:t], np.fliplr(spikefilt)), axis=1)

                Vcurr = Vnow +cal_hist

                V = jax.ops.index_update(V, jax.ops.index[:,t], Vcurr)
                
                r = jax.ops.index_update(r, jax.ops.index[:,t], c*np.log(1+np.exp(a*Vcurr+b)))
        return -np.mean(np.sum(y[:, :]*np.log(1-np.exp(-r*p['dt'])+0.00001)-(1-y[:, :])*r*p['dt'], axis=1))

    def plot_filters(self, n):

        plt.subplot(3,1,1)
        plt.plot(model.theta['ke'][n,:]@ np.transpose(self.Qbasstim))
        plt.plot(ground_theta['ke'][n,:]@ np.transpose(self.Qbasstim))
        plt.legend(['model', 'ground'])
        plt.title('Excitatory stimulus filter')
        plt.xlabel('time in ms')

        plt.subplot(3,1,2)
        plt.plot(model.theta['ki'][n,:]@ np.transpose(model.Qbasstim))
        plt.plot(ground_theta['ki'][n,:]@ np.transpose(model.Qbasstim))
        plt.legend(['model', 'ground'])
        plt.title('Inhibitory stimulus filter')
        plt.xlabel('time in ms')

        plt.subplot(3,1,3)
        plt.plot(model.theta['ps'][n,:]@ np.transpose(model.Qbasspike))
        plt.plot(ground_theta['ps'][n,:]@ np.transpose(model.Qbasspike))
        plt.legend(['model', 'ground'])
        plt.title('Postspike filter')
        plt.xlabel('time in ms')


        plt.savefig('data/filters'+str(n)+'/'+str(i)+'.png')
        
        plt.clf()

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


def MSE(x, y):

    return (np.square(x - y)).mean(axis=None)

def plot_and_save_rv(rcheck, Vcheck, r_ground, V_ground, n, i):

    '''
    r and V come from the model fitting, r_ground and V_ground come from the ground truth data, 
    n is the neuron you want to look at, and i is the current time step 
    This function plots the firing rate and voltage for both the model fit and ground truth
    Can be used to make save frames and eventually make videos of how they change over time
    Please make a folder called 'neuronN' for each neuron you want to look at before running
    '''

    if i < 110:

        plt.subplot(2, 1, 1)
        plt.plot(np.arange(i), c*np.log(1+np.exp(a*np.ones((i,))*-60+b)))
        plt.plot(np.arange(i), np.reshape(r_ground[n,0:i], (i,)))
        plt.legend(['Firing rate model', 'Firing rate ground'])
        plt.xlabel('Time bin in ms')
        plt.ylabel('Firing rate in spikes/s')

        plt.subplot(2,1,2)
        plt.plot(np.arange(i), np.ones((i,))*-60)
        plt.plot(np.arange(i), np.reshape(V_ground[n, 0:i], (i,)))
        plt.legend(['Voltage model', 'Voltage ground'])
        plt.xlabel('Time bin in ms')
        plt.ylabel('Voltage in mV')

        plt.savefig('data/neuron'+str(n)+'/'+str(i)+'.png')
        
        plt.clf()

    elif i<1000: 


        plt.subplot(2, 1, 1)
        plt.plot(np.arange(i), np.reshape(rcheck[n,0:i], (i,)))
        plt.plot(np.arange(i), np.reshape(r_ground[n,0:i], (i,)))
        plt.legend(['Firing rate model', 'Firing rate ground'])
        plt.xlabel('Time bin in ms')
        plt.ylabel('Firing rate in spikes/s')

        plt.subplot(2,1,2)
        plt.plot(np.arange(i), np.reshape(Vcheck[n, 0:i], (i,)))
        plt.plot(np.arange(i), np.reshape(V_ground[n, 0:i], (i,)))
        plt.legend(['Voltage model', 'Voltage ground'])
        plt.xlabel('Time bin in ms')
        plt.ylabel('Voltage in mV')

        plt.savefig('data/neuron'+str(n)+'/'+str(i)+'.png')

        plt.clf()

    else:

        plt.subplot(2, 1, 1)
        plt.plot(np.arange(i), np.reshape(rcheck[n, 0:i], (i,)))
        plt.plot(np.arange(i), np.reshape(r_ground[n, 0:i], (i,)))
        plt.legend(['Firing rate model', 'Firing rate ground'])
        plt.xlabel('Time bin in ms')
        plt.ylabel('Firing rate in spikes/s')

        plt.subplot(2,1,2)
        plt.plot(np.arange(i), np.reshape(Vcheck[n, 0:i],(i,)))
        plt.plot(np.arange(i), np.reshape(V_ground[n,0:i], (i,)))
        plt.legend(['Voltage model', 'Voltage ground'])
        plt.xlabel('Time bin in ms')
        plt.ylabel('Voltage in mV')

        plt.savefig('data/neuron'+str(n)+'/'+str(i)+'.png')

        plt.clf()


if __name__ == '__main__':  # Test
    key = random.PRNGKey(42)

    N = 50
    M = 5000
    dh = 2
    ds = 8
    p = {'N': N, 'M': M, 'dh': dh, 'ds': ds, 'dt': 0.001, 'n': 0, 'N_lim': N, 'M_lim': M, 'λ1': 4, 'λ2':0.0}

    with open('theta_dict.pickle', 'rb') as f:
        ground_theta= pickle.load(f)

    
    base = onp.random.rand(N, 10)/10
    base = (base-0.05)*2

    ke = base/100 #onp.zeros((N,ds)) 
    ki = base/100 #onp.zeros((N,ds)) 

    be = onp.zeros((N,1))
    bi = onp.zeros((N,1))

    refract = np.zeros((N, 2))

    refract = jax.ops.index_update(refract, jax.ops.index[:,0], -0.001)
    refract = jax.ops.index_update(refract, jax.ops.index[:,1], -0.001)
    postspike = -onp.random.rand(N, 7)/100

    ps = np.hstack((refract, postspike))

    y= onp.loadtxt('data_sample.txt')
    s= onp.loadtxt('stim_sample.txt')

    #theta = {'be': ground_theta['be'], 'ke': ground_theta['ke'], 'ki': ground_theta['ki'], 'bi': ground_theta['bi'], 'ps':ground_theta['ps']}

    theta = {'be': be, 'ke': ke, 'ki': ki, 'bi': bi, 'ps':ps}
    model = GLMJax(p, theta, optimizer={'name': 'adam', 'step_size': 1e-2})

    #model_verify = GLMJax(p, ground_theta, optimizer={'name': 'adam', 'step_size': 1e-2})

    n_iters= 1

    indicator= None

    model_ll = np.zeros(M)

    r_ground= onp.loadtxt('rates_sample.txt')
    V_ground = onp.loadtxt('volt_sample.txt')

    a = 0.45
    b = 53*a
    c = 90

    window= 10

    Vcheck = np.ones((N,M))*-60
    rcheck = c*np.log(1+np.exp(a*Vcheck*-60+b))

    for j in range(n_iters):
        Vin= np.ones(p['N_lim'])*-60
        ycurr = y[:,0]

        for i in range(M):

            print(i)

            if i < 110:

                #plot_and_save_rv(rcheck, Vcheck, r_ground, V_ground, 30, i)

                #plot_and_save_rv(rcheck, Vcheck, r_ground, V_ground, 15, i)
                
                #model.plot_filters(30)
                #model.plot_filters(15)


                continue

            elif i<1000:

                print(model.theta['ke'][30,0:10])

                yfit = y[:, i-110:i]
                sfit = s[i-110:i]

                llres, rres, Vres= model.ll_step(yfit, sfit, p, Vin)

                Vcheck = jax.ops.index_update(Vcheck, jax.ops.index[:,i-1], Vres[:,-1])

                rcheck = jax.ops.index_update(rcheck, jax.ops.index[:,i-1], rres[:,-1]) 

                #plot_and_save_rv(rcheck, Vcheck, r_ground, V_ground, 30, i)
                #plot_and_save_rv(rcheck, Vcheck, r_ground, V_ground, 15, i)
                
                #model.plot_filters(30)
                #model.plot_filters(15)


            else:
                print(model.theta['ke'][30,0:10])

                yfit = y[:, i-110:i]
                sfit = s[i-110:i]

                llres, rres, Vres = model.ll_step(yfit, sfit, p, Vin)

                Vcheck = jax.ops.index_update(Vcheck, jax.ops.index[:,0:999], Vcheck[:,1:1000])
                Vcheck = jax.ops.index_update(Vcheck, jax.ops.index[:,999], Vres[:,-1])

                rcheck = jax.ops.index_update(rcheck, jax.ops.index[:,0:999], rcheck[:,1:1000])
                rcheck = jax.ops.index_update(rcheck, jax.ops.index[:, 999], rres[:,-1])

                model_ll = jax.ops.index_update(model_ll, i, llres)

                #plot_and_save_rv(rcheck, Vcheck, r_ground, V_ground, 30, i)
                #plot_and_save_rv(rcheck, Vcheck, r_ground, V_ground, 15, i)
                
                #model.plot_filters(30)
                #model.plot_filters(15)

                

            Vret, ycurr = model.fit(yfit, sfit, Vin, ycurr, return_ll=False, indicator=onp.ones(y.shape))
            Vin = Vret


    plt.plot(model_ll)
    plt.title('model log likelihood over time')

    with open('theta_model.pickle', 'wb') as f:
        pickle.dump(model.theta, f)

    

