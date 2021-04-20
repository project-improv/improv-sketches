import pickle
import random

import matplotlib.pyplot as plt
import networkx as nx
import numba
import seaborn as sns
import numpy as np
import scipy
from numpy import matlib as mb
import time

from GLM.utils import *

sns.set()


class DataGenerator:
    """
    Class for synthetic data generation.
    Can both generate θ or spikes from θ or both.
    """

    def __init__(self, params, params_theta=None):
        """
        If theta is not given, one will be provided for you based on params_θ.
        """
        self.params = params.copy()
        self.params_θ = None if params_theta is None else params_theta.copy()
        self.theta = self._gen_theta(**self.params_θ)

        self.Qstim, self.Qbasstim, self.ihtstim = self.makeCosBasis(10, 1, np.asarray([0,150]), 0.02)
        self.Qspike, self.Qbasspike, self.ihtspike = self.makeCosBasis(7, 1, np.asarray([2,90]), 0.02)

        self.Qbasspike[0:2][:] = 0

        refract = np.zeros((100, 2))
        refract[0:2][:]=1

        self.Qbasspike = np.hstack((refract, self.Qbasspike))

        self.i = 0

    def gen_new_theta(self):
        if 'seed' not in self.params_θ:
            self.params_θ['seed'] = 1
        else:
            self.params_θ['seed'] += 1

        return self._gen_theta(**self.params_θ)

    def _gen_theta(self, seed=0, p_inh=0.5, base=0, connectedness=3, p_rand=0., rand_w=False, max_w=0.05, **kwargs):
        '''
        Generates model parameters based on the Watt-Strogatz small world model.
        Weights for θ_w are binary {-max_w, max_w} by default.
        θ_b is `base` and is the same for all neurons.
        θ_h is an exponentially decaying negative feedback, also same for all neurons.

        :param seed: Seed.
        :param p_inh: Percent of neurons with negative weights.
        :param base: Value for θ_b

        TODO: Set parameters for θ_k.

        Returns a theta dictionary:
        theta['h']: history filters for n neurons (N x dh)
        theta['w']: coupling filters for n neurons (N x N) aka weight matrix
        theta['b']: baseline firing (N x 1)
        theta['k']: stimulus factor (N x ds)
        '''

        random.seed(seed)  # For networkx.
        np.random.seed(seed)

        # get parameters
        dh = self.params['dh']
        N = self.params['N']

        # store model as dictionary
        theta = dict()
        n_inh = int(N * p_inh)
        neutype = np.concatenate((-1 * np.ones(n_inh), np.ones(N - n_inh)))
        np.random.shuffle(neutype)

        # baseline rates
        theta['be'] = np.zeros(N)
        theta['bi'] = np.zeros(N)
        theta['be'][:] = 0  # 2 * np.random.rand(np.sum(excinh == 1)) - 2
        theta['bi'][:] = 0  # 1 + np.random.rand(np.sum(excinh == -1))


        # history filter over time dh
        #theta['h'] = np.zeros((N, dh))
        theta['h']= np.ones((N,1))-10 # mag (1.5e-3, 2e-1)

        theta['ke'] = self.gen_theta_k() if 'ds' in self.params else np.zeros((N, 1))
        theta['ki'] = self.gen_theta_k() if 'ds' in self.params else np.zeros((N, 1))

        theta['ps'] = self.gen_theta_ps()

        return theta

    def gen_theta_k(self, r=0.5, sd=1):
        """
        Generate θ_k. Based on the von Mises distribution.
        A Gaussian centered at each of the ds spread out on the unit circle.
        The first N // ds neurons would be centered at 0. The next would be centered at 1 and so on.

        :param r: Scaling factor.
        :param sd: Standard deviation of the normal.
        :return θ_k (N x ds)
        """

        N = self.params['N']
        ds = self.params['ds']

        base = np.random.rand(N, 10)/10
        base = (base-0.05)*2

        #center = ds // 2
        #bell = np.array([r * np.exp(-((i - center) ** 2 / sd ** 2)) for i in range(ds)])
        #chunk = N // ds
        #for n in range(ds):
        #    base[n * chunk: np.clip((n + 1) * chunk, a_min=0, a_max=N), :] = np.roll(bell, n - center)

        return base

    def gen_theta_ps(self):

        N= self.params['N']

        refract = np.zeros((N, 2))
        refract[:,0]= -1
        refract[:,1]= -0.2
        base = -np.random.rand(N, 7)/20

        return np.hstack((refract, base))

    def gen_spikes(self, params=None, **kwargs):
        """
        Wrapper for self._gen_spikes. Need for dealing with cases without stimulus input.
        """
        p = self.params.copy() if params is None else params.copy()
        if 'ds' not in p:
            p['ds'] = 1
        return self._gen_spikes(self.theta['h'], self.theta['be'], self.theta['bi'], self.theta['ke'], self.theta['ki'], self.theta['ps'],
                p['dt'], p['dh'], p['ds'], p['N'], p['M'], self.Qbasstim, self.Qbasspike, **kwargs)


    @staticmethod
    @numba.jit(nopython=True)
    def _gen_spikes(h, be, bi, ke, ki, ps, dt, dh, ds, N, M, Q, Qspike, seed=2, limit=20., stim_int=10):
        '''
        Generates spike counts from the model

        :param limit: Maximum firing rate. Anything above will be clipped.
        :param stim_int: Duration of each stimulus.
            Each stimulus is active for a duration of `stim_int` and cycles deterministically from first to last.

        Returns a data dict:
        data['r']: firing rates (lambda) for each time step.
        data['y']: spikes for each time step.
        data['s']: active stimulus over time.
        '''
        np.random.seed(seed)

        El = -60
        Ee = 0
        Ei = -80
        gl = 0.5
        Vinit = -60
        a = 0.45
        b = 53*a
        c = 90

        # nonlinearity; exp()

        r = np.zeros((N, M))  # rates
        y = np.zeros((N, M))  # spikes
        s = np.zeros(M)
        #s = np.random.rand(ds, M)
        V = np.zeros((N, M))

        # the initial rate (no history); generate randomly
        r[:, 0] = 0
        V[:, 0] = Vinit
        y[:, 0] = np.array([np.random.poisson(r[i, 0]) for i in range(N)])

        for t in range(M):
            stim_curr = (t//stim_int) %ds
            s[t] = (stim_curr-4)/10


        # simulate the model for next M samples (time steps)

        stimE = ke @ np.transpose(Q) #NX100
        stimI = ki @ np.transpose(Q)

        spikefilt = ps @ np.transpose(Qspike) #NX90

        for i in range(N):  # step through neurons
            # compute model firing rate

            stimbaseE = np.convolve(stimE[i, :], s) + be[i] #convolve produces M size array
            stimbaseI = np.convolve(stimI[i, :], s) + bi[i]

            ge= np.log(1+np.exp(stimbaseE)) #M size
            gi= np.log(1+np.exp(stimbaseI))

            gtot= ge +gi + gl  #M size
            Itot= ge*Ee +gi*Ei +gl*El  

            for t in range(M):
                spikeconv = np.fliplr(spikefilt)
                if t==0:
                    hist = 0
                elif t < 100:
                    hist = np.sum(np.multiply(y[i, 0:t], spikeconv[i, -t:]))
                else:
                    hist = np.sum(np.multiply(y[i, t-100:t], spikeconv[i,:]))

                if t ==0:
                    V[i, t] = Vinit
                else:
                    V[i, t] = np.exp(-dt*gtot[t])*(V[i, t-1]-Itot[t]/gtot[t])+Itot[t]/gtot[t] 

                V[i, t]= V[i, t] + hist

                r[i, t] = c*np.log(1+np.exp(a*V[i,t]+b))

                y[i, t] = np.random.poisson(r[i, t] * dt)

        return r, y, s, V


    def plot_theta(self, name):
        scale = np.max(np.abs(self.theta[name]))
        x = plt.imshow(self.theta[name], vmin=-scale, vmax=scale)
        x.set_cmap('bwr')
        plt.grid(False)
        plt.colorbar()
        plt.show()

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


if __name__ == '__main__':
    n = 50
    dh = 2
    ds = 8
    m = 5000
    dt = 0.001

    p = {
        'N': n,
        'dh': dh,
        'M': m,
        'dt': dt,
        'ds': ds
    }

    params_θ = {
        'seed': 3,
        'p_inh': 0.6,
        'p_rand': 0.,
        'base': 0.3,
        'connectedness': 9,
        'max_w': 0.05
    }

    gen = DataGenerator(params=p, params_theta=params_θ)
    
    Q, Qbas, iht = gen.makeCosBasis(7, 1, np.asarray([2,90]), 0.02)

    '''

    plt.subplot(2,1,1)
    for i in range(Q.shape[1]):
        plt.plot(iht, Q[:,i])
    plt.title('Orthogonal basis for cosine bumps')
    plt.xlabel('time in ms')

    plt.subplot(2,1,2)
    for i in range(Qbas.shape[1]):
        plt.plot(iht, Qbas[:,i])
    plt.title('Stimulus basis cosine bumps')
    plt.xlabel('time in ms')
    plt.show()
    
    '''
    # %% Save θ
    with open('theta_dict.pickle', 'wb') as f:
        pickle.dump(gen.theta, f)
    with open('params_dict.pickle', 'wb') as f:
        pickle.dump(p, f)
    print('Simulating model...')

    start_time = time.time()

    # %% Generate data
    r, y, s, V = gen.gen_spikes(seed=0)


    plt.subplot(2,1,1)
    plt.plot(gen.theta['ke'][25,:]@ np.transpose(gen.Qbasstim))
    plt.title('Excitatory stimulus filter')
    plt.xlabel('time in ms')

    plt.subplot(2,1,2)
    plt.plot(gen.theta['ki'][25,:]@ np.transpose(gen.Qbasstim))
    plt.title('Inhibitory stimulus filter')
    plt.xlabel('time in ms')
    plt.show()
    

    print("--- %s seconds ---" % (time.time() - start_time))

    print('log_likelihood= '+str(-np.mean(np.sum(y*np.log(1-np.exp(-r*dt))-(1-y)*r*dt, axis=1))))

    data = {'y': y.astype(np.uint8), 'r': r, 's':s, 'V':V}
    print('Spike Counts:')
    print('mean: ', np.mean(data['y']))
    print('var.: ', np.var(data['y']), '\n')
    print('Rates:')
    print('mean: ', np.mean(data['r']))
    print('var.: ', np.var(data['r']))
    print('*** %g percent of rates are over the limit. ***' % (100 * np.mean(data['r'] > 10)))
    print(np.percentile(data['y'], [50, 90, 99]))
    print('Saving data.')
    np.savetxt('data_sample.txt', data['y'])
    np.savetxt('rates_sample.txt', data['r'])
    np.savetxt('stim_sample.txt', data['s'])
    np.savetxt('volt_sample.txt', data['V'])

    fig, ax = plt.subplots(dpi=100)
    u = ax.imshow(r[:, :])
    ax.grid(0)
    fig.colorbar(u)
    ax.set_title('Synthetic Spiking Data')
    plt.show()
    