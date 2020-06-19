import pickle
import random

import matplotlib.pyplot as plt
import networkx as nx
import numba
import seaborn as sns
import numpy as np

from GLM.utils import *

sns.set()


class DataGenerator:
    """
    Class for synthetic data generation.
    Can both generate θ or spikes from θ or both.
    """

    def __init__(self, params, params_theta=None, theta=None):
        """
        If theta is not given, one will be provided for you based on params_θ.
        """
        self.params = params.copy()
        self.params_θ = None if params_theta is None else params_theta.copy()
        if theta is None:
            self.theta = self._gen_theta(**self.params_θ)
        else:
            self.theta = {k: np.asarray(v, dtype=np.float64) for k, v in theta.items()}
            assert self.theta['w'].shape == (self.params['N'], self.params['N'])
            assert self.theta['h'].shape == (self.params['N'], self.params['dh'])
            assert self.theta['k'].shape == (self.params['N'], self.params['ds'])
            if len(self.theta['b'].shape) == 2:
                self.theta['b'] = self.theta['b'].flatten()
            assert len(self.theta['b']) == params['N']

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

        # Parameters for θ_w
        :param connectedness: Number of neighboring neurons to connect with (see networkx documentation).
        :param p_rand: Percent of connections to be randomized (by location not weight).
        :param rand_w: Randomize θ_w weight within [-max_w, max_w].
        :param max_w: Upper absolute bound of weight value.

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
        theta['b'] = np.zeros(N)
        theta['b'][neutype == 1] = [base] * np.sum(neutype == 1)  # 2 * np.random.rand(np.sum(excinh == 1)) - 2
        theta['b'][neutype == -1] = [base] * np.sum(neutype == -1)  # 1 + np.random.rand(np.sum(excinh == -1))

        # coupling filters
        theta['w'] = max_w * np.random.random(size=(N, N)) if rand_w else max_w * np.ones((N, N))
        G = nx.connected_watts_strogatz_graph(N, connectedness, p_rand)

        theta['w'] *= nx.adjacency_matrix(G).todense()
        theta['w'] *= neutype  # - (excinh == -1) * 1  # Scaling factor for inhibitory neurons.
        theta['w'] = theta['w'].T

        # for i in range(N):  # Add inhibitory connections.
        #     if inh[i] == -1:
        #         theta['w'][i, np.random.randint(0, N, int(np.sqrt(N)))] = np.random.rand(int(np.sqrt(N))) * np.min(theta['w'][i,:])

        # history filter over time dh
        theta['h'] = np.zeros((N, dh))
        tau = np.linspace(1, 0, dh).reshape((dh, 1))
        theta['h'][neutype == 1, :] = -0.1 * np.exp(-3 * tau).T  # mag (1.5e-3, 2e-1)
        theta['h'][neutype == -1, :] = -0.1 * np.exp(-3 * tau).T  # mag (7e-2, 5e-1)

        theta['k'] = self.gen_theta_k() if 'ds' in self.params else np.zeros((N, 1))

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

        base = np.zeros((N, 3))

        base[:,0]= np.random.rand(N)
        base[:,1]= np.random.rand(N)
        base[:,2]= np.random.rand(N)

        #center = ds // 2
        #bell = np.array([r * np.exp(-((i - center) ** 2 / sd ** 2)) for i in range(ds)])
        #chunk = N // ds
        #for n in range(ds):
        #    base[n * chunk: np.clip((n + 1) * chunk, a_min=0, a_max=N), :] = np.roll(bell, n - center)
        return base

    def gen_spikes(self, params=None, **kwargs):
        """
        Wrapper for self._gen_spikes. Need for dealing with cases without stimulus input.
        """
        p = self.params.copy() if params is None else params.copy()
        if 'ds' not in p:
            p['ds'] = 1
        return self._gen_spikes(self.theta['w'], self.theta['h'], self.theta['b'], self.theta['k'],
                                p['dt'], p['dh'], p['ds'], p['N'], p['M'], **kwargs)

    @staticmethod
    @numba.jit(nopython=True)
    def _gen_spikes(w, h, b, k, dt, dh, ds, N, M, seed=2, limit=20., stim_int=50):
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

        # nonlinearity; exp()
        f = np.exp

        r = np.zeros((N, M))  # rates
        y = np.zeros((N, M))  # spikes
        s = np.zeros((ds, M))
        sv= np.zeros((1, M))

        # the initial rate (no history); generate randomly
        init = np.random.randn(N) * 0.1 - 1
        r[:, 0] = f(init[0])
        y[:, 0] = np.array([np.random.poisson(r[i, 0]) for i in range(N)])

        # simulate the model for next M samples (time steps)
        for t in range(M):  # step through time

            stim_curr = (t // stim_int) % ds
            s[stim_curr, t] = 1.
            sv[0, t] = stim_curr

            for i in range(N):  # step through neurons
                # compute model firing rate
                if t == 0:
                    hist = 0
                elif t < dh:
                    hist = np.sum(h[i, :t] * y[i, :t])
                else:
                    hist = np.sum(h[i, :] * y[i, t - dh:t])

                if t == 0:
                    weights = 0
                else:
                    weights = np.dot(w[i, :], y[:, t - 1])

                if t == 0:
                    stim =np.zeros(3) 

                else:
                    stim = k[i, :]


                r[i, t] = f(b[i] + hist + weights + stim[0]*np.exp(-np.square(stim_curr*(np.pi/4)-stim[1])/(2*(stim[2]+0.001)**2)))

                # Clip. np.clip not supported in Numba.
                above = (r[i, t] >= limit) * limit
                below = (r[i, t] < limit)
                r[i, t] = r[i, t] * below + above

                y[i, t] = np.random.poisson(r[i, t] * dt)

        return r, y, s, sv

    def plot_theta(self, name):
        scale = np.max(np.abs(self.theta[name]))
        x = plt.imshow(self.theta[name], vmin=-scale, vmax=scale)
        x.set_cmap('bwr')
        plt.grid(False)
        plt.colorbar()
        plt.show()


if __name__ == '__main__':
    n = 10
    dh = 2
    ds = 8
    m = 200
    dt = 1.

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
        'base': 0.,
        'connectedness': 9,
        'max_w': 0.05
    }

    gen = DataGenerator(params=p, params_theta=params_θ)

    # %% Plot θ_w
    fig, ax = plt.subplots(dpi=100)
    plot_redblue(ax, gen.theta['w'], fig=fig)
    plt.show()
    print(np.sum(gen.theta['w'] != 0) / gen.theta['w'].size)

    # %% Save θ
    with open('theta_dict.pickle', 'wb') as f:
        pickle.dump(gen.theta, f)
    with open('params_dict.pickle', 'wb') as f:
        pickle.dump(p, f)
    print('Simulating model...')

    # %% Generate data
    r, y, s, sv = gen.gen_spikes(seed=0)

    print(gen.theta['b'])
    print(gen.theta['w'])
    print(gen.theta['h'])

    print('log_likelihood= '+str((np.sum(r)-np.sum(y*np.log(r)))/(m*n**2)))

    data = {'y': y.astype(np.uint8), 'r': r, 's':s}
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
    np.savetxt('stim_info_sample.txt', sv)

    fig, ax = plt.subplots(dpi=100)
    u = ax.imshow(r[:, :p['N'] * 4])
    ax.grid(0)
    fig.colorbar(u)
    ax.set_title('Synthetic data with stim')
    plt.show()
