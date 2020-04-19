import pickle
import random

import matplotlib.pyplot as plt
import networkx as nx
import numba
import numpy as np
import seaborn as sns

sns.set()


class DataGenerator:
    def __init__(self, params, params_θ=None, theta=None):
        self.params = params.copy()
        self.params_θ = None if params_θ is None else params_θ.copy()
        self.theta = theta
        if self.theta is None:
            self.theta = self._gen_theta(**self.params_θ)
        self.i = 0

    def gen_new_theta(self):
        if 'seed' not in self.params_θ:
            self.params_θ['seed'] = 1
        else:
            self.params_θ['seed'] += 1

        return self._gen_theta(**self.params_θ)

    def _gen_theta(self, seed=0, p_inh=0.5, base=0, connectedness=3, p_rand=0., rand_w=False, max_w=0.05, **kwargs):
        '''
        Generates model parameters.
        Returns a theta dictionary:
        theta['h']: history filters for the n neurons (dh x N)
        theta['w']: coupling filters for the n neurons (N x N) aka weight matrix
        theta['b']: baseline firing (N,)
        '''
        random.seed(seed)  # For networkx.
        np.random.seed(seed)

        # get parameters
        dh = self.params['dh']
        N = self.params['N']

        # store model as dictionary
        theta = dict()
        n_inh = int(N * p_inh)
        neutype = np.concatenate((-1 * np.ones(n_inh), np.ones(N-n_inh)))
        np.random.shuffle(neutype)

        # baseline rates
        theta['b'] = np.zeros(N)
        theta['b'][neutype == 1] = [base] * np.sum(neutype == 1)  # 2 * np.random.rand(np.sum(excinh == 1)) - 2
        theta['b'][neutype == -1] = [base] * np.sum(neutype == -1)  # 1 + np.random.rand(np.sum(excinh == -1))
        # theta['b'][5] = 1.

        # coupling filters
        theta['w'] =  max_w * np.random.random(size=(N, N)) if rand_w else max_w * np.ones((N,N))
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


        theta['k'] = self.gen_stim_w() if 'ds' in self.params else np.zeros((N, 1))

        return theta

    def gen_stim_w(self):
        r = 0.5
        sd = 1
        N = self.params['N']
        ds = self.params['ds']

        base = np.zeros((N, ds))
        center = ds // 2
        bell = np.array([r * np.exp(-((i - center) ** 2 / sd ** 2)) for i in range(ds)])
        chunk = N//ds
        for n in range(ds):
            base[n * chunk: np.clip((n+1) * chunk, a_min=0, a_max=N), :] = np.roll(bell, n - center)

        return base

    def gen_spikes(self, params=None, seed=2):
        p = self.params.copy() if params is None else params.copy()
        if 'ds' not in p:
            p['ds'] = 1
        return self._gen_spikes(self.theta['w'], self.theta['h'], self.theta['b'], self.theta['k'],
                                p['dt'], p['dh'], p['ds'], p['N'], p['M'], seed=seed)

    @staticmethod
    @numba.jit(nopython=True)
    def _gen_spikes(w, h, b, k, dt, dh, ds, N, m, seed=2, limit=100., stim_int=50):
        '''
        Generates spike counts from the model
        Returns a data dict:
        data['y']: spikes for each time step
        data['r']: firing rates (lambda) for each time step
        '''
        np.random.seed(seed)

        # nonlinearity; exp()
        f = np.exp

        r = np.zeros((N, m))  # rates
        y = np.zeros((N, m))  # spikes

        # the initial rate (no history); generate randomly
        init = np.random.randn(N) * 0.1 - 1
        r[:, 0] = f(init[0])
        y[:, 0] = np.array([np.random.poisson(r[i, 0]) for i in range(N)])

        # simulate the model for next M samples (time steps)
        for t in range(m):  # step through time
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
                    weights = w[i, :].dot(y[:, t - 1])

                if t == 0:
                    stim = 0
                else:
                    stim = k[i, (t // stim_int) % ds]

                # Clip. np.clip not supported in Numba.
                r[i, t] = f(b[i] + hist + weights + stim)
                above = (r[i, t] >= limit) * limit
                below = (r[i, t] < limit)
                r[i, t] = r[i, t] * below + above
                y[i, t] = np.random.poisson(r[i, t] * dt)

        return r, y


    def plot_theta_w(self):
        scale = np.max(np.abs(self.theta['w']))
        x = plt.imshow(self.theta['w'], vmin=-scale, vmax=scale)
        x.set_cmap('bwr')
        plt.grid(False)
        plt.colorbar()
        plt.show()


if __name__ == '__main__':

    n = 80
    dh = 2
    ds = 8
    m = 2000
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

    gen = DataGenerator(params=p, params_θ=params_θ)

    #%% Plot θ_w
    gen.plot_theta_w()
    print(np.sum(gen.theta['w'] != 0) / gen.theta['w'].size)

    #%% Save θ
    with open('theta_dict.pickle', 'wb') as f:
        pickle.dump(gen.theta, f)
    with open('params_dict.pickle', 'wb') as f:
        pickle.dump(p, f)
    print('Simulating model...')

    #%% Generate data
    r, y = gen.gen_spikes(seed=0)
    data = {'y': y.astype(np.uint8), 'r': r}
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
    # np.savetxt('rates_sample.txt', data['r'])

    plt.imshow(r[:, :p['N']*4])
    plt.colorbar()
    plt.show()
