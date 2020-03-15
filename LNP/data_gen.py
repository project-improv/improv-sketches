import pickle

from multiprocessing import Pool, set_start_method

set_start_method('fork')

import matplotlib.pyplot as plt
import numba
import numpy as np
import seaborn as sns

sns.set()


def setParameters(n=50, dh=10, m=1000, dt=0.1, alpha=0.05):
    '''
    n: number of neurons
    dh: dimension of the coupling filters
    m: sample size
    dt: time bin size
    alpha: scaling parameter for sparseness
    '''
    # parameters dictionary
    params = {'numNeurons': int(n), 'hist_dim': int(dh), 'numSamples': int(m), 'dt': dt, 'alpha': alpha}
    # nonlinearity
    params['f'] = np.exp
    return params


def generateModel(params):
    '''
    Generates model parameters.
    Returns a theta dictionary:
    theta['h']: history filters for the n neurons (dh x N)
    theta['w']: coupling filters for the n neurons (N x N) aka weight matrix
    theta['b']: baseline firing (N,)
    '''
    np.random.seed(2)
    # get parameters
    dh = params['hist_dim']
    N = int(params['numNeurons'])
    alpha = params['alpha']

    # store model as dictionary
    theta = {}

    # pick out two pools for inhibitory and excitatory connections
    fracInh = 0.2
    numInh = int(np.ceil(fracInh * N))
    numExc = N - numInh
    inh = -0.6 * np.random.rand(N, numInh)
    exc = 11 * np.random.rand(N, numExc)

    # baseline rates
    theta['b'] = np.zeros((N))
    theta['b'][:numInh] = 3.5 * np.ones((1, numInh))
    theta['b'][numInh:] = 1.5 + np.random.rand(1, numExc)

    # sparsity
    temp = np.random.rand(N, numExc)
    exc[temp >= alpha] = 0
    exc /= numExc
    temp2 = np.random.rand(N, numInh)
    inh[temp2 >= alpha * 3] = 0

    # coupling filters
    theta['w'] = np.zeros((N, N))
    wn = np.hstack((inh, exc)).T
    wn -= np.diag(np.diag(wn))
    theta['w'] = wn

    # history filter over time dh
    theta['h'] = np.zeros((N, dh))
    tau = np.linspace(1, 0, dh).reshape((dh, 1))
    tauInh = tau.dot(np.ones((1, numInh)))
    tauExc = tau.dot(np.ones((1, numExc)))
    theta['h'][:numInh, :] = -0.05 * np.exp(-4 * tauInh).T / p['dt']  # mag (1.5e-3, 2e-1)
    theta['h'][numInh:, :] = -0.05 * np.exp(-3 * tauExc).T / p['dt']  # mag (7e-2, 5e-1)

    return theta


@numba.jit(nopython=True)
def generateData(w, h, b, dt, dh, N, m):
    '''
    Generates spike counts from the model
    Returns a data dict:
    data['y']: spikes for each time step
    data['r']: firing rates (lambda) for each time step
    '''

    # nonlinearity; exp()
    f = np.exp  # params['f']

    # store output in a dictionary
    y = np.zeros((N, m))
    r = np.zeros((N, m))  # spikes, rates

    # the initial rate (no history); generate randomly
    init = 2 * np.random.randn(m)
    y[:, 0] = np.random.poisson(f(init[0]))
    r[:, 0] = f(init[0])

    # simulate the model for next M samples (time steps)
    for j in range(m):  # step through time
        for i in range(N):  # step through neurons
            # compute model firing rate
            if j < 1:
                hist = 0
            elif j < dh:
                hist = np.sum(h[i, :j] * y[i, :j])
            else:
                hist = np.sum(h[i, :] * y[i, j - dh:j])

            if j > 0:
                weights = w[i, :].dot(y[:, j - 1])
            else:
                weights = 0

            r[i, j] = f(b[i] + hist + weights)
            y[i, j] = np.random.poisson(r[i, j] * dt)

    return r, y


if __name__ == '__main__':
    p = setParameters(n=100, dh=10, m=2e5, alpha=0.05)

    print('Generating model...')
    theta = generateModel(p)
    scale = np.max(np.abs(theta['w']))
    x = plt.imshow(theta['w'], vmin=-scale, vmax=scale)
    x.set_cmap('bwr')
    plt.grid(False)
    plt.colorbar()
    plt.show()

    with open('theta_dict.pickle', 'wb') as f:
        pickle.dump(theta, f)
    with open('params_dict.pickle', 'wb') as f:
        pickle.dump(p, f)
    print(theta['w'])
    print('Simulating model...')

    cores = 8
    M = p['numSamples']
    M_core = [M // cores] * (cores - 1) + [M - M // cores * (cores - 1)]


    def proc(m):
        return generateData(theta['w'], np.flip(theta['h']), theta['b'], p['dt'], p['hist_dim'], p['numNeurons'], m)


    with Pool(cores) as pool:
        futures = list(pool.map(proc, M_core))

    r, y = np.hstack([r for (r, y) in futures]), np.hstack([y for (r, y) in futures])
    data = {'y': y, 'r': r}

    print('Spike Counts:')
    print('mean: ', np.mean(data['y']))
    print('var.: ', np.var(data['y']), '\n')
    print('Rates:')
    print('mean: ', np.mean(data['r']))
    print('var.: ', np.var(data['r']))
    print('*** %g percent of rates are over the limit. ***' % (100 * np.mean(data['r'] > 10)))
    print(np.percentile(data['y'], [50, 90, 99]))

    np.savetxt('data_sample.txt', data['y'])
    np.savetxt('rates_sample.txt', data['r'])
