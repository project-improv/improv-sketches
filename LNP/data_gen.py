import pickle

import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
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
    np.random.seed(42)
    # get parameters
    dh = params['hist_dim']
    N = int(params['numNeurons'])
    M = int(params['numSamples'])
    alpha = params['alpha']

    # store model as dictionary
    theta = {}

    # pick out two pools for inhibitory and excitatory connections
    fracInh = 0.2
    numInh = int(np.ceil(fracInh * N))
    numExc = N - numInh
    inh = -0.05 * np.random.rand(N, numInh)
    exc = 0.5 * np.random.rand(N, numExc)

    # baseline rates
    theta['b'] = np.zeros((N))
    theta['b'][:numInh] = 0.1 * np.ones((1, numInh))
    theta['b'][numInh:] = -0.8 + np.random.rand(1, numExc)

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
    theta['h'][:numInh, :] = -0.02 * np.exp(-10 * tauInh).T  # mag (1.5e-3, 2e-1)
    theta['h'][numInh:, :] = 0.05 * np.exp(-5 * tauExc).T  # mag (7e-2, 5e-1)

    return theta


def generateData(theta, params):
    '''
    Generates spike counts from the model
    Returns a data dict:
    data['y']: spikes for each time step
    data['r']: firing rates (lambda) for each time step
    '''
    # constants
    dh = params['hist_dim']
    N = int(params['numNeurons'])
    M = int(params['numSamples'])

    # nonlinearity; exp()
    f = params['f']

    # model parameters
    w = theta['w']
    h = theta['h']
    b = theta['b']

    # store output in a dictionary
    data = {'y': np.zeros((N, M)), 'r': np.zeros((N, M))}  # spikes, rates

    # the initial rate (no history); generate randomly
    init = 0.1 * np.random.randn(M)
    data['y'][:, 0] = poisson.rvs(f(init[0]))
    data['r'][:, 0] = f(init[0])

    # simulate the model for next M samples (time steps)
    for j in np.arange(0, M):  # step through time
        for i in np.arange(0, N):  # step through neurons
            # compute model firing rate
            if j < 1:
                hist = 0
            elif j < dh:
                hist = np.sum(np.flip(h[i, :j]) * data['y'][i, :j])
            else:
                hist = np.sum(np.flip(h[i, :]) * data['y'][i, j - dh:j])

            if j > 0:
                weights = w[i, :].dot(data['y'][:, j - 1])
            else:
                weights = 0

            r = f(b[i] + hist + weights)

            # draw spikes
            data['r'][i, j] = r
            data['y'][i, j] = poisson.rvs(r)

    return data


if __name__ == '__main__':
    p = setParameters(n=20, dh=2, m=1e5, alpha=0.08)

    print('Generating model...')
    theta = generateModel(p)
    plt.imshow(np.abs(theta['w']))
    plt.colorbar()
    plt.show()

    with open('theta_dict.pickle', 'wb') as f:
        pickle.dump(theta, f)
    with open('params_dict.pickle', 'wb') as f:
        pickle.dump(p, f)
    print(theta['w'])
    print('Simulating model...')
    data = generateData(theta, p)
    print('Spike Counts:')
    print('mean: ', np.mean(data['y']))
    print('var.: ', np.var(data['y']), '\n')
    print('Rates:')
    print('mean: ', np.mean(data['r']))
    print('var.: ', np.var(data['r']))
    print('*** %g percent of rates are over the limit. ***' % (100 * np.mean(data['r'] > 10)))

    np.savetxt('data_sample.txt', data['y'])
    np.savetxt('rates_sample.txt', data['r'])
