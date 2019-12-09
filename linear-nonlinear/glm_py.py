import pickle
import time
from pathlib import Path

import numpy as np


class GLMPy:
    # TODO: Add additional error handling
    # def __init__(self, theta, p, ):
    #     self.p = p
    #     self.theta = theta

    def setup(self, param_file=None):
        '''
        '''
        np.seterr(divide='ignore')

        # TODO: same as behaviorAcquisition, need number of stimuli here. Make adaptive later
        self.num_stim = 21
        self.frame = 0
        # self.curr_stim = 0 #start with zeroth stim unless signaled otherwise
        self.stim = {}
        self.stimStart = {}
        self.window = 500  # TODO: make user input, choose scrolling window for Visual
        self.C = None
        self.S = None
        self.Call = None
        self.Cx = None
        self.Cpop = None
        self.coords = None
        self.color = None
        self.runMean = None
        self.runMeanOn = None
        self.runMeanOff = None
        self.lastOnOff = None
        self.recentStim = [0] * self.window
        self.currStimID = np.zeros((8, 100000))  # FIXME
        self.currStim = -10
        self.allStims = {}

        self.total_times = []
        self.puttime = []
        self.colortime = []
        self.stimtime = []
        self.timestamp = []
        self.LL = []

        w = np.zeros((2, 2))  # guess 2 neurons initially?
        h = np.zeros((2, 2))  # dh is 2
        k = np.zeros((2, 8))
        b = np.zeros(2)
        self.theta = np.concatenate((w, h, b, k), axis=None).flatten()
        self.p = {'numNeurons': 2, 'hist_dim': 2, 'numSamples': 1, 'dt': 0.1, 'stim_dim': 8}  # TODO: from config file..

        data = np.zeros((2, 10))

    def fit(self):
        '''
        '''
        if self.p["numNeurons"] < self.S.shape[0]:  # check for more neurons
            self.updateTheta()

        self.p["numSamples"] = self.frame

        # print('First test: ', self.ll(self.S[:,:self.frame]))

        if self.frame < 100:
            y_step = self.S[:, :self.frame]
            stim_step = self.currStimID[:, :self.frame]
        else:
            y_step = self.S[:, self.frame - 100:self.frame]
            stim_step = self.currStimID[:, self.frame - 100:self.frame]

        y_step = np.where(np.isnan(y_step), 0, y_step)  # Why are there nans here anyway?? #TODO
        ll = self.ll(y_step, stim_step)
        # t0 = time.time()
        self.theta -= 1e-5 * self.ll_grad(y_step, stim_step) * (self.frame / 100)
        # self.LL.append(self.ll(y_step, stim_step))

        return ll

        # gradStep = self.j_ll_grad(self.theta, y_step, self.p)
        # self.theta -= 1e-5*gradStep
        # self.theta -= 1e-5 * self.j_ll_grad(self.theta, y_step, self.p)

    def ll(self, y, s):
        '''
        log-likelihood objective and gradient
        '''
        # get parameters
        dh = self.p['hist_dim']
        ds = self.p['stim_dim']
        dt = self.p['dt']
        N = self.p['numNeurons']
        # M  = self.p['numSamples']
        eps = np.finfo(float).eps

        # run model at theta
        data = {}
        data['y'] = y
        data['s'] = s
        rhat = self.runModel(data)
        try:
            rhat = rhat * dt
        except FloatingPointError:
            print('FPE in rhat*dt; likely underflow')

        # model parameters
        # h = self.theta[N*N:N*(N+dh)].reshape((N,dh))
        # b = self.theta[N*(N+dh):].reshape(N)

        # compute negative log-likelihood
        # include l1 or l2 penalty on weights

        ll_val = ((np.sum(rhat) - np.sum(y * np.log(rhat + eps)))) / y.shape[1] / N  # + l1

        return ll_val

    def ll_grad(self, y, s):
        # get parameters
        dh = self.p['hist_dim']
        dt = self.p['dt']
        N = self.p['numNeurons']
        M = y.shape[1]  # params['numSamples'] #TODO: should be equal

        # run model at theta
        data = {}
        data['y'] = y
        data['s'] = s
        rhat = self.runModel(data)
        rhat = rhat * dt

        # compute gradient
        grad = dict()

        # difference in computed rate vs. observed spike count
        rateDiff = (rhat - data['y'])

        # graident for baseline
        grad['b'] = np.sum(rateDiff, axis=1) / M

        # gradient for stim
        grad['k'] = rateDiff.dot(data['s'].T) / M

        # gradient for coupling terms
        yr = np.roll(data['y'], 1)
        # yr[0,:] = 0
        grad['w'] = rateDiff.dot(yr.T) / M  # + d_abs(theta['w'])

        # gradient for history terms
        grad['h'] = np.zeros((N, dh))
        # grad['h'][:,0] = rateDiff[:,0].dot(data['y'][:,0].T)/M
        for i in np.arange(0, N):
            for j in np.arange(0, dh):
                grad['h'][i, j] = np.sum(np.flip(data['y'], 1)[i, :] * rateDiff[i, :]) / M

        # check for nans
        grad = self.gradCheck(grad)

        # flatten grad
        grad_flat = np.concatenate((grad['w'], grad['h'], grad['b'], grad['k']), axis=None).flatten() / N

        return grad_flat

    def gradCheck(self, grad):
        resgrad = {}
        for key in grad.keys():
            resgrad[key] = self.arrayCheck(grad[key], key)
        return resgrad

    def arrayCheck(self, arr, name):
        if ~np.isfinite(arr).all():
            print('**** WARNING: Found non-finite value in ' + name + ' (%g percent of the values were bad)' % (
                np.mean(np.isfinite(arr))))
        arr = np.where(np.isnan(arr), 0, arr)
        arr[arr == np.inf] = 0
        return arr

    def runModel(self, data):
        '''
        Generates the output of the model given some theta
        Returns data dict like generateData()
        '''

        # constants
        dh = self.p['hist_dim']
        N = self.p['numNeurons']

        # nonlinearity (exp)
        f = np.exp

        expo = np.zeros((N, data['y'].shape[1]))
        # simulate the model for t samples (time steps)
        for j in np.arange(0, data['y'].shape[1]):
            expo[:, j] = self.runModelStep(data['y'][:, j - dh:j], data['s'][:, j])

        # computed rates
        try:
            rates = f(expo)
        except:
            import pdb;
            pdb.set_trace()

        return rates

    def runModelStep(self, y, s):
        ''' Runs the model forward one point in time
            y should contain only up to t-dh:t points per neuron
        '''
        # constants
        N = self.p['numNeurons']
        dh = self.p['hist_dim']
        ds = self.p['stim_dim']

        # model parameters
        w = self.theta[:N * N].reshape((N, N))
        h = self.theta[N * N:N * (N + dh)].reshape((N, dh))
        b = self.theta[N * (N + dh):N * (N + dh + 1)].reshape(N)
        k = self.theta[N * (N + dh + 1):].reshape((N, ds))

        # data length in time
        t = y.shape[1]

        expo = np.zeros(N)
        for i in np.arange(0, N):  # step through neurons
            # compute model firing rate
            if t < 1:
                hist = 0
            else:
                hist = np.sum(np.flip(h[i, :]) * y[i, :])

            if t > 0:
                weights = w[i, :].dot(y[:, -1])
            else:
                weights = 0

            stim = k[i, :].dot(s)

            expo[i] = (b[i] + hist + weights + stim)  # + eps #remove log 0 errors

        return expo

    def updateTheta(self):
        ''' TODO: Currently terribly inefficient growth
            Probably initialize large and index into it however many N we have
        '''
        N = self.p['numNeurons']
        dh = self.p['hist_dim']
        ds = self.p['stim_dim']

        old_w = self.theta[:N * N].reshape((N, N))
        old_h = self.theta[N * N:N * (N + dh)].reshape((N, dh))
        old_b = self.theta[N * (N + dh):N * (N + dh + 1)].reshape(N)
        old_k = self.theta[N * (N + dh + 1):].reshape((N, ds))

        self.p["numNeurons"] = self.S.shape[0]  # confirm this
        M = self.p['numNeurons']

        w = np.zeros((M, M))
        w[:N, :N] = old_w
        h = np.zeros((M, dh))
        h[:N, :] = old_h
        b = np.zeros(M)
        b[:N] = old_b
        k = np.zeros((M, ds))
        k[:N, :] = old_k

        self.theta = np.concatenate((w, h, b, k), axis=None).flatten()
