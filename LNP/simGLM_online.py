import numpy as np
import scipy

import copy
import pickle
from pathlib import Path
from scipy.stats import poisson
from scipy.signal import fftconvolve

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def runModel(theta, data, params):
    '''
    Generates the output of the model given some theta
    Returns data dict like generateData()
    '''

    # constants
    dh = params['hist_dim']
    N  = params['numNeurons']

    # nonlinearity (exp)
    f = params['f']

    # hist = np.zeros((N,M))
    # y = np.concatenate((np.zeros((N,1)), data['y']), axis=1)
    # for i in np.arange(0,N):
    #     # compute model firing rate
    #     hist[i,:] = np.sum(fftconvolve(h[i,:], y, axes=1), axis=0)[:-2]

    expo = np.zeros((N,data['y'].shape[1]))
    # simulate the model for t samples (time steps)
    for j in np.arange(0,data['y'].shape[1]): 
        expo[:,j] = runModelStep(theta, data['y'][:,j-dh:j], params)
    
    # computed rates
    try:
        rates = f(expo)
    except:
        import pdb; pdb.set_trace()

    return rates

def runModelStep(theta, y, params):
    ''' Runs the model forward one point in time
        y should contain only up to t-dh:t points per neuron
    '''
    # constants
    N  = params['numNeurons']

    # model parameters
    w = theta['w']
    h = theta['h']
    b = theta['b']

    # data length in time
    t = y.shape[1] 

    expo = np.zeros(N)
    for i in np.arange(0,N): # step through neurons
        # compute model firing rate
        if t<1:
            hist = 0
        else:
            hist = np.sum(np.flip(h[i,:])*y[i,:])
            
        if t>0:
            weights = w[i,:].dot(y[:,-1])
        else:
            weights = 0
        
        expo[i] = (b[i] + hist + weights) #+ eps #remove log 0 errors
    
    return expo

def ll(theta_flat, data, params):
    '''
    log-likelihood objective and gradient
    '''

    # get parameters
    dh = params['hist_dim']
    dt = params['dt']
    N  = params['numNeurons']
    M  = params['numSamples']
    eps = np.finfo(float).eps

    # de-flatten theta
    theta = {}
    theta['w'] = theta_flat[:N*N].reshape((N,N))
    theta['h'] = theta_flat[N*N:N*(N+dh)].reshape((N,dh))
    theta['b'] = theta_flat[N*(N+dh):].reshape(N)

    # run model at theta
    rhat = runModel(theta, data, params)
    try:
        rhat = rhat*dt
    except FloatingPointError:
        print('FPE in rhat*dt; likely underflow')

    # compute negative log-likelihood
    # include l1 or l2 penalty on weights
    l2 = scipy.linalg.norm(theta['w']) #np.sqrt(np.sum(np.square(theta['w'])))/N
    l1 = np.sum(np.sum(np.abs(theta['w'])))

    try:
        ll_val = (np.sum(rhat) - np.sum(data['y']*np.log(rhat+eps)))/M  #+ l1
    except:
        import pdb; pdb.set_trace()
    print(ll_val)

    return ll_val

def ll_step(theta_flat, y, params):
    '''
    log-likelihood objective and gradient
    '''

    # get parameters
    dh = params['hist_dim']
    dt = params['dt']
    N  = params['numNeurons']
    M  = params['numSamples']
    eps = np.finfo(float).eps

    # de-flatten theta
    theta = {}
    theta['w'] = theta_flat[:N*N].reshape((N,N))
    theta['h'] = theta_flat[N*N:N*(N+dh)].reshape((N,dh))
    theta['b'] = theta_flat[N*(N+dh):].reshape(N)

    # run model at theta
    data['y'] = y
    rhat = runModel(theta, data, params)
    try:
        rhat = rhat*dt
    except FloatingPointError:
        print('FPE in rhat*dt; likely underflow')

    # compute negative log-likelihood
    # include l1 or l2 penalty on weights
    l2 = scipy.linalg.norm(theta['w']) #100*np.sqrt(np.sum(np.square(theta['w'])))
    l1 = np.sum(np.sum(np.abs(theta['w'])))/(N*N)

    # try:
    ll_val = ((np.sum(rhat) - np.sum(y*np.log(rhat+eps))) )/y.shape[1]/(N*N)
    # except:
    #     import pdb; pdb.set_trace()
    # print(ll_val)

    return ll_val

def ll_grad(theta_flat, y, params):
    # get parameters
    dh = params['hist_dim']
    dt = params['dt']
    N  = params['numNeurons']
    M  = y.shape[1] #params['numSamples']

    # de-flatten theta
    theta = {}
    theta['w'] = theta_flat[:N*N].reshape((N,N))
    theta['h'] = theta_flat[N*N:N*(N+dh)].reshape((N,dh))
    theta['b'] = theta_flat[-N:].reshape(N)

    # run model at theta
    data['y'] = y
    rhat = runModel(theta, data, params)
    rhat = rhat*dt

    # compute gradient
    grad = dict()

    # difference in computed rate vs. observed spike count
    rateDiff = (rhat - data['y'])

    # graident for baseline
    grad['b'] = np.sum(rateDiff, axis=1)/M

    # gradient for coupling terms
    yr = np.roll(data['y'], 1)
    #yr[0,:] = 0
    grad['w'] = rateDiff.dot(yr.T)/M + 10*d_abs(theta['w'])/(N*N)
    
    # gradient for history terms
    grad['h'] = np.zeros((N,dh))
    #grad['h'][:,0] = rateDiff[:,0].dot(data['y'][:,0].T)/M
    # for i in np.arange(0,N):
    for i in np.arange(0,N):
        for j in np.arange(0,dh):
            grad['h'][i,j] = np.sum(np.flip(data['y'],1)[i,:]*rateDiff[i,:])/M

    # import pdb; pdb.set_trace()

    # print('W....', grad['w'])
    # print('h....', grad['h'])
    # print('b....', grad['b'])

    # check for nans
    gradCheck(grad)
    # flatten grad
    grad_flat = flat_x(grad)/(N*N)

    # print('Mean grad ', np.mean(grad_flat))

    return grad_flat

def d_abs(weights):
    pos = (weights>=0)*1
    neg = (weights<0)*-1
    return pos+neg

def gradCheck(grad):
    for key in grad.keys():
        arrayCheck(grad[key], key)

def arrayCheck(arr, name):
    if ~np.isfinite(arr).all():
        print('**** WARNING: Found non-finite value in ' + name + ' (%g percent of the values were bad)'%(np.mean(np.isfinite(arr))))

def flat_x(theta):
    ''' Flatten dict (theta, also grad) into 1D vector for optimization (scipy)
    '''
    w = theta['w']
    h = theta['h']
    b = theta['b']
    combo = np.concatenate((w,h,b), axis=None)
    return combo.flatten()


if __name__ == "__main__":
    θ_gnd = pickle.loads(Path('theta_dict.pickle').read_bytes())
    p = pickle.loads(Path('params_dict.pickle').read_bytes())
    data = {'y': np.loadtxt('data_sample.txt'), 'r': np.loadtxt('rates_sample.txt')}
    N = p['numNeurons']

    print('Data Spike Counts: mean: ', np.mean(data['y']), 'var.: ', np.var(data['y']))
    print('Data Rates:  mean: ', np.mean(data['r']), 'var.: ', np.var(data['r']))


    rates_gnd = runModel(θ_gnd, data, p)
    print('Model Rates: mean: ', np.mean(rates_gnd), 'var.: ', np.var(rates_gnd))
    print('LL_step gnd:', ll_step(flat_x(θ_gnd), data['y'], p))

    θ_zero = {k: 0 * v for k, v in θ_gnd.items()}
    rates_zero = runModel(θ_zero, data, p)
    print('Model (zero) Rates: mean: ', np.mean(rates_zero), 'var.: ', np.var(rates_zero))
    print('LL_step zero:', ll_step(0*flat_x(θ_gnd), data['y'], p))

    θ_flat = flat_x(copy.deepcopy(θ_gnd))
    θ_flat = np.random.random(θ_flat.shape)
    print(ll_step(θ_flat, data['y'], p))

    plt.imshow(np.abs(θ_gnd['w']))
    plt.colorbar()
    plt.show()

    # res = scipy.optimize.minimize(ll_step, θ_flat, args=(data['y'][:,:],p), method='BFGS', options={'disp':True})

    # θ_rand['w'] = θ_flat[:N*N].reshape((N,N)) - θ_gnd['w']
    # wdiff = theta_orig['w'] - theta['w']
    # print('Norm diff: ', scipy.linalg.norm(wdiff))
    #
    #
    # # rates = runModel(theta, data, p)
    # # print('Rates:')
    # # print('mean: ', np.mean(rates))
    # # print('var.: ', np.var(rates))
    # # print('*** %g percent of rates are over the limit. ***'%(100*np.mean(rates>10)))
    # # t = time.time()
    # res = scipy.optimize.minimize(ll_step, θ_flat, args=(data['y'],p), method='BFGS', options={'disp':True})
    #plt.imshow(res['x'][:N * N].reshape((N, N)))
    # plt.show()
    # res = scipy.optimize.minimize(ll, theta_flat, args=(data,p), method='Nelder-Mead', options={'maxiter':1e6, 'disp':True})
    # # print('Quick computation: ', res.fun, ' time: ', time.time()-t)
    # # theta_flat = res.x
    #
    # # import pdb; pdb.set_trace()
    #
    # import time
    # y = data['y']
    # save_theta = np.zeros((460,9999))
    # save_ll = np.zeros((5,9999))
    # for i in np.arange(1,M):
    #     save_theta[:,i-1] = theta_flat
    #     # print('theta_flat mean: ', np.mean(theta_flat))
    #     # print(y[:,:i].shape)
    #     # print(ll_step(theta_flat, y[:,:i], p))
    #     if i<100:
    #         y_step = y[:,:i]
    #     else:
    #         y_step =  y[:,i-100:i]
    #     wdiff = theta_orig['w'] - theta_flat[:N*N].reshape((N,N))
    #     hdiff = theta_orig['h'] - theta_flat[N*N:N*(N+dh)].reshape((N,dh))
    #     bdiff = theta_orig['b'] - theta_flat[-N:]
    #     t0 = time.time()
    #     gradStep = ll_grad(theta_flat, y_step, p) #ll_grad(theta_flat, y_step, p) #scipy.optimize.approx_fprime(theta_flat, ll_step, 1e-2, y_step, p)
    #     theta_flat -= 2e-5*gradStep#*(i/100)
    #     save_ll[:,i-1] = np.array([time.time()-t0, scipy.linalg.norm(wdiff), scipy.linalg.norm(hdiff), scipy.linalg.norm(bdiff), ll_step(theta_flat, y_step, p)])
    #     print(time.time()-t0, scipy.linalg.norm(wdiff), scipy.linalg.norm(hdiff), scipy.linalg.norm(bdiff), ll_step(theta_flat, y_step, p)) #ll_step(theta_flat, y_step, p))
    #
    # np.savetxt('saved_theta_l1.txt', save_theta)
    # np.savetxt('saved_ll_l1.txt', save_ll)
    #
    # t = time.time()
    data_s = {'y': data['y'][:, :100]}
    res = scipy.optimize.minimize(ll_step, θ_flat, args=(data['y'],p), method='BFGS', options={'disp':True})
    res = scipy.optimize.minimize(ll, θ_flat, args=(data,p), method='Nelder-Mead', options={'maxiter':1e6, 'disp':True})
    # print('Quick computation: ', res.fun, ' time: ', time.time()-t)
    # # import pdb; pdb.set_trace()
    #
    # theta_flat = res.x
    #
    # theta = {}
    # N  = p['numNeurons']
    # dh = p['hist_dim']
    # theta['w'] = theta_flat[:N*N].reshape((N,N))
    # theta['h'] = theta_flat[N*N:N*(N+dh)].reshape((N,dh))
    # theta['b'] = theta_flat[-N:].reshape(N)
    # # # print('Eigenvectors...')
    # # # val, vect = np.linalg.eig(theta['w'])
    # # # idx = val.argsort()[::-1]
    # # # val = val[idx]
    # # # vect = vect[:,idx]
    # # # print(val)
    #
    # wdiff = theta_orig['w'] - theta['w']
    # print('Norm diff: ', scipy.linalg.norm(wdiff))

#%%
    # import networkx as nx
    #
    # G=nx.from_numpy_matrix(θ_gnd['w'])

