import numpy as np
import scipy
from scipy.stats import poisson
from scipy.signal import fftconvolve

def setParameters(n = 50, dh = 10, m = 1000, dt = 0.1, alpha=0.05):
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

    # get parameters
    dh = params['hist_dim']
    N  = int(params['numNeurons'])
    M  = int(params['numSamples'])
    alpha = params['alpha']

    # store model as dictionary
    theta = {}

    # pick out two pools for inhibitory and excitatory connections
    fracInh = 0.2
    numInh = int(np.ceil(fracInh*N))
    numExc = N - numInh
    inh = -0.05*np.random.rand(N, numInh)
    exc = 0.5*np.random.rand(N, numExc)

    # baseline rates
    theta['b'] = np.zeros((N))
    theta['b'][:numInh] = 0.1*np.ones((1,numInh))
    theta['b'][numInh:] = -0.8 + np.random.rand(1,numExc)

    #sparsity
    temp = np.random.rand(N, numExc)
    exc[temp>=alpha] = 0
    exc /= numExc
    temp2 = np.random.rand(N, numInh)
    inh[temp2>=alpha*3] = 0

    # coupling filters
    theta['w'] = np.zeros((N,N))
    wn = np.hstack((inh, exc)).T
    wn -= np.diag(np.diag(wn))
    theta['w'] = wn

    # history filter over time dh
    theta['h'] = np.zeros((N,dh))
    tau = np.linspace(1,0,dh).reshape((dh,1))
    tauInh = tau.dot(np.ones((1, numInh)))
    tauExc = tau.dot(np.ones((1, numExc)))
    theta['h'][:numInh, :] = -0.02*np.exp(-10*tauInh).T # mag (1.5e-3, 2e-1)
    theta['h'][numInh:, :] = 0.05*np.exp(-5*tauExc).T # mag (7e-2, 5e-1) 

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
    N  = int(params['numNeurons'])
    M  = int(params['numSamples'])
    eps = np.finfo(float).eps

    # nonlinearity; exp()
    f = params['f']

    # model parameters
    w = theta['w']
    h = theta['h']
    b = theta['b']

    # store output in a dictionary
    data = {}
    data['y'] = np.zeros((N, M)) # spikes
    data['r'] = np.zeros((N, M)) # rates

    # the initial rate (no history); generate randomly
    init = 0.1*np.random.randn(M) 
    data['y'][:,0] = poisson.rvs(f(init[0]))
    data['r'][:,0] = f(init[0])

    print(h)
    print(w)

    # simulate the model for next M samples (time steps)
    for j in np.arange(0,M): # step through time
        for i in np.arange(0,N): # step through neurons
            # compute model firing rate
            if j<1:
                hist = 0
            elif j<dh:
                hist = np.sum(np.flip(h[i,:j])*data['y'][i,:j])
            else:
                hist = np.sum(np.flip(h[i,:])*data['y'][i,j-dh:j])

            if j>0:
                weights = w[i,:].dot(data['y'][:,j-1])
            else:
                weights = 0
            
            r = f(b[i] + hist + weights) #+ eps #remove log 0 errors

            # cap rates
            maxVal = 100
            # print(np.count_nonzero(r>maxVal))
            if r>maxVal: 
                print('capping!')
                import pdb; pdb.set_trace()
                r = maxVal

            # draw spikes
            data['r'][i,j] = r
            data['y'][i,j] = poisson.rvs(r)

        # print('Rates time ', j, data['r'][:,j].T)
        # print('Spikes time ', j, data['y'][:,j].T)

    return data

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
    # try:
    rates = f(expo)
    # except:
    #     import pdb; pdb.set_trace()

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
    ll_val = ((np.sum(rhat) - np.sum(y*np.log(rhat+eps))) )/y.shape[1]/(N*N) + 10*l1
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

if __name__=="__main__":
    #
    # import julia
    # julia = julia.Julia(compiled_modules=False)
    # julia.include('sim_GLM.jl')
    # j_ll_grad = julia.eval('pyfunction(simGLM.ll_grad,PyArray,PyArray,PyDict)')
    # j_ll = julia.eval('pyfunction(simGLM.ll,PyArray,PyArray,PyDict)')

    np.seterr(all='raise')
    np.random.seed(0) #15531511)
    eps = np.finfo(float).eps

    print('Initializing parameters...')
    p = setParameters(n = 10, dh = 2, m = 1e4, alpha=0.02)

    print('Generating model...')
    theta = generateModel(p)
    import pickle
    with open('theta_dict.pickle', 'wb') as f:
        pickle.dump(theta, f)
    with open('params_dict.pickle', 'wb') as f:
        pickle.dump(p, f)

    print('Simulating model...')
    data = generateData(theta, p)
    print('Spike Counts:')
    print('mean: ', np.mean(data['y']))
    print('var.: ', np.var(data['y']), '\n')
    print('Rates:')
    print('mean: ', np.mean(data['r']))
    print('var.: ', np.var(data['r']))
    print('*** %g percent of rates are over the limit. ***'%(100*np.mean(data['r']>10)))

    np.savetxt('data_sample.txt', data['y']) 
    np.savetxt('theta_orig.txt', theta['w'])

    rates = runModel(theta, data, p)
    print('Rates:')
    print('mean: ', np.mean(rates))
    print('var.: ', np.var(rates))
    print('*** %g percent of rates are over the limit. ***'%(100*np.mean(rates>10)))

    N  = p['numNeurons']
    dh = p['hist_dim']
    M  = p['numSamples']
    theta_flat = flat_x(theta)
    # ll_grad(theta_flat, data, p)
    # print('check grad...', scipy.optimize.check_grad(ll_step, ll_grad, theta_flat, data['y'], p))
    # print('apporxing ...')
    # result = scipy.optimize.approx_fprime(theta_flat, ll, 1e-4, data, p)
    # print('w', result[:N*N])
    # print('h', result[N*N:N*(N+dh)].reshape((N,dh)))
    # print('b', result[-N:])

    # print('Eigenvectors...')
    # val, vect = np.linalg.eig(theta['w'])
    # idx = val.argsort()[::-1]   
    # val = val[idx]
    # vect = vect[:,idx]
    # print(val)

    from matplotlib.pylab import *

    # figure(1)
    # clf()
    # imshow(theta['w'])
    # colorbar()
    # draw()

    # figure(2)
    # clf()
    # imshow(theta['h'])
    # colorbar()
    # draw()

    # figure(3)
    # clf()
    # plot(theta['b'])
    # draw()

    # figure(4)
    # clf()
    # plot(data['y'][:,-100:])
    # draw()

    # figure(5)
    # clf()
    # imshow(np.log(1+ data['r'][:,-100:]))
    # colorbar()
    # draw()

    # # figure(4)
    # # clf()
    # # imshow(np.log(1+ rates[:,-100:]))
    # # colorbar()
    # # draw()

    # show(block=True)

    import copy
    theta_orig = copy.deepcopy(theta)

    theta_flat = flat_x(theta)
    np.random.seed(0)
    theta_flat = np.random.random(theta_flat.shape)# zeros(theta_flat.shape) # random.random(theta_flat.shape)
    # print(ll_step(theta_flat, data['y'], p))
    # print(j_ll(theta_flat, data['y'], p))
    theta['w'] = theta_flat[:N*N].reshape((N,N))
    wdiff = theta_orig['w'] - theta['w']
    print('Norm diff: ', scipy.linalg.norm(wdiff))

    # from julia import Main
    # Main.data = data['y']
    # print('testing ... ', j_ll_grad(theta_flat, data['y'], p))

    # rates = runModel(theta, data, p)
    # print('Rates:')
    # print('mean: ', np.mean(rates))
    # print('var.: ', np.var(rates))
    # print('*** %g percent of rates are over the limit. ***'%(100*np.mean(rates>10)))
    # t = time.time()
    # res = scipy.optimize.minimize(ll_step, theta_flat, args=(data['y'],p), method='BFGS', options={'disp':True})
    # # res = scipy.optimize.minimize(ll, theta_flat, args=(data,p), method='Nelder-Mead', options={'maxiter':1e6, 'disp':True})
    # print('Quick computation: ', res.fun, ' time: ', time.time()-t)
    # theta_flat = res.x
    
    # import pdb; pdb.set_trace()

    import time
    y = data['y']
    total_size = theta['w'].size + theta['h'].size + theta['b'].size
    save_theta = np.zeros((total_size,9999))
    save_ll = np.zeros((5,9999))
    for i in np.arange(1,M):
        save_theta[:,i-1] = theta_flat
        # print('theta_flat mean: ', np.mean(theta_flat))
        # print(y[:,:i].shape)
        # print(ll_step(theta_flat, y[:,:i], p))
        if i<100:
            y_step = y[:,:i]
        else:
            y_step =  y[:,i-100:i]
        wdiff = theta_orig['w'] - theta_flat[:N*N].reshape((N,N))
        hdiff = theta_orig['h'] - theta_flat[N*N:N*(N+dh)].reshape((N,dh))
        bdiff = theta_orig['b'] - theta_flat[-N:]
        t0 = time.time()
        gradStep = ll_grad(theta_flat, y_step, p) #ll_grad(theta_flat, y_step, p) #scipy.optimize.approx_fprime(theta_flat, ll_step, 1e-2, y_step, p)
        theta_flat -= 2e-5*gradStep#*(i/100)
        save_ll[:,i-1] = np.array([time.time()-t0, scipy.linalg.norm(wdiff), scipy.linalg.norm(hdiff), scipy.linalg.norm(bdiff), ll_step(theta_flat, y_step, p)])
        # print(time.time()-t0, scipy.linalg.norm(wdiff), scipy.linalg.norm(hdiff), scipy.linalg.norm(bdiff), ll_step(theta_flat, y_step, p)) #ll_step(theta_flat, y_step, p))
        if i % 100 == 0:
            print(f'Step {i}')

    np.savetxt('saved_theta_l1.txt', save_theta)
    np.savetxt('saved_ll_l1.txt', save_ll)

    t = time.time()
    # res = scipy.optimize.minimize(ll_step, theta_flat, args=(data['y'],p), method='BFGS', options={'disp':True})
    # res = scipy.optimize.minimize(ll, theta_flat, args=(data,p), method='Nelder-Mead', options={'maxiter':1e6, 'disp':True})
    # print('Quick computation: ', res.fun, ' time: ', time.time()-t)
    # import pdb; pdb.set_trace()

    # theta_flat = res.x

    theta = {}
    N  = p['numNeurons']
    dh = p['hist_dim']
    theta['w'] = theta_flat[:N*N].reshape((N,N))
    theta['h'] = theta_flat[N*N:N*(N+dh)].reshape((N,dh))
    theta['b'] = theta_flat[-N:].reshape(N)
    # # print('Eigenvectors...')
    # # val, vect = np.linalg.eig(theta['w'])
    # # idx = val.argsort()[::-1]   
    # # val = val[idx]
    # # vect = vect[:,idx]
    # # print(val)

    wdiff = theta_orig['w'] - theta['w']
    print('Norm diff: ', scipy.linalg.norm(wdiff))

    # import pdb; pdb.set_trace()

    # figure(6)
    # clf()
    # imshow(theta['w'])
    # colorbar()
    # draw()

    # figure(7)
    # clf()
    # imshow(theta['h'])
    # colorbar()
    # draw()

    # figure(8)
    # clf()
    # plot(theta['b'])
    # draw()
    
    # figure(9)
    # clf()
    # imshow(theta_orig['w'] - theta['w'])
    # colorbar()
    # draw()

    # figure(10)
    # clf()
    # imshow(theta_orig['h'] - theta['h'])
    # colorbar()
    # draw()

    # figure(11)
    # clf()
    # plot(theta_orig['b'] - theta['b'])
    # draw()

    # figure(7)
    # clf()
    # imshow(np.log(1 + data['y'][-100:]))
    # colorbar()
    # draw()

    # figure(8)
    # clf()
    # imshow(np.log(data['r'][-100:]))
    # colorbar()
    # draw()

    # show(block=True)