import numpy as np
from jax.numpy import sqrt

def gen_rand_theta(p, seed=1234):
    """
    Random initialization of θ.
    """
    np.random.seed(seed)
    w = 1 / 20 * np.random.random((p['N_lim'], p['N_lim'])) # 1 / 20 * np.random.random((p['N_lim'], p['N_lim']))
    k = np.zeros((p['N_lim'], p['ds']))
    b = 1 / 10 * np.random.random((p['N_lim'], 1))
    h = 1 / 20 * np.random.random((p['N_lim'], p['dh']))
    return {'h': h, 'w': w, 'b': b, 'k': k}

def online_sqrt_decay(step_size, window_size):
    def schedule(i):
        return (i <= window_size) * i / window_size * step_size \
               + (i > window_size) * step_size / sqrt(((i <= window_size) * window_size*2) + (i - window_size))
    return schedule

def online_exp_decay(step_size, window_size, decay_steps, decay_rate):
    def schedule(i):
        return (i <= window_size) * i / window_size * step_size \
               + (i > window_size) * decay_rate ** ((i - window_size) / decay_steps)
    return schedule

def offline_decay(step_size, decay_steps, decay_rate):
    def schedule(i):
        return step_size * decay_rate ** (i / decay_steps)
    return schedule
