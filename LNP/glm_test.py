import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from jax.experimental import optimizers

from glm_jax import GLMJax
from glm_py import GLMPy

sns.set()


def online_sch(i_frame):
    """ Learning rate schedule for JAX. Linear increase over frame number. """
    return 1e-5 * i_frame / 100


def run_py(n):
    """
    Run the python version of the LNL model with a window size of M.
    """
    m = GLMPy()
    m.setup()
    ll, t = np.zeros(n + 1), np.zeros(n + 1)

    for i in range(1, n + 1):
        start = time.time()
        if i < M:
            curr_S, curr_stim = S[:, :i], stim[:, :i]
        else:
            curr_S, curr_stim = S[:, i - M:i], stim[:, i - M:i]

        m.S = curr_S
        m.frame = i
        ll[i] = m.fit()
        t[i] = time.time() - start
        print(f'Py Iter: {i}')
    return ll[1:], t[1:]


def run_jax(opt, n):
    m = GLMJax(p, optimizer=opt)
    ll, t = np.zeros(n + 1), np.zeros(n + 1)

    for i in range(1, n + 1):
        start = time.time()
        if i < M:
            curr_S, curr_stim = S[:, :i], stim[:, :i]
        else:
            curr_S, curr_stim = S[:, i - M:i], stim[:, i - M:i]

        ll[i] = m.ll(curr_S, curr_stim)
        m.fit(curr_S, curr_stim)
        t[i] = time.time() - start
        print(f'JAX Iter: {i}')

    return ll[1:], t[1:]


if __name__ == '__main__':

    S = pickle.loads(Path('cnmf_s.pk').read_bytes())  # See caiman.estimates.S
    T = 50  # Length of test

    # Run python
    llt_py = run_py(T)

    N = 2
    M = 100
    dh = 2
    ds = 8
    stim = np.zeros((ds, T))

    p = {'dh': dh, 'ds': ds, 'dt': 0.1, 'n': 0, 'N_lim': N, 'M_lim': M}

    # Compare optimizers
    opts = {
        'SGD': optimizers.sgd(online_sch),
        'Adam': optimizers.adam(online_sch)
    }

    llt_jax = {name: run_jax(opt, T) for name, opt in opts.items()}

    # Plots
    fig = plt.figure(dpi=300)
    ax = plt.subplot(1, 1, 1)

    ax.plot(llt_py[0], '--', label='Python')
    for name, llt in llt_jax.items():
        ax.plot(llt[0], label=name)

    ax.legend()
    ax.set_xlabel('Time step')
    ax.set_ylabel('LL')
    ax.set_title('LL from Rainbow Dataset (N=1496)')
    plt.show()

    fig = plt.figure(dpi=300)
    ax = plt.subplot(1, 1, 1)

    ax.plot(llt_py[1], label='Python')
    for name, llt in llt_jax.items():
        ax.plot(llt[1], label=name)

    ax.legend()
    ax.set_xlabel('Time step')
    ax.set_ylabel('t(s)')
    ax.set_yscale('log')
    ax.set_title('Time from Rainbow Dataset (N=1496)')

    plt.show()
