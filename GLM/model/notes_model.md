# Model Code

### `glm_jax.py`

Implements the actual model in `jax`. Designed to accommodate increases in `N` and `M` by padding input matrices to `N_lim` and `M_lim` (this is due to the fact that compiled functions require a constant matrix size).

Automatically increases `N_lim` and `M_lim` by 2-fold if limit is reached.

L1 and L2 regularizations are implemented in the `ll` function with parameters `λ1` and `λ2`.


### `compare_opt.py`
A high-level wrapper of `glm_jax`. Take in a set of data and fit using the `fit` function. Primary use is to compare different optimizers using the same dataset.