# Poisson GLM

This package is a JAX implementation of a Poisson generalized linear model (GLM) (see [Pillow _et al._ (2008)](https://www.nature.com/articles/nature07140)) used to fit larval zebrafish two-photon calcium fluorescence data from CaImAn.


## Parameters

The hyperparameters are
* ![N](https://render.githubusercontent.com/render/math?math=N): Number of neurons.
* ![M](https://render.githubusercontent.com/render/math?math=M): Number of time steps in the input.
* ![dh](https://render.githubusercontent.com/render/math?math=dh): Number of time steps to consider for convolution.
* ![ds](https://render.githubusercontent.com/render/math?math=ds): Number of stimuli
* ![dt](https://render.githubusercontent.com/render/math?math=dt): Time per step (seconds).


### Model

Let ![h(y)](https://render.githubusercontent.com/render/math?math=h(y)) be a sliding window convolution between ![\theta_h](https://render.githubusercontent.com/render/math?math=%5Ctheta_h) and ![y](https://render.githubusercontent.com/render/math?math=y).

![h(y) = \theta_h * y(:,i).](https://render.githubusercontent.com/render/math?math=h(y)%20%3D%20%5Ctheta_h%20*%20y(%3A%2Ci).)

Then, the predicted activity ![\hat{r}](https://render.githubusercontent.com/render/math?math=%5Chat%7Br%7D) (a vector of length ![N](https://render.githubusercontent.com/render/math?math=N)) for the next time step is

![\hat{r}(y, s) = dt \times \exp\left\{\theta_ks + h(y) + \theta_wy + \theta_b \right\}](https://render.githubusercontent.com/render/math?math=%5Chat%7Br%7D(y%2C%20s)%20%3D%20dt%20%5Ctimes%20%5Cexp%5Cleft%5C%7B%5Ctheta_ks%20%2B%20h(y)%20%2B%20%5Ctheta_wy%20%2B%20%5Ctheta_b%20%5Cright%5C%7D)

<img width="260" alt="image" src="https://user-images.githubusercontent.com/34997334/83041934-05c56e80-a00f-11ea-83b6-b8259f90eb23.png">

The log-likelihood is

![\ell(y, \hat{r}) = \frac{1}{MN} \left( \hat{r} - \sum_{i=0}^N y_i \odot \log(\hat{r}) \right).](https://render.githubusercontent.com/render/math?math=%5Cell(y%2C%20%5Chat%7Br%7D)%20%3D%20%5Cfrac%7B1%7D%7BMN%7D%20%5Cleft(%20%5Chat%7Br%7D%20-%20%5Csum_%7Bi%3D0%7D%5EN%20y_i%20%5Codot%20%5Clog(%5Chat%7Br%7D)%20%5Cright).)


### Implementation

The model is coded for the just-in-time (JIT) compiler of JAX for speed and automatic differentiation.

The compiler needs to know the exact dimensions of all matrices that are passed into each function. Direct substitutions are not allowed as it disrupts the model differentiability.

Any changes in input dimensions will lead to a recompilation which destroys all speed benefit. Hence, the hyperparameter ![M](https://render.githubusercontent.com/render/math?math=M) or ![M_{\text{lim}}](https://render.githubusercontent.com/render/math?math=M_%7B%5Ctext%7Blim%7D%7D) is assumed to be fixed. As the number of detected neurons do increase through the run, ![N_{\text{lim}}](https://render.githubusercontent.com/render/math?math=N_%7B%5Ctext%7Blim%7D%7D) will automatically increase by 2-fold when the dimensions of ![y](https://render.githubusercontent.com/render/math?math=y) and ![s](https://render.githubusercontent.com/render/math?math=s) exceed ![N_{\text{lim}}](https://render.githubusercontent.com/render/math?math=N_%7B%5Ctext%7Blim%7D%7D). Extra space in the weight matrices ![\theta](https://render.githubusercontent.com/render/math?math=%5Ctheta) are padded with 0.

### Organization

* `eva`: Data fitting from `improv Data Sharing` folder in Box.
* `model`: Low-level implementations.
* `synthetic`: Synthetic data fit and benchmarks.


# Installation
This package was tested on Ubuntu 20.04 LTS (Focal Fossa).

Do *NOT* install `CaImAn` via `conda`. Tensorflow now defaults to v2.x and the developers have not addressed that yet (05/27/20).

To create the environment, run

```shell script
conda env create -n caiman-glm -f environment.yml
```

`JAX` must be installed afterwards (distinct requirements for each OS and CUDA version). See [JAX GitHub](https://github.com/google/jax#installation).

TL;DR Run
```shell script
pip install --upgrade https://storage.googleapis.com/jax-releases/`nvidia-smi | sed -En "s/.* CUDA Version: ([0-9]*)\.([0-9]*).*/cuda\1\2/p"`/jaxlib-0.1.47-`python3 -V | sed -En "s/Python ([0-9]*)\.([0-9]*).*/cp\1\2/p"`-none-linux_x86_64.whl jax
```

# Run

These [environmental variables](https://caiman.readthedocs.io/en/master/Installation.html?highlight=export#setting-up-environment-variables) must be set in every session that `CaImAn` runs.

```shell script
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
```
