# improv-sketches

Private repository for projects using improv as a framework for streaming analysis and adaptive experimental methods.

# Overview

In this branch, we implement the conductance based encoding model outlined in 'Inferring synaptic inputs from spikes with a conductance-based neural encoding model' (Latimer et al. 2019) to be fit in an online fashion using JAX. The python file for synthetic data generation can be found in the GLM/synthetic folder as data_gen_CBEM.py and the python file for model fitting can be found in the GLM/model folder as CBEM_online.py. Within both of these files there are functions and code for plotting and saving relevant information and metrics. Currently, the model has only been tested on synthetic data produced by the same model. 

# Data generation

Within the data generation file, the user can set the number of neurons, the number of time steps, and the size of each time step. The methods of generating the filters can also be changed within the functions of the DataGenerator class.

# Model fitting

Within the model fitting file, the user can similarly set the number of neurons, number of time steps, and the size of each time step, as well as the size of the window that the algorithm views at each time step. The initialization and hyperparameters can also be adjusted here. There are additionally two functions that can be used for plotting: one to plot the firing rate and voltage for a given neuron and the other to plot the filters for a given neuron. 
