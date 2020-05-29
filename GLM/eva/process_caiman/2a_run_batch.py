#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Basic stripped-down demo for running the CNMF source extraction algorithm with
CaImAn and evaluation the components. The analysis can be run either in the
whole FOV or in patches. For a complete pipeline (including motion correction)
check demo_pipeline.py
Data courtesy of W. Yang, D. Peterka and R. Yuste (Columbia University)

This demo is designed to be run under spyder or jupyter; its plotting functions
are tailored for that environment.

@authors: @agiovann and @epnev

"""

import cv2
import glob
import logging
import numpy as np
import os

try:
    cv2.setNumThreads(0)
except:
    pass

try:
    if __IPYTHON__:
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass


import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.summary_images import local_correlations_movie_offline

# %%
# Set up the logger; change this if you like.
# You can log to a file using the filename parameter, or make the output more or less
# verbose by setting level to logging.DEBUG, logging.INFO, logging.WARNING, or logging.ERROR

logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
                    "[%(process_caiman)d] %(message)s",
                    level=logging.WARNING)
                    # filename="/tmp/caiman.log",

# %% start a cluster

c, dview, n_processes =\
    cm.cluster.setup_cluster(backend='local', n_processes=None,
                         single_thread=False)

# %% set up some parameters
fnames = ['/Users/chaichontat/Desktop/08-17-14_1437_F1_6dpfCOMPLETESET_WB_overclimbing_z-1.npz.tif']

is_patches = False
if is_patches:          # PROCESS IN PATCHES AND THEN COMBINE
    rf = 20             # half size of each patch
    stride = 4          # overlap between patches
    K = 4               # number of components in each patch
else:                   # PROCESS THE WHOLE FOV AT ONCE
    rf = None           # setting these parameters to None
    stride = None       # will run CNMF on the whole FOV
    K = 500              # number of neurons expected (in the whole FOV)



params_dict = {'fnames': fnames,
               'fr': 2,
               'decay_time': 0.7,
               'gSig': (3, 3),
               'rf': rf,
               'stride': stride,
               'merge_thr': 0.8,
               'p': 1,
               'min_SNR': 1.5,
               'rval_thr': 0.9,
               'nb': 2,
               'K': 500,
               'normalize': True,
               'dist_shape_update': True,
               'show_movie': False,
               'pw_rigid': True
               }

opts = params.CNMFParams(params_dict=params_dict)

# %%% MOTION CORRECTION
# first we create a motion correction object with the specified parameters
mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
# note that the file is not loaded in memory

# %% Run (piecewise-rigid motion) correction using NoRMCorre
mc.motion_correct(save_movie=True)
border_to_0 = 0 if mc.border_nan is 'copy' else mc.border_to_0
# you can include the boundaries of the FOV if you used the 'copy' option
# during motion correction, although be careful about the components near
# the boundaries

# memory map the file in order 'C'
fname_new = cm.save_memmap(mc.mmap_file, base_name='memmap_', order='C',
                           border_to_0=border_to_0)  # exclude borders

# now load the file
Yr, dims, T = cm.load_memmap(fname_new)
images = np.reshape(Yr.T, [T] + list(dims), order='F')

import tifffile
tifffile.imsave('mc.tif', images)
# load frames in python format (T x X x Y)

# %% restart cluster to clean up memory
cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)

# %% Now RUN CaImAn Batch (CNMF)
cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
cnm = cnm.fit_file()

# %% plot contour plots of components
Cns = local_correlations_movie_offline(fnames[0],
                                       remove_baseline=True,
                                       swap_dim=False, window=1000, stride=1000,
                                       winSize_baseline=100, quantil_min_baseline=10,
                                       dview=dview)
Cn = Cns.max(axis=0)
# cnm.estimates.plot_contours(img=Cn)

# %% load memory mapped file
Yr, dims, T = cm.load_memmap(cnm.mmap_file)
images = np.reshape(Yr.T, [T] + list(dims), order='F')

# %% refit
cnm2 = cnm.refit(images, dview=dview)

# %% COMPONENT EVALUATION
# the components are evaluated in three ways:
#   a) the shape of each component must be correlated with the data
#   b) a minimum peak SNR is required over the length of a transient
#   c) each shape passes a CNN based classifier (this will pick up only neurons
#           and filter out active processes)

# min_SNR = 2      # peak SNR for accepted components (if above this, acept)
# rval_thr = 0.85     # space correlation threshold (if above this, accept)
# use_cnn = True      # use the CNN classifier
# min_cnn_thr = 0.99  # if cnn classifier predicts below this value, reject
# cnn_lowest = 0.1 # neurons with cnn probability lower than this value are rejected

cnm2.params.set('quality', {'SNR_lowest': 0.8,
 'cnn_lowest': 0.1,
 'min_SNR': 2,
 'min_cnn_thr': 0.9,
 'rval_lowest': -1,
 'rval_thr': 0.9,
 'use_cnn': True,})

cnm2.estimates.evaluate_components(images, cnm2.params, dview=dview)

# %% visualize selected and rejected components
# cnm2.estimates.plot_contours(img=Cn, idx=cnm2.estimates.idx_components)
# %% visualize selected components
# cnm2.estimates.view_components(images, idx=cnm2.estimates.idx_components, img=Cn)
#%% only select high quality components (destructive)
# cnm2.estimates.select_components(use_object=True)
# cnm2.estimates.plot_contours(img=Cn)
#%% save results
cnm2.save('batch.hdf5')
# cnm2.estimates.Cn = Cn


# %% play movie with results (original, reconstructed, amplified residual)
#     cnm2.estimates.play_movie(images, magnification=4)

# %% STOP CLUSTER and clean up log files
cm.stop_server(dview=dview)

log_files = glob.glob('Yr*_LOG_*')
for log_file in log_files:
    os.remove(log_file)

# %%
# This is to mask the differences between running this demo in Spyder
# versus from the CLI

