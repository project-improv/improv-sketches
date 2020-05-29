#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Complete pipeline for online processing using CaImAn Online (OnACID).
The demo demonstates the analysis of a sequence of files using the CaImAn online
algorithm. The steps include i) motion correction, ii) tracking current 
components, iii) detecting new components, iv) updating of spatial footprints.
The script demonstrates how to construct and use the params and online_cnmf
objects required for the analysis, and presents the various parameters that
can be passed as options. A plot of the processing time for the various steps
of the algorithm is also included.
@author: Eftychios Pnevmatikakis @epnev
Special thanks to Andreas Tolias and his lab at Baylor College of Medicine
for sharing the data used in this demo.
"""

import glob
import numpy as np
import os
import logging
import matplotlib.pyplot as plt

try:
    if __IPYTHON__:
        # this is used for debugging purposes only.
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

import caiman as cm
from caiman.paths import caiman_datadir
from caiman.source_extraction import cnmf as cnmf
from caiman.utils.utils import download_demo

logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
                    "[%(process)d] %(message)s",
                    level=logging.INFO)

# %%
def main():
    pass # For compatibility between running under Spyder and the CLI

# %%  download and list all files to be processed

    # folder inside ./example_movies where files will be saved
    fnames = ['/Users/chaichontat/Desktop/08-17-14_1437_F1_6dpfCOMPLETESET_WB_overclimbing_z-1.npz.tif']

    # your list of files should look something like this
    logging.info(fnames)

# %%   Set up some parameters
    params_dict = {'fnames': fnames,
                   'fr': 2,
                   'decay_time': 0.7,
                   'gSig': (3, 3),
                   'p': 1,
                   'min_SNR': 1.5,
                   'thresh_CNN_noisy': 0.6,
                   'ds_factor': 1,
                   'nb': 2,
                   'motion_correct': True,
                   'init_batch': 200,
                   'init_method': 'bare',
                   'sniper_mode': True,
                   'K': 10,
                   'epochs': 1,
                   'max_shifts_online': 10,
                   'pw_rigid': True,
                   'dist_shape_update': True,
                   'show_movie': False,
                   'rval_thr': 0.85}

    opts = cnmf.params.CNMFParams(params_dict=params_dict)
    cnm = cnmf.online_cnmf.OnACID(params=opts)
    cnm.fit_online()
    cnm.save('onacid.hdf5')

# %% plot contours (this may take time)
#     logging.info('Number of components: ' + str(cnm.estimates.A.shape[-1]))
#     images = cm.load(fnames)
#     Cn = images.local_correlations(swap_dim=False, frames_per_chunk=500)
    # cnm.estimates.plot_contours(img=Cn, display_numbers=False)

# %% view components
#     cnm.estimates.view_components(img=Cn)

# %% plot timing performance (if a movie is generated during processing, timing
# will be severely over-estimated)

    # T_motion = 1e3*np.array(cnm.t_motion)
    # T_detect = 1e3*np.array(cnm.t_detect)
    # T_shapes = 1e3*np.array(cnm.t_shapes)
    # T_track = 1e3*np.array(cnm.t_online) - T_motion - T_detect - T_shapes
    # plt.figure()
    # plt.stackplot(np.arange(len(T_motion)), T_motion, T_track, T_detect, T_shapes)
    # plt.legend(labels=['motion', 'tracking', 'detect', 'shapes'], loc=2)
    # plt.title('Processing time allocation')
    # plt.xlabel('Frame #')
    # plt.ylabel('Processing time [ms]')
#%% RUN IF YOU WANT TO VISUALIZE THE RESULTS (might take time)
    # c, dview, n_processes = \
    #     cm.cluster.setup_cluster(backend='local', n_processes=None,
    #                              single_thread=False)
    # if opts.online['motion_correct']:
    #     shifts = cnm.estimates.shifts[-cnm.estimates.C.shape[-1]:]
    #     if not opts.motion['pw_rigid']:
    #         memmap_file = cm.motion_correction.apply_shift_online(images, shifts,
    #                                                     save_base_name='MC')
    #     else:
    #         mc = cm.motion_correction.MotionCorrect(fnames, dview=dview,
    #                                                 **opts.get_group('motion'))
    #
    #         mc.y_shifts_els = [[sx[0] for sx in sh] for sh in shifts]
    #         mc.x_shifts_els = [[sx[1] for sx in sh] for sh in shifts]
    #         memmap_file = mc.apply_shifts_movie(fnames, rigid_shifts=False,
    #                                             save_memmap=True,
    #                                             save_base_name='MC')
    # else:  # To do: apply non-rigid shifts on the fly
    #     memmap_file = images.save(fnames[0][:-4] + 'mmap')
    # cnm.mmap_file = memmap_file
    # Yr, dims, T = cm.load_memmap(memmap_file)
    #
    # images = np.reshape(Yr.T, [T] + list(dims), order='F')
    # min_SNR = 2  # peak SNR for accepted components (if above this, acept)
    # rval_thr = 0.85  # space correlation threshold (if above this, accept)
    # use_cnn = True  # use the CNN classifier
    # min_cnn_thr = 0.99  # if cnn classifier predicts below this value, reject
    # cnn_lowest = 0.1  # neurons with cnn probability lower than this value are rejected
    #
    # cnm.params.set('quality',   {'min_SNR': min_SNR,
    #                             'rval_thr': rval_thr,
    #                             'use_cnn': use_cnn,
    #                             'min_cnn_thr': min_cnn_thr,
    #                             'cnn_lowest': cnn_lowest})
    #
    # cnm.estimates.evaluate_components(images, cnm.params, dview=dview)
    # cnm.estimates.Cn = Cn
    # cnm.save(os.path.splitext(fnames[0])[0]+'_results.hdf5')
    #
    # dview.terminate()

#%%
# This is to mask the differences between running this demo in Spyder
# versus from the CLI
if __name__ == "__main__":
    main()
