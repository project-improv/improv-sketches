import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np

from caiman.source_extraction.cnmf import cnmf as cnmf
from pathlib import Path

try:
    cv2.setNumThreads(0)
except:
    pass

if Path('batch.hdf5').exists():
    print('Processing batch.')
    cnm = cnmf.load_CNMF('batch.hdf5')
    cnm.estimates.select_components(use_object=True)

    # Check for sensibility
    keep = np.mean(cnm.estimates.C, axis=1) > 3000
    overlay = np.sum(cnm.estimates.A[:, keep], axis=1).reshape((800, 500)) > 0
    plt.imshow(overlay)
    plt.show()

    cnm.estimates.detrend_df_f()
    cnm.deconvolve(p=1)
    cnm.save('batch.hdf5')

    # plt.imshow(cnm.estimates.S[keep, :]/1000, vmin=0, vmax=10)
    # plt.grid(0)
    # plt.show()

    with open('../batch_A.pk', 'wb') as f:
        pickle.dump(cnm.estimates.A, f)

    mean = np.max(cnm.estimates.S)
    with open('../batch_S.pk', 'wb') as f:
        pickle.dump(cnm.estimates.S[keep, :] / np.mean(cnm.estimates.S[cnm.estimates.S>0]), f)  # Normalization

if Path('onacid.hdf5').exists():
    print('Processing OnACID')
    cnm = cnmf.load_CNMF('onacid.hdf5')
    with open('../onacid_A.pk', 'wb') as f:
        pickle.dump(cnm.estimates.A, f)

    mean = np.mean(cnm.estimates.S)
    with open('../onacid_S.pk', 'wb') as f:
        pickle.dump(cnm.estimates.S / np.mean(cnm.estimates.S[cnm.estimates.S>0]), f)
