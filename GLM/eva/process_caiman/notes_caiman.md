# CaImAn Analysis of Neural Data

This set of scripts

* Run the CaImAn pipeline to get `cnm.estimates`, which includes [spatial and temporal components](https://caiman.readthedocs.io/en/master/Getting_Started.html#result-variables-for-2p-batch-analysis).

* Compare results from `CaImAn batch` (offline) and `OnACID` (online) algorithms.


## Instructions

1. Download `08-17-14_1437_F1_6dpfCOMPLETESET_WB_overclimbing_z-1.tbif` from `improv Data Sharing` into this directory.
2. Run each script sequentially.
    * `1_process_tbif.py` extracts information of the `tbif` file.
    * `2x_.py` runs `CaImAn` with tuned parameters, based on official demos.
    * `3_extract_estimates.py` extracts inferred spikes into a pickle file.
    
The output would be in the `eva` folder, ready to be fit.