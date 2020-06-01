# Notes - Synthetic

### `data_gen.py`

Generate simulated data from a given set of θ.

Can also generate θ using the Watt-Strogatz small world model or other models available in `networkx`.

### `sampling_base.py`

Generate sampling distributions of spike trains from a given θ (y_hat and s_hat). The data are then used to generate θ_hat.

θ_hata are compared with θ for false positives and false negatives using the Hamming metric.

Can check for dependence of FP/FN on model parameter.  

### `sampling_lambda.py`

Use `sampling_base.py` to check for dependence of regularization parameters λ1 and λ2 with other parameters.

