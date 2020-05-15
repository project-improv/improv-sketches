# Notes - Synthetic

### `data_gen.py`

Generate simulated data from a given set of θ.

Can also generate θ using the Watt-Strogatz small world model.

### `sampling_base.py`

Designed primarily to generate a sampling distribution of the fitted θ.

### `sampling_lambda.py`

Wraps `sampling_base.py` to check for dependence of regularization parameters λ1 and λ2 with other parameters.