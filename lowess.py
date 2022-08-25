def lowess_with_confidence_bounds(
    x, y, eval_x, N=200, conf_interval=0.95, **lowess_kw,
):
    """
    Perform Lowess regression and determine a confidence interval by bootstrap resampling

    https://www.statsmodels.org/dev/examples/notebooks/generated/lowess.html
    """
    import statsmodels.api as sm
    import numpy as np
    # Lowess smoothing
    smoothed = sm.nonparametric.lowess(
        exog=x, endog=y, xvals=eval_x, **lowess_kw)

    # Perform bootstrap resamplings of the data
    # and  evaluate the smoothing at a fixed set of points
    smoothed_values = np.empty((N, len(eval_x)))
    for i in range(N):
        sample = np.random.choice(len(x), len(x), replace=True)
        sampled_x = x[sample]
        sampled_y = y[sample]

        smoothed_values[i] = sm.nonparametric.lowess(
            exog=sampled_x, endog=sampled_y, xvals=eval_x, **lowess_kw
        )

    # Get the confidence interval
    sorted_values = np.sort(smoothed_values, axis=0)
    bound = int(N * (1 - conf_interval) / 2)
    bottom = sorted_values[bound - 1]
    top = sorted_values[-bound]

    return smoothed, bottom, top
