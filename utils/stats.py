import numpy as np

def stats_report(array, ignore_nan):
    if ignore_nan:
        return np.nanmean(array), np.nanvar(array)
    else:
        return np.mean(array), np.var(array)
