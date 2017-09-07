
import numpy as np

def create_sample_index(rate, len):
    np.random.seed(13)
    return np.random.choice(len, int(rate * len))

def sample_arrays(arrays, index):
    ret = []
    for a in arrays:
        if a is None:
            ret.append(None)
            continue
        ret.append(a[index])
    return tuple(ret)
