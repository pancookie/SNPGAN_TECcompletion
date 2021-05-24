import numpy as np

list_steps = [0, 5, 11, 16, 21, 27, 32]

def translateMAT(idx):
    n = 64

    d_ = 6 - idx
    # import pdb; pdb.set_trace()
    if d_ >= 0:
        d = list_steps[d_]
    elif d_ < 0:
        d = -1 * list_steps[abs(d_)]
    
    if d >= 0:
        return list(np.arange(d, n)) + list(np.arange(d))
    elif d < 0:
        return list(np.arange(n+d, n)) + list(np.arange(n+d))
