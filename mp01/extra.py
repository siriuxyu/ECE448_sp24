import numpy as np

def estimate_geometric(PX):
    '''
    @param:
    PX (numpy array of length cX): PX[x] = P(X=x), the observed probability mass function

    @return:
    p (scalar): the parameter of a matching geometric random variable
    PY (numpy array of length cX): PY[x] = P(Y=y), the first cX values of the pmf of a
      geometric random variable such that E[Y]=E[X].
    '''
    # raise RuntimeError("You need to write this")
    mean = 0
    for i in range(len(PX)):
        mean += i * PX[i]
    p = 1 / (mean + 1)
    
    PY = np.zeros(len(PX))
    for i in range(len(PX)):
        PY[i] = (1 - p) ** i * p
    return p, PY
