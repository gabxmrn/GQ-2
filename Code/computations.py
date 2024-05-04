import numpy as np


def computationfdr(pvec, pnul, threshold):
    """
    Calculate the false discovery rate (FDR) for positive and negative test results.
    
    Parameters:
        pvec (numpy.ndarray): An array of p-values.
        pnul (float): The proportion of null hypothesis.
        threshold (float): The threshold for p-value rejection.
    
    Returns:
        tuple: Contains three floats representing the overall FDR, FDR for negative side, 
               and FDR for positive side.
    """

    n = len(pvec)
    pr = np.sum(pvec < threshold) / n
    fdr = pnul * threshold / pr
    signpvec = np.where(pvec >= 0, 1, -1)

    # computation of the fdr negative side
    selecn = np.where(signpvec < 0)
    pvecneg = pvec[selecn]
    prn = np.sum(pvecneg < threshold) / n
    fdrneg = (pnul * threshold / 2) / prn

    # computation of the fdr positive side
    selecp = np.where(signpvec > 0)
    pvecpos = pvec[selecp]
    prp = np.sum(pvecpos < threshold) / n
    fdrpos = (pnul * threshold / 2) / prp

    return fdr, fdrneg, fdrpos


def computationproportions(pvec, nbsimul):
    """
    Compute the null, negative, and positive proportions of p-values based on a bootstrap method.
    
    Parameters:
        pvec (numpy.ndarray): An array of p-values.
        nbsimul (int): Number of bootstrap simulations to perform.
    
    Returns:
        tuple: Contains three floats representing the null proportion, negative proportion,
               and positive proportion.
    """
    n = len(pvec)
    R = np.arange(0.50, 1.0, 0.05)
    nbtest = len(R)
    pnultot = np.zeros(nbtest)
    signpvec = np.where(pvec >= 0, 1, -1)
    
    for i in range(nbtest):
        W = np.sum(pvec >= R[i])
        pnultot[i] = (W / n) / (1 - R[i])
    minp = np.min(pnultot)
    bootpnultot = np.zeros((nbtest, nbsimul))
    
    for j in range(nbsimul):
        B = pvec[np.random.randint(0, n, n)]
        for i in range(nbtest):
            W = np.sum(B >= R[i])
            bootpnultot[i, j] = W / ((1 - R[i]) * n)
            
    difference = bootpnultot - minp
    squared = np.square(difference)
    mse = np.mean(squared, axis=1)
    lambda_ = np.min(mse)
    index = np.argmin(mse)
    optimalR = R[index]
    pnul = pnultot[index]
    pnul = np.clip(pnul, 0, 1)

    # computation of the negative and positive proportions
    selecn = np.where(signpvec < 0)
    pvecneg = pvec[selecn]
    pnega = np.sum(pvecneg < optimalR) / n
    pneg = pnega - pnul * optimalR / 2
    pneg = max(pneg, 0)
    selecp = np.where(signpvec > 0)
    pvecpos = pvec[selecp]
    pposa = np.sum(pvecpos < optimalR) / n
    ppos = pposa - pnul * optimalR / 2
    ppos = max(ppos, 0)

    return pnul, pneg, ppos
