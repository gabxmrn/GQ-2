import numpy as np

def computationfdr(alphas, pvec, pnul, threshold):
    """
    Calculate the false discovery rate (FDR) for positive and negative test results,
    handling cases where no p-values fall below the threshold by returning None or zero.

    Parameters:
        alphas (numpy.ndarray): An array of alphas
        pvec (numpy.ndarray): An array of p-values.
        pnul (float): The proportion of null hypothesis.
        threshold (float): The threshold for p-value rejection.

    Returns:
        tuple: Contains three floats representing the overall FDR, FDR for negative side, 
               and FDR for positive side. Returns None or zero where calculation is not possible.
    """
    n = len(pvec)
    pr = np.sum(pvec < threshold) / n #if n > 0 else 0  
    fdr = pnul * threshold / pr #if pr > 0 else 0  # 

    neg_pvec = pvec[alphas < 0]
    pos_pvec = pvec[alphas > 0]

    # Compute for the negative side
    prn = np.sum(neg_pvec < threshold) / len(neg_pvec) if n > 0 else 0
    fdrneg = (pnul * threshold / 2) / prn if prn > 0 else 0

    # Compute for the positive side
    prp = np.sum(pos_pvec < threshold) / len(pos_pvec) if n > 0 else 0
    fdrpos = (pnul * threshold / 2) / prp if prp > 0 else 0

    return fdr, fdrneg, fdrpos

def computationproportions(alphas, pvec, nbsimul):
    """
    Compute the null, negative, and positive proportions of p-values based on a bootstrap method,
    considering the sign of alphas.

    Parameters:
        alphas (numpy.ndarray): Array of alpha values.
        pvec (numpy.ndarray): An array of p-values corresponding to alphas.
        nbsimul (int): Number of bootstrap simulations to perform.

    Returns:
        tuple: Contains three floats representing the null proportion, proportion of significant negative alphas,
               and proportion of significant positive alphas.
    """
    n = len(pvec)
    R = np.arange(0.50, 1.0, 0.05)
    nbtest = len(R)
    pnultot = np.zeros(nbtest)

    for i in range(nbtest):
        W = np.sum(pvec >= R[i])
        pnultot[i] = (W / n) / (1 - R[i])

    minp = np.min(pnultot)
    bootpnultot = np.zeros((nbtest, nbsimul))

    for j in range(nbsimul):
        B = np.random.choice(pvec, size=n, replace=True)
        for i in range(nbtest):
            W = np.sum(B >= R[i])
            bootpnultot[i, j] = W / ((1 - R[i]) * n)

    mse = np.mean(np.square(bootpnultot - minp), axis=1)
    index = np.argmin(mse)
    optimalR = R[index]
    pnul = pnultot[index]
    pnul = np.clip(pnul, 0, 1)

    # Split based on alpha sign and compute proportions
    neg_pvec = pvec[alphas < 0]
    pos_pvec = pvec[alphas > 0]
    pnega = np.sum(neg_pvec < optimalR) / n
    pposa = np.sum(pos_pvec < optimalR) / n
    pneg = max(pnega - pnul * optimalR / 2, 0)
    ppos = max(pposa - pnul * optimalR / 2, 0)

    return pnul, pneg, ppos, np.sum([pnul, pneg, ppos])
