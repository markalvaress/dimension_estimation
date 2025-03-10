import numpy as np
import scipy as sp
import warnings
from typing import *
import multiprocessing
from multiprocessing.pool import Pool

def open_ball_subset(X, i: int, r: float):
    """Get the subset of points in x contained in the open ball around the i'th point (but excluding the i'th point itself)"""
    m, D = X.shape
    
    W = None
    for j in range(m): # is there a faster way to do this?
        if j == i:
            # by definition in their paper W does not include the point itself
            continue
        elif np.linalg.norm(X[j] - X[i], 2) < r:
            if W is not None:
                W = np.vstack((W, X[j]))
            else:
                # on the first time round, fill in the first col of W
                W = X[j]

    if W is None:
        warnings.warn(f"Warning: no points found within radius {r} of the {i}th point in X.")

    return W

def Thr(L: np.ndarray, eta: float) -> int:
    """The threshold function mentioned on p651 of the paper."""
    k = 0

    # doing it like this so i don't have to compute sum(L) lots of times
    sum_L = sum(L)
    rhs = eta * sum_L
    lhs = sum_L
    while k < len(L):
        lhs -= L[k]
        if lhs < rhs:
            return k+1
        k += 1

    raise Exception("couldn't find a threshold k value")
    

def tgt_and_dim_estimate_1point(X: np.ndarray, i: int, r: float, eta: float) -> Tuple[int, int, np.ndarray]:
    """Estimate the tangent space and the intrinsic dimension of X using the neighbourhood of the i'th point.

    Args:
        X (np.ndarray): An m x D matrix, representing m points in R^D (each row is a point)
        i (int): The index of the point you want to compute the local estimates around
        r (float): The radius to use for local estimation of dimension
        eta (float): Threshold parameter for dimension estimation

    Returns:
        Tuple[int, int, np.ndarray]: Returns i, the index of the data point; d, the estimated intrinsic dimension; and T, a matrix with rows
        representing d points in R^D, and the span of these is the estimated tangent space.
    """
    m_X, D = X.shape
    
    # take only points within the ball of radius r centred on the pt
    W = open_ball_subset(X, i, r)

    m = W.shape[0]

    # compute centroid and empirical covariance matrix
    x_bar = (1/m) * sum([W[i] for i in range(m)])
    Sigma = (1/m) * sum([np.outer(W[i] - x_bar, W[i] - x_bar) for i in range(m)])

    # compute svd
    U, L, U_T = sp.linalg.svd(Sigma) # L sorted in decreasing (not strictly) order

    # estimated intrinsic dimension
    d_hat = Thr(L, eta)

    return i, d_hat, U[:d_hat]

def tgt_and_dim_estimates(X: np.ndarray, r: float, eta: float) -> List[Tuple[int, int, np.ndarray]]:
    """Estimate the intrinsic dimension and the tangent space at every single point in X.

    Args:
        X (np.ndarray): An m x D matrix, representing m points in R^D (each row is a point)
        r (float): The radius to use for local estimation of dimension
        eta (float): Threshold parameter for dimension estimation

    Returns:
        List[Tuple[int, int, np.ndarray]]: A list of size len(X). For each point contains a tuple with i, the index of the data point;
        d, the estimated intrinsic dimension; and T, a matrix with rows representing d points in R^D, and the span of these is the estimated tangent space.
    """
    ## this was the attempt to parallelise but it didn't work - I'd need to call it as a module.
    # num_processes = multiprocessing.cpu_count()
    # with Pool(num_processes) as p:
    #     results = p.map(lambda i: tgt_and_dim_estimate_1point(X, i, r, eta), range(len(X)))

    results = [tgt_and_dim_estimate_1point(X, i, r, eta) for i in range(len(X))]

    return results

