import numpy as np
from scipy.stats import t as tdist

def ols(y, X):
    X1 = np.column_stack([np.ones(len(y)), X]); return np.linalg.lstsq(X1, y, rcond=None)[0]

def coef_p(Y, X, j=1):
    """Vectorised OLS of each column of Y on [1, X]; returns coefficient j and its two-sided p."""
    X1 = np.column_stack([np.ones(X.shape[0]), X]); B = np.linalg.lstsq(X1, Y, rcond=None)[0]
    R = Y - X1 @ B; df = X1.shape[0] - X1.shape[1]
    se = np.sqrt((R ** 2).sum(0) / df * np.linalg.inv(X1.T @ X1)[j, j]); b = B[j]
    return b, 2 * tdist.sf(np.abs(b / se), df)

def evalue_from_d(d):
    rr = np.exp(0.91 * abs(d)); return rr + np.sqrt(rr * (rr - 1))
