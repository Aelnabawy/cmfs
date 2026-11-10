import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import StratifiedKFold
from .stats import ols

KINDS = ("pca", "lda", "pls")

def fit_index(kind, Xtr, e_tr, y_tr):
    """Fit a one-dimensional transformation of a feature block. pca = label-free (linear autoencoder),
    lda = exposure-supervised (shrinkage LDA), pls = outcome-supervised. Returns a scoring function and
    the standardisation constants learned on the training data."""
    if kind == "pca":
        w = PCA(1).fit(Xtr).components_[0]; f = lambda M: M @ w
    elif kind == "lda":
        m = LDA(solver="lsqr", shrinkage="auto").fit(Xtr, e_tr); f = m.decision_function
    elif kind == "pls":
        m = PLSRegression(1, scale=False).fit(Xtr, y_tr); f = lambda M: m.transform(M).ravel()
    else: raise ValueError(kind)
    s = f(Xtr); mu, sd = s.mean(), s.std()
    return lambda M: (f(M) - mu) / sd

def crossfit_index(kind, X, e, y, n_splits=5, seed=0):
    """Out-of-fold scores: each participant is scored by a transformation that never saw them."""
    out = np.zeros(len(e))
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, e):
        out[te] = fit_index(kind, X[tr], e[tr], y[tr])(X[te])
    return (out - out.mean()) / out.std()

def shuffle_check(kind, X, e, y, C, n=100, seed=0):
    """Median |indirect effect| the in-sample transformation produces when exposure and outcome are
    shuffled: the effect built into the method itself (should be ~0 for honest use)."""
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        ep, yp = rng.permutation(e), rng.permutation(y); s = fit_index(kind, X, ep, yp)(X)
        a = ols(s, np.column_stack([ep, C]))[1]; b = ols(yp, np.column_stack([s, ep, C]))[1]; vals.append(abs(a * b))
    return float(np.median(vals))
