import numpy as np
from .stats import coef_p
from sklearn.decomposition import PCA

def score_upstream(Xb, E, Y, C, NI):
    """Route-specific scores for an upstream tier (e.g. blood) given the downstream index NI (e.g. neuro):
    direct = a*b1 (feature -> outcome), via_down = a*d*b2 (feature -> downstream -> outcome)."""
    a, pa = coef_p(Xb, np.column_stack([E, C]), 1); n = len(E); out = np.zeros((Xb.shape[1], 3))
    for j in range(Xb.shape[1]):
        dj = np.linalg.lstsq(np.column_stack([np.ones(n), Xb[:, j], E, C]), NI, rcond=None)[0][1]
        bb = np.linalg.lstsq(np.column_stack([np.ones(n), Xb[:, j], NI, E, C]), Y, rcond=None)[0]
        out[j] = [a[j] * bb[1], a[j] * dj * bb[2], a[j] * (bb[1] + dj * bb[2])]
    return a, pa, out

def score_last(Xn, E, Y, C, upstream=None, n_pc=5):
    """Scores for the last tier (e.g. neuro): a*b, with the upstream block summarised by label-free
    principal components and adjusted for (it is upstream and may confound the feature -> outcome path)."""
    adj = C if upstream is None else np.column_stack([C, PCA(min(n_pc, upstream.shape[1])).fit_transform(upstream)])
    a, pa = coef_p(Xn, np.column_stack([E, adj]), 1); n = len(E); b = np.zeros(Xn.shape[1])
    for j in range(Xn.shape[1]):
        b[j] = np.linalg.lstsq(np.column_stack([np.ones(n), Xn[:, j], E, adj]), Y, rcond=None)[0][1]
    return a, pa, b, a * b

def pick(score, pa, fam, k, alpha=0.1, min_pool=None):
    """Eligible if exposure -> feature p < alpha (fallback: smallest p); rank by score (most negative =
    carries the exposure's harmful effect); keep one feature per measurement family."""
    el = np.where(pa < alpha)[0]; need = k + 5 if min_pool is None else min_pool
    if len(el) < k: el = np.argsort(pa)[:need]
    out, seen = [], set()
    for j in el[np.argsort(score[el])]:
        if fam[j] in seen: continue
        out.append(j); seen.add(fam[j])
        if len(out) == k: break
    return np.array(out, dtype=int)
