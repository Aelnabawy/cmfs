import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

class EnvImputer:
    """Standardise, then KNN-impute with the environment as an extra (weighted) coordinate,
    so missing values are filled from participants of the same study phase."""
    def __init__(self, k=5, env_weight=2.0): self.k, self.w = k, env_weight
    def fit(self, X, env=None):
        self.sc = StandardScaler().fit(X); A = self._aug(self.sc.transform(X), env)
        self.imp = KNNImputer(n_neighbors=self.k).fit(A); return self
    def _aug(self, Z, env): return Z if env is None else np.column_stack([Z, self.w * env])
    def transform(self, X, env=None):
        out = self.imp.transform(self._aug(self.sc.transform(X), env)); return out if env is None else out[:, :-1]

def curate(df, cols, max_missing=0.5):
    """Keep numeric candidate columns with <= max_missing missingness and non-zero variance."""
    keep = []
    for c in cols:
        v = df[c].astype(float)
        if v.isna().mean() <= max_missing and v.nunique() > 1: keep.append(c)
    return keep
