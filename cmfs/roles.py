import numpy as np
from .stats import coef_p

def lesion_side(Xf, strata, E, C, levels):
    """Natural experiment: if a feature differs between two exposure strata (e.g. left vs right lesion),
    the difference is attributable to the lesion, not to pre-exposure traits. Returns per-feature p."""
    if levels is None or len(levels) != 2: return None
    D1, D2 = (strata == levels[0]).astype(float), (strata == levels[1]).astype(float)
    X1 = np.column_stack([np.ones(len(E)), D1, D2, C]); B = np.linalg.lstsq(X1, Xf, rcond=None)[0]
    R = Xf - X1 @ B; df = len(E) - X1.shape[1]; V = np.linalg.inv(X1.T @ X1)
    s2 = (R ** 2).sum(0) / df; diff = B[1] - B[2]; se = np.sqrt(s2 * (V[1, 1] + V[2, 2] - 2 * V[1, 2]))
    from scipy.stats import t as tdist
    return dict(eff1=B[1], eff2=B[2], p=2 * tdist.sf(np.abs(diff / se), df))

def invariance(Xf, E, Y, C, env_bin):
    """Do the exposure -> feature and feature -> outcome paths hold across environments?
    Returns the smaller interaction p (a-path E x env, b-path feature x env)."""
    p_out = np.ones(Xf.shape[1])
    for j in range(Xf.shape[1]):
        f = Xf[:, j]
        _, pa = coef_p(f[:, None], np.column_stack([E * env_bin, E, env_bin, C]), 1)
        _, pb = coef_p(Y[:, None], np.column_stack([f * env_bin, f, env_bin, E, C]), 1)
        p_out[j] = min(pa[0], pb[0])
    return p_out

def orientation(Xp, E, Y, C, n_boot=100, alpha=0.1, seed=0):
    """Collider-based orientation (PC algorithm, bootstrapped) of each feature-outcome edge, with
    confounders before exposure before {features, outcome}; features and outcome share a tier so
    the direction between them comes from the data. Returns shares (adjacent, feature->Y, Y->feature)."""
    try:
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
        from causallearn.graph.GraphNode import GraphNode
    except Exception:
        return None
    import io, contextlib
    k = Xp.shape[1]; cols = [f"C{i}" for i in range(C.shape[1])] + ["E"] + [f"M{j}" for j in range(k)] + ["Y"]
    tier = {c: 0 for c in cols[:C.shape[1]]}; tier["E"] = 1; tier.update({c: 2 for c in cols[C.shape[1] + 1:]})
    D = np.column_stack([C, E, Xp, Y]); rng = np.random.default_rng(seed); iY = len(cols) - 1
    cnt = np.zeros((k, 3))
    for _ in range(n_boot):
        i = rng.integers(0, len(E), len(E))
        if E[i].min() == E[i].max(): continue
        bk = BackgroundKnowledge()
        for c in cols: bk.add_node_to_tier(GraphNode(c), tier[c])
        with contextlib.redirect_stdout(io.StringIO()):
            g = pc(D[i] + rng.normal(0, 1e-6, D.shape), alpha, "fisherz", background_knowledge=bk, show_progress=False, node_names=cols).G.graph
        for j in range(k):
            m = C.shape[1] + 1 + j; mY, mM = g[iY, m], g[m, iY]
            if mY == 0 and mM == 0: continue
            cnt[j, 0] += 1
            if mY == 1 and mM == -1: cnt[j, 1] += 1
            if mY == -1 and mM == 1: cnt[j, 2] += 1
    return cnt / n_boot

def assign(pa, pb, side_p, inv_p, orient, alpha=0.05):
    """Evidence-graded role for one feature."""
    if orient is not None and orient[0] >= 0.3 and orient[2] > orient[1] + 0.1:
        role = "possible consequence of outcome"
    elif inv_p < alpha:
        role = "environment-sensitive"
    elif pa < alpha and pb < 0.1:
        role = "mediator"
    elif pa < alpha:
        role = "diagnostic marker"
    else:
        role = "weak / unconfirmed"
    evidence = []
    if side_p is not None and side_p < alpha: evidence.append("lesion-confirmed")
    if orient is not None and orient[0] >= 0.3 and orient[1] > orient[2] + 0.1: evidence.append("oriented toward outcome")
    if inv_p >= alpha: evidence.append("invariant across environments")
    return role, "; ".join(evidence)
