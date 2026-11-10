import numpy as np, pandas as pd
from collections import Counter
from scipy.stats import mannwhitneyu
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from .data import EnvImputer, curate
from .transform import fit_index, crossfit_index, shuffle_check
from .scoring import score_upstream, score_last, pick
from .roles import lesion_side, invariance, orientation, assign
from .stats import ols, coef_p, evalue_from_d

class CMFS:
    """Causal mediation feature selection within a tiered causal graph (currently two tiers:
    upstream -> downstream -> outcome, with direct upstream -> outcome effects allowed)."""
    def __init__(self, spec, k=None, alpha=0.1, transform="pca", n_splits=5, n_repeats=10, seed=987654321):
        assert len(spec.tiers) == 2, "this version supports two mediator tiers"
        self.spec, self.alpha, self.kind = spec, alpha, transform
        self.k = k or {t: 5 for t in spec.tiers}; self.n_splits, self.n_repeats, self.seed = n_splits, n_repeats, seed

    # ---------- data
    def _arrays(self, df):
        s = self.spec; up, dn = s.tier_names()
        cols = {t: curate(df, s.tiers[t]) for t in (up, dn)}
        E = df[s.exposure].astype(float).values; Y = df[s.outcome].astype(float).values
        Cm = df[s.confounders].astype(float); Cm = ((Cm - Cm.mean()) / Cm.std()).fillna(0).values
        env = None
        if s.environment:
            e = df[s.environment].astype(float); env = ((e - e.mean()) / e.std()).fillna(0).values; Cm = np.column_stack([Cm, env])
        strata = df[s.strata].values if s.strata else None
        fam = {t: np.array([s.family(c) if s.family else c for c in cols[t]]) for t in (up, dn)}
        return dict(E=E, Y=Y, C=Cm, env=env, strata=strata, cols=cols, fam=fam, up=up, dn=dn,
                    X=df[cols[up] + cols[dn]].astype(float).values, n_up=len(cols[up]))
    def _groups(self, A):
        """List of (label, index of exposed participants in this group); 'all' if no strata."""
        E, st = A["E"], A["strata"]
        if st is None: return [("all", np.where(E == 1)[0])]
        lv = [v for v in pd.unique(st[E == 1]) if isinstance(v, str) and v]
        return [(v, np.where((E == 1) & (st == v))[0]) for v in sorted(lv)]
    def _index(self, Xn, E, Y):
        return fit_index("pca", Xn, E, Y)(Xn) if self.kind == "pca" else crossfit_index(self.kind, Xn, E, Y)

    # ---------- selection on one data set
    def _select(self, Xi, E, Y, C, A, idx):
        up, dn = A["up"], A["dn"]; nu = A["n_up"]
        Xu, Xd = Xi[idx][:, :nu], Xi[idx][:, nu:]; e, y, c = E[idx], Y[idx], C[idx]
        a_d, pa_d, b_d, s_d = score_last(Xd, e, y, c, upstream=Xu)
        sel_d = pick(s_d, pa_d, A["fam"][dn], self.k[dn], self.alpha)
        NI = self._index(Xd, e, y)
        a_u, pa_u, o_u = score_upstream(Xu, e, y, c, NI)
        sel_u = pick(o_u[:, 2], pa_u, A["fam"][up], self.k[up], self.alpha)
        tab_u = pd.DataFrame({"feature": np.array(A["cols"][up]), "exposure_to_feature": a_u, "p": pa_u, "direct": o_u[:, 0],
                              f"via_{dn}": o_u[:, 1], "total": o_u[:, 2]})
        tab_d = pd.DataFrame({"feature": np.array(A["cols"][dn]), "exposure_to_feature": a_d, "p": pa_d, "feature_to_outcome": b_d, "total": s_d})
        return sel_u, sel_d, tab_u, tab_d, NI

    # ---------- full-data analysis with roles
    def fit_full(self, df, orient_boot=100):
        A = self._arrays(df); E, Y, C = A["E"], A["Y"], A["C"]
        imp = EnvImputer().fit(A["X"], A["env"]); Xi = imp.transform(A["X"], A["env"]); ctrl = np.where(E == 0)[0]
        out = dict(groups={}, arrays=A)
        for g, ex in self._groups(A):
            idx = np.r_[ctrl, ex]; su, sd, tu, td, NI = self._select(Xi, E, Y, C, A, idx)
            tot = ols(Y[idx], np.column_stack([E[idx], C[idx]]))[1]
            chk = shuffle_check(self.kind, Xi[idx][:, A["n_up"]:], E[idx], Y[idx], C[idx], n=50) if self.kind != "pca" else 0.0
            out["groups"][g] = dict(n_exposed=len(ex), total=tot, upstream=tu.sort_values("total"), downstream=td.sort_values("total"),
                                    panel=list(np.array(A["cols"][A["up"]])[su]) + list(np.array(A["cols"][A["dn"]])[sd]), shuffle_check=chk)
        # roles for the union of selected features (pooled data)
        allcols = A["cols"][A["up"]] + A["cols"][A["dn"]]; panel = sorted(set(sum([v["panel"] for v in out["groups"].values()], [])), key=allcols.index)
        J = [allcols.index(f) for f in panel]; Xp = Xi[:, J]; keep = np.where((E == 0) | np.isin(np.arange(len(E)), np.concatenate([ex for _, ex in self._groups(A)])))[0]
        pa = coef_p(Xp[keep], np.column_stack([E[keep], C[keep]]), 1)[1]
        pb = np.array([coef_p(Y[keep][:, None], np.column_stack([Xp[keep][:, j], E[keep], C[keep]]), 1)[1][0] for j in range(len(J))])
        levels = [g for g, _ in self._groups(A)] if A["strata"] is not None else None
        ls = lesion_side(Xp[keep], A["strata"][keep] if A["strata"] is not None else None, E[keep], C[keep], levels)
        envb = (A["env"][keep] > np.median(A["env"][keep])).astype(float) if A["env"] is not None else np.zeros(len(keep))
        inv = invariance(Xp[keep], E[keep], Y[keep], C[keep], envb) if A["env"] is not None else np.ones(len(J))
        ori = orientation(Xp[keep], E[keep], Y[keep], C[keep], n_boot=orient_boot) if orient_boot else None
        rows = []
        for j, f in enumerate(panel):
            role, ev = assign(pa[j], pb[j], None if ls is None else ls["p"][j], inv[j], None if ori is None else ori[j])
            r = dict(feature=f, tier=A["up"] if J[j] < A["n_up"] else A["dn"], p_exposure=pa[j], p_outcome=pb[j], invariance_p=inv[j], role=role, evidence=ev,
                     selected_in=", ".join(g for g, v in out["groups"].items() if f in v["panel"]))
            if ls is not None: r.update({f"{levels[0]}_vs_control": ls["eff1"][j], f"{levels[1]}_vs_control": ls["eff2"][j], "side_p": ls["p"][j]})
            if ori is not None: r.update(adjacent=ori[j, 0], toward_outcome=ori[j, 1], from_outcome=ori[j, 2])
            rows.append(r)
        out["roles"] = pd.DataFrame(rows); return out

    # ---------- nested evaluation
    def evaluate(self, df, verbose=True):
        A = self._arrays(df); E, Y, C, X = A["E"], A["Y"], A["C"], A["X"]; groups = self._groups(A)
        lab = np.zeros(len(E), int)
        for gi, (g, ex) in enumerate(groups): lab[ex] = gi + 1
        coh = np.where(lab > 0 if True else E >= 0)[0]; coh = np.r_[np.where(E == 0)[0], coh]
        nu, up, dn = A["n_up"], A["up"], A["dn"]; allcols = np.array(A["cols"][up] + A["cols"][dn])
        famall = np.r_[A["fam"][up], A["fam"][dn]]; ktot = self.k[up] + self.k[dn]
        res, sel, oof = {}, {g: [] for g, _ in groups}, np.full((self.n_repeats, len(E)), np.nan)
        def auc(Xtr, ytr, Xte, yte):
            if len(np.unique(yte)) < 2: return np.nan
            return roc_auc_score(yte, RidgeClassifierCV(alphas=[0.1, 1, 10, 100, 1000]).fit(Xtr, ytr).decision_function(Xte))
        def add(k, v): res.setdefault(k, []).append(v)
        sp = RepeatedStratifiedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.seed)
        for f, (trp, tep) in enumerate(sp.split(coh, lab[coh])):
            tr, te = coh[trp], coh[tep]; imp = EnvImputer().fit(X[tr], None if A["env"] is None else A["env"][tr])
            Xtr = imp.transform(X[tr], None if A["env"] is None else A["env"][tr]); Xte = imp.transform(X[te], None if A["env"] is None else A["env"][te])
            panels = []
            for gi, (g, _) in enumerate(groups):
                ktr = np.where((lab[tr] == 0) | (lab[tr] == gi + 1))[0]; kte = np.where((lab[te] == 0) | (lab[te] == gi + 1))[0]
                su, sd, _, _, _ = self._select(Xtr, (lab[tr] == gi + 1).astype(float), Y[tr], C[tr], A, ktr)
                pan = np.r_[su, nu + sd]; panels.append(pan); sel[g].append(tuple(allcols[pan]))
                etr, ete = (lab[tr][ktr] == gi + 1).astype(float), (lab[te][kte] == gi + 1).astype(float)
                add(f"{g}: causal panel", auc(Xtr[ktr][:, pan], etr, Xte[kte][:, pan], ete))
                ni = fit_index(self.kind if self.kind == "pca" else self.kind, Xtr[ktr][:, nu:], etr, Y[tr][ktr])
                add(f"{g}: causal panel + {dn} index", auc(np.column_stack([Xtr[ktr][:, pan], ni(Xtr[ktr][:, nu:])]), etr, np.column_stack([Xte[kte][:, pan], ni(Xte[kte][:, nu:])]), ete))
                assoc = np.argsort([mannwhitneyu(Xtr[ktr][etr == 1, j], Xtr[ktr][etr == 0, j]).pvalue for j in range(Xtr.shape[1])])[:ktot]
                add(f"{g}: association panel", auc(Xtr[ktr][:, assoc], etr, Xte[kte][:, assoc], ete))
            Etr, Ete = (lab[tr] > 0).astype(float), (lab[te] > 0).astype(float); uni = np.unique(np.concatenate(panels))
            if len(groups) > 1:
                add("all: union of group panels", auc(Xtr[:, uni], Etr, Xte[:, uni], Ete))
                su, sd, _, _, _ = self._select(Xtr, Etr, Y[tr], C[tr], A, np.arange(len(tr)))
                pooled = np.r_[su, nu + sd]; add("all: pooled (group-blind) causal panel", auc(Xtr[:, pooled], Etr, Xte[:, pooled], Ete))
            assoc = np.argsort([mannwhitneyu(Xtr[Etr == 1, j], Xtr[Etr == 0, j]).pvalue for j in range(Xtr.shape[1])])[:len(uni)]
            add("all: association panel", auc(Xtr[:, assoc], Etr, Xte[:, assoc], Ete))
            a_sign = np.sign(coef_p(Xtr[:, uni], np.column_stack([Etr, C[tr]]), 1)[0]); a_sign[a_sign == 0] = 1
            ctr = (Xtr[:, uni] * a_sign).mean(1); oof[f // self.n_splits, te] = ((Xte[:, uni] * a_sign).mean(1) - ctr.mean()) / ctr.std()
            if verbose and f % 10 == 9: print(f"  fold {f+1}/{self.n_splits*self.n_repeats}", flush=True)
        M = np.nanmean(oof, 0)[coh]; Ec, Yc, Cc = (lab[coh] > 0).astype(float), Y[coh], C[coh]; rng = np.random.default_rng(self.seed)
        def med(i):
            a = ols(M[i], np.column_stack([Ec[i], Cc[i]]))[1]; b = ols(Yc[i], np.column_stack([M[i], Ec[i], Cc[i]]))[1]; t = ols(Yc[i], np.column_stack([Ec[i], Cc[i]]))[1]; return a * b, t
        e = med(np.arange(len(coh))); B = np.array([med(rng.integers(0, len(coh), len(coh))) for _ in range(2000)])
        lo, hi = np.percentile(B[:, 0], [2.5, 97.5]); sdy = Yc.std(); lim = hi if e[0] < 0 else lo
        stab = {g: np.mean([len(set(a) & set(b)) / len(set(a) | set(b)) for i, a in enumerate(v) for b in v[i + 1:]]) for g, v in sel.items()}
        freq = {g: Counter(x for t in v for x in t) for g, v in sel.items()}
        return dict(auc={k: (np.nanmean(v), np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)) for k, v in res.items()}, stability=stab, frequency=freq,
                    mediated=dict(estimate=e[0], lo=lo, hi=hi, total=e[1], share=e[0] / e[1], evalue=evalue_from_d(e[0] / sdy),
                                  evalue_ci=evalue_from_d(lim / sdy) if np.sign(lim) == np.sign(e[0]) else 1.0), n_folds=len(sp.split(coh, lab[coh]).__class__.__name__) and self.n_splits * self.n_repeats)
