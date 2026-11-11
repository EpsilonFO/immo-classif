"""Modèles. Signature commune : fit(Xtr, ytr, Xva, yva, Xte, params) -> (proba validation, proba test).
Le pli de validation sert à l'early stopping ; la prédiction est la moyenne de `seeds` modèles."""

import numpy as np
import pandas as pd

from hmda.config import SEED
from hmda.features import HIGH_CARD, cat_columns, key_columns


def _pop(p, n_est=5000, patience=200):
    p = dict(p)
    return p, p.pop("seeds", 1), p.pop("n_estimators", n_est), p.pop("patience", patience)


def target_encode(Xtr, ytr, Xva, Xte, cols, smooth=20.0, n_inner=5):
    """Target encoding lissé appris dans le pli : valeurs OOF (plis internes) pour le train, ajustées sur tout le
    train du pli pour validation et test. Ajoute une colonne `te_<col>` par variable."""
    from sklearn.model_selection import KFold

    Xtr, Xva, Xte = Xtr.copy(), Xva.copy(), Xte.copy()
    prior = float(np.mean(ytr))
    ys = pd.Series(np.asarray(ytr, float))

    def enc(keys_fit, y_fit, keys_apply):
        st = y_fit.groupby(keys_fit.to_numpy()).agg(["sum", "count"])
        m = (st["sum"] + prior * smooth) / (st["count"] + smooth)
        return keys_apply.map(m).astype(float).fillna(prior).to_numpy(np.float32)

    for c in cols:
        ktr, kva, kte = (X[c].astype(str) for X in (Xtr, Xva, Xte))
        te_tr = np.zeros(len(Xtr), np.float32)
        for a, b in KFold(n_inner, shuffle=True, random_state=SEED).split(Xtr):
            te_tr[b] = enc(ktr.iloc[a], ys.iloc[a], ktr.iloc[b])
        Xtr[f"te_{c}"], Xva[f"te_{c}"], Xte[f"te_{c}"] = te_tr, enc(ktr, ys, kva), enc(ktr, ys, kte)
    return Xtr, Xva, Xte


def _maybe_te(Xtr, ytr, Xva, Xte, p):
    """`te=true` : target encoding des identifiants à forte cardinalité et des clés composites `k_*` (v2).
    Les clés `k_*` sont toujours retirées ensuite : ce sont des codes arbitraires."""
    te, smooth = p.pop("te", None), p.pop("te_smooth", 20.0)
    keys = key_columns(Xtr)
    if te:
        cols = [c for c in (HIGH_CARD + ["fingerprint", "state_code"] + keys if te is True else te) if c in Xtr.columns]
        Xtr, Xva, Xte = target_encode(Xtr, ytr, Xva, Xte, cols, smooth=smooth)
    if keys:
        Xtr, Xva, Xte = (X.drop(columns=keys) for X in (Xtr, Xva, Xte))
    return Xtr, Xva, Xte


def lgbm_fn(Xtr, ytr, Xva, yva, Xte, p):
    import lightgbm as lgb

    p, seeds, n_est, patience = _pop(p)
    Xtr, Xva, Xte = _maybe_te(Xtr, ytr, Xva, Xte, p)
    base = dict(learning_rate=0.03, num_leaves=63, min_child_samples=40, colsample_bytree=0.5, subsample=0.8,
                subsample_freq=1, reg_lambda=5.0, cat_smooth=20, cat_l2=10, max_cat_to_onehot=8)
    base.update(p)
    pva, pte = 0.0, 0.0
    for s in range(seeds):
        m = lgb.LGBMClassifier(n_estimators=n_est, random_state=SEED + s, n_jobs=8, verbose=-1, **base)
        m.fit(Xtr, ytr, eval_X=Xva, eval_y=yva, eval_metric="auc", callbacks=[lgb.early_stopping(patience, verbose=False)])
        pva, pte = pva + m.predict_proba(Xva)[:, 1] / seeds, pte + m.predict_proba(Xte)[:, 1] / seeds
    return pva, pte


def xgb_fn(Xtr, ytr, Xva, yva, Xte, p):
    import xgboost as xgb

    p, seeds, n_est, patience = _pop(p)
    Xtr, Xva, Xte = _maybe_te(Xtr, ytr, Xva, Xte, p)
    base = dict(learning_rate=0.03, max_depth=7, min_child_weight=5, colsample_bytree=0.5, subsample=0.8,
                reg_lambda=5.0, max_cat_to_onehot=8, max_bin=256)
    base.update(p)
    pva, pte = 0.0, 0.0
    for s in range(seeds):
        m = xgb.XGBClassifier(n_estimators=n_est, tree_method="hist", enable_categorical=True, eval_metric="auc",
                              early_stopping_rounds=patience, random_state=SEED + s, n_jobs=8, **base)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        pva, pte = pva + m.predict_proba(Xva)[:, 1] / seeds, pte + m.predict_proba(Xte)[:, 1] / seeds
    return pva, pte


def cat_fn(Xtr, ytr, Xva, yva, Xte, p):
    from catboost import CatBoostClassifier, Pool

    p, seeds, n_est, patience = _pop(p, n_est=8000)
    p_verbose, threads, seed0 = p.pop("verbose", 0), p.pop("threads", 8), p.pop("seed0", 0)
    high = p.pop("high_card", True)  # comté, MSA, pseudo-quartier en catégories natives (statistiques ordonnées)
    keys = p.pop("keys", True)  # clés composites v2 en catégories natives
    cats = cat_columns(Xtr) + ([c for c in (HIGH_CARD if high is True else high or []) if c in Xtr.columns])
    if keys is True:
        cats += key_columns(Xtr)
    else:
        cats += list(keys or [])
        Xtr, Xva, Xte = (X.drop(columns=[c for c in key_columns(X) if c not in cats]) for X in (Xtr, Xva, Xte))

    def pool(X, y=None):
        X = X.copy()
        for c in cats:
            X[c] = X[c].astype(object).where(X[c].notna(), "NA").astype(str)
        return Pool(X, y, cat_features=cats)

    ptr, pv, pt = pool(Xtr, ytr), pool(Xva, yva), pool(Xte)
    base = dict(learning_rate=0.05, depth=6, l2_leaf_reg=5.0)
    base.update(p)
    pva, pte = 0.0, 0.0
    for s in range(seeds):
        m = CatBoostClassifier(iterations=n_est, eval_metric="AUC", od_type="Iter", od_wait=patience, random_seed=SEED + seed0 + s,
                               thread_count=threads, verbose=p_verbose, allow_writing_files=False, **base)
        m.fit(ptr, eval_set=pv, use_best_model=True)
        pva, pte = pva + m.predict_proba(pv)[:, 1] / seeds, pte + m.predict_proba(pt)[:, 1] / seeds
    return pva, pte


def hgb_fn(Xtr, ytr, Xva, yva, Xte, p):
    from sklearn.ensemble import HistGradientBoostingClassifier

    p, seeds, n_est, _ = _pop(p, n_est=1500)
    Xtr, Xva, Xte = _maybe_te(Xtr, ytr, Xva, Xte, p)
    cats = [c for c in cat_columns(Xtr) if Xtr[c].cat.categories.size <= 255]
    drop = [c for c in cat_columns(Xtr) if c not in cats]
    Xtr, Xva, Xte = (X.drop(columns=drop) for X in (Xtr, Xva, Xte))
    m = HistGradientBoostingClassifier(max_iter=n_est, learning_rate=0.05, categorical_features=cats, early_stopping=True,
                                       validation_fraction=0.1, n_iter_no_change=50, random_state=SEED, **p)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]


def _dense(Xtr, Xva, Xte, onehot_max=60):
    """Matrice dense pour modèles linéaires / forêts : numériques transformées en quantiles gaussiens (médiane pour
    les manquants + indicatrices), catégories en one-hot (modalités rares regroupées)."""
    from sklearn.preprocessing import QuantileTransformer

    cats = cat_columns(Xtr)
    num = [c for c in Xtr.columns if c not in cats]
    nan_cols = [c for c in num if Xtr[c].isna().any()]
    med = Xtr[num].median()
    qt = QuantileTransformer(n_quantiles=500, output_distribution="normal", random_state=SEED)
    fin = lambda X: X[num].astype(float).replace([np.inf, -np.inf], np.nan).fillna(med)  # noqa: E731
    qt.fit(fin(Xtr))
    keep = {c: Xtr[c].value_counts().pipe(lambda v: v.index[v >= 30][:onehot_max]) for c in cats}

    def enc(X):
        parts = [qt.transform(fin(X)), X[nan_cols].isna().to_numpy(float)]
        for c in cats:
            parts.append((X[c].astype(str).to_numpy()[:, None] == np.asarray(keep[c].astype(str))[None, :]).astype(float))
        return np.hstack(parts).astype(np.float32)

    return enc(Xtr), enc(Xva), enc(Xte)


def logreg_fn(Xtr, ytr, Xva, yva, Xte, p):
    from sklearn.linear_model import LogisticRegression

    p = dict(p); p.pop("seeds", None)
    Xtr, Xva, Xte = _maybe_te(Xtr, ytr, Xva, Xte, p)
    A, B, C = _dense(Xtr, Xva, Xte)
    m = LogisticRegression(C=p.get("C", 0.1), max_iter=2000).fit(A, ytr)
    return m.predict_proba(B)[:, 1], m.predict_proba(C)[:, 1]


def _forest(cls):
    def fn(Xtr, ytr, Xva, yva, Xte, p):
        p = dict(p); p.pop("seeds", None)
        Xtr, Xva, Xte = _maybe_te(Xtr, ytr, Xva, Xte, p)
        A, B, C = _dense(Xtr, Xva, Xte)
        base = dict(n_estimators=500, min_samples_leaf=3, max_features=0.3, n_jobs=8, random_state=SEED)
        base.update(p)
        m = cls(**base).fit(A, ytr)
        return m.predict_proba(B)[:, 1], m.predict_proba(C)[:, 1]
    return fn


def _rf(*a):
    from sklearn.ensemble import RandomForestClassifier
    return _forest(RandomForestClassifier)(*a)


def _et(*a):
    from sklearn.ensemble import ExtraTreesClassifier
    return _forest(ExtraTreesClassifier)(*a)


def _nn_frame(Xtr, Xva, Xte, min_count=20):
    """Entrée des réseaux pytabkit : numériques à queue lourde en log signé, manquants -> médiane + indicatrice,
    catégories gardées (pytabkit les plonge lui-même), modalités rares regroupées."""
    cats = cat_columns(Xtr)
    num = [c for c in Xtr.columns if c not in cats]
    nan_cols = [c for c in num if Xtr[c].isna().any()]
    heavy = [c for c in num if Xtr[c].abs().max() > 1000]

    def pre(X):
        X = X.copy()
        X[num] = X[num].astype(float).replace([np.inf, -np.inf], np.nan)
        X[heavy] = np.sign(X[heavy]) * np.log1p(X[heavy].abs())
        return X

    Xtr, Xva, Xte = pre(Xtr), pre(Xva), pre(Xte)
    med, lo, hi = Xtr[num].median(), Xtr[num].quantile(0.001), Xtr[num].quantile(0.999)
    keep = {c: set(Xtr[c].astype(str).value_counts().pipe(lambda v: v.index[v >= min_count])) | {"rare"} for c in cats}

    def post(X):
        ind = {f"na_{c}": X[c].isna().astype(np.float32) for c in nan_cols}
        X[num] = X[num].fillna(med).clip(lo, hi, axis=1).astype(np.float32)
        for c in cats:
            s = X[c].astype(str)
            X[c] = pd.Categorical(s.where(s.isin(keep[c]), "rare"), categories=sorted(keep[c]))
        return pd.concat([X, pd.DataFrame(ind, index=X.index)], axis=1)

    return post(Xtr), post(Xva), post(Xte)


def _pytabkit(cls_name):
    def fn(Xtr, ytr, Xva, yva, Xte, p):
        import pytabkit
        import torch

        p = dict(p)
        seeds, threads, seed0 = p.pop("seeds", 1), p.pop("threads", 4), p.pop("seed0", 0)
        torch.set_num_threads(threads)
        p.setdefault("device", "mps" if torch.backends.mps.is_available() else "cpu")
        Xtr, Xva, Xte = _maybe_te(Xtr, ytr, Xva, Xte, p)
        if "tract_id" in Xtr.columns:  # identifiant arbitraire : inutilisable comme numérique par un réseau
            Xtr, Xva, Xte = (X.drop(columns=["tract_id", "county_code", "derived_msa-md"]) for X in (Xtr, Xva, Xte))
        Xtr, Xva, Xte = _nn_frame(Xtr, Xva, Xte)
        pva, pte = 0.0, 0.0
        for s in range(seeds):
            m = getattr(pytabkit, cls_name)(random_state=SEED + seed0 + s, n_threads=threads, val_metric_name="1-auc_ovr", **p)
            m.fit(Xtr, np.asarray(ytr), X_val=Xva, y_val=np.asarray(yva))
            pva, pte = pva + m.predict_proba(Xva)[:, 1] / seeds, pte + m.predict_proba(Xte)[:, 1] / seeds
        return pva, pte
    return fn


MODELS = {
    "lgbm": lgbm_fn, "xgb": xgb_fn, "cat": cat_fn, "hgb": hgb_fn, "logreg": logreg_fn, "rf": _rf, "et": _et,
    "realmlp": _pytabkit("RealMLP_TD_Classifier"), "tabm": _pytabkit("TabM_D_Classifier"),
}
