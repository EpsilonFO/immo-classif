"""Ensemble (journal, étape « ensemble ») : moyenne des rangs, poids optimisés, stacking logistique sur les logits.

Les AUC annoncées sont honnêtes : poids ou méta-modèle appris sur 4 plis d'OOF, évalués sur le 5e."""

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from hmda.config import N_FOLDS, OOF_DIR, PRED_DIR, SUB_DIR, TARGET
from hmda.cv import log_experiment, read_log, score_row


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def load_predictions(names):
    return (np.column_stack([np.load(OOF_DIR / f"{n}.npy") for n in names]),
            np.column_stack([np.load(PRED_DIR / f"{n}.npy") for n in names]))


def hill_climb(Z, y, n_iter=60):
    """Sélection gloutonne avec remise (Caruana) maximisant l'AUC de la moyenne des logits. Retourne les poids."""
    counts, cur, best = np.zeros(Z.shape[1]), np.zeros(len(y)), -1.0
    for it in range(n_iter):
        scores = [roc_auc_score(y, (cur + Z[:, j]) / (it + 1)) for j in range(Z.shape[1])]
        j = int(np.argmax(scores))
        if scores[j] <= best + 1e-7 and it >= 5:
            break
        best, cur = scores[j], cur + Z[:, j]
        counts[j] += 1
    return counts / counts.sum()


def honest(fit_predict, Z, y, folds):
    out = np.zeros(len(y))
    for k in range(N_FOLDS):
        va = folds == k
        out[va] = fit_predict(Z[~va], y[~va], Z[va])
    return out


def run_ensemble(names, y, folds, sample, out_name="ensemble"):
    if not names:
        log = read_log()
        names = [n for n in log.loc[log["model"] != "ensemble", "name"] if (OOF_DIR / f"{n}.npy").exists()]
    O, T = load_predictions(names)
    Z, Zt = logit(O), logit(T)
    auc = lambda p: roc_auc_score(y, p)  # noqa: E731
    cv = lambda p: float(np.mean([roc_auc_score(y[folds == k], p[folds == k]) for k in range(N_FOLDS)]))  # noqa: E731
    print("AUC CV individuelle :")
    for i in np.argsort([-cv(O[:, i]) for i in range(len(names))]):
        print(f"  {names[i]:28s} {cv(O[:, i]):.5f}")
    R = np.column_stack([rankdata(O[:, i]) for i in range(O.shape[1])])
    print(f"Moyenne simple des logits : {cv(Z.mean(1)):.5f} ; moyenne des rangs : {cv(R.mean(1)):.5f}")

    res = {}
    # 1. poids gloutons
    hc = honest(lambda A, b, C: C @ hill_climb(A, b), Z, y, folds)
    w = hill_climb(Z, y)
    res["hc"] = (hc, Zt @ w)
    print(f"Sélection gloutonne : {cv(hc):.5f} ; poids = { {n: round(float(x), 3) for n, x in zip(names, w) if x > 0} }")
    # 2. stacking logistique sur les logits
    for C in (0.01, 0.1, 1.0):
        f = lambda A, b, B, C=C: LogisticRegression(C=C, max_iter=2000).fit(A, b).decision_function(B)  # noqa: E731
        st = honest(f, Z, y, folds)
        print(f"Stacking logistique C={C} : {cv(st):.5f} (OOF globale {auc(st):.5f})")
        if "stack" not in res or cv(st) > cv(res["stack"][0]):
            meta = LogisticRegression(C=C, max_iter=2000).fit(Z, y)
            res["stack"] = (st, meta.decision_function(Zt))
            coefs = {n: round(float(c), 3) for n, c in zip(names, meta.coef_[0])}
    print("  coefficients :", coefs)
    for kind, (o, p) in res.items():
        name = f"{out_name}_{kind}"
        o, p = 1 / (1 + np.exp(-o)), 1 / (1 + np.exp(-p))
        np.save(OOF_DIR / f"{name}.npy", o)
        np.save(PRED_DIR / f"{name}.npy", p)
        log_experiment(score_row(name, "ensemble", "-", 1, len(names), y, o, folds, 0.0, {"models": names}))
        sub = sample.copy()
        sub[TARGET] = p
        sub.to_csv(SUB_DIR / f"{name}.csv", index=False)
    return res
