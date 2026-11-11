"""Protocole de validation unique (journal, étape 2) : 5 plis stratifiés fixes, OOF + prédictions test sauvegardées,
journal `outputs/experiments.csv`."""

import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from hmda.config import EXPERIMENTS_FILE, FOLDS_FILE, ID, N_FOLDS, OOF_DIR, PARAM_DIR, PRED_DIR, SEED, TARGET

LOG_COLUMNS = ["name", "model", "features", "seeds", "n_features", "cv_auc", "cv_std", "oof_auc", "logloss",
               "fold_auc", "public_score", "minutes", "date", "params"]


def make_folds(train: pd.DataFrame) -> np.ndarray:
    if FOLDS_FILE.exists():
        saved = pd.read_csv(FOLDS_FILE).set_index(ID)["fold"]
        if set(saved.index) == set(train[ID]):
            return saved.loc[train[ID]].to_numpy()
    folds = np.zeros(len(train), dtype=int)
    for k, (_, va) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(train, train[TARGET])):
        folds[va] = k
    pd.DataFrame({ID: train[ID], "fold": folds}).to_csv(FOLDS_FILE, index=False)
    return folds


def read_log() -> pd.DataFrame:
    return pd.read_csv(EXPERIMENTS_FILE) if EXPERIMENTS_FILE.exists() else pd.DataFrame(columns=LOG_COLUMNS)


def log_experiment(row: dict) -> None:
    log = read_log()
    log = log[log["name"] != row["name"]]
    new = pd.DataFrame([{c: row.get(c, np.nan) for c in LOG_COLUMNS}])
    (new if log.empty else pd.concat([log, new], ignore_index=True))[LOG_COLUMNS].to_csv(EXPERIMENTS_FILE, index=False)


def record_public_score(name: str, score: float) -> None:
    log = read_log()
    log.loc[log["name"] == name, "public_score"] = score
    log.to_csv(EXPERIMENTS_FILE, index=False)


def score_row(name, model, features, seeds, n_features, y, oof, folds, minutes, params=None) -> dict:
    fold_auc = [roc_auc_score(y[folds == k], oof[folds == k]) for k in range(N_FOLDS)]
    return {
        "name": name, "model": model, "features": features, "seeds": seeds, "n_features": n_features,
        "cv_auc": float(np.mean(fold_auc)), "cv_std": float(np.std(fold_auc)), "oof_auc": float(roc_auc_score(y, oof)),
        "logloss": float(log_loss(y, np.clip(oof, 1e-6, 1 - 1e-6))), "fold_auc": json.dumps([round(s, 5) for s in fold_auc]),
        "minutes": round(minutes, 1), "date": time.strftime("%Y-%m-%d %H:%M"), "params": json.dumps(params or {}),
    }


def load_params(name: str) -> dict:
    f = PARAM_DIR / f"{name}.json"
    return json.loads(f.read_text()) if f.exists() else {}


def run_cv(name, model, y, F_tr, F_te, folds, *, features="v1", seeds=1, params=None, only_folds=None, save=True,
           verbose=True) -> dict:
    """Entraîne `model` sur chaque pli ; garde les prédictions OOF et la moyenne des prédictions test des 5 modèles."""
    from hmda.models import MODELS

    fit = MODELS[model]
    oof, pred = np.full(len(y), np.nan), np.zeros(len(F_te))
    ran = list(range(N_FOLDS)) if only_folds is None else list(only_folds)
    t0 = time.time()
    for k in ran:
        tr, va = np.where(folds != k)[0], np.where(folds == k)[0]
        Xtr, Xva = F_tr.iloc[tr].reset_index(drop=True), F_tr.iloc[va].reset_index(drop=True)
        p_va, p_te = fit(Xtr, y[tr], Xva, y[va], F_te, {**(params or {}), "seeds": seeds})
        oof[va], pred = p_va, pred + p_te / len(ran)
        if verbose:
            print(f"  [{name}] pli {k} : AUC {roc_auc_score(y[va], p_va):.5f}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    if only_folds is not None:
        aucs = [roc_auc_score(y[folds == k], oof[folds == k]) for k in ran]
        return {"cv_auc": float(np.mean(aucs)), "oof": oof, "pred": pred}
    row = score_row(name, model, features, seeds, F_tr.shape[1], y, oof, folds, (time.time() - t0) / 60, params)
    if verbose:
        print(f"  [{name}] CV AUC = {row['cv_auc']:.5f} ± {row['cv_std']:.5f}  (OOF {row['oof_auc']:.5f}, logloss {row['logloss']:.4f})", flush=True)
    if save:
        np.save(OOF_DIR / f"{name}.npy", oof)
        np.save(PRED_DIR / f"{name}.npy", pred)
        log_experiment(row)
    row["oof"], row["pred"] = oof, pred
    return row
