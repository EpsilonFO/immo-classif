"""Point d'entrée : `uv run main.py train --model lgbm --name lgbm_v1` ; `uv run main.py ensemble` ; `report`."""

import argparse
import json

import numpy as np

from hmda.config import ID, SUB_DIR, TARGET
from hmda.cv import load_params, make_folds, read_log, run_cv
from hmda.data import load_data
from hmda.features import build_features


def _parse(v: str):
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def write_submission(name, pred, sample):
    sub = sample.copy()
    sub[TARGET] = pred
    path = SUB_DIR / f"{name}.csv"
    sub.to_csv(path, index=False)
    return path


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train")
    t.add_argument("--model", required=True)
    t.add_argument("--name")
    t.add_argument("--features", default="v1")
    t.add_argument("--seeds", type=int, default=1)
    t.add_argument("--params", help="nom d'un fichier outputs/params/<nom>.json")
    t.add_argument("--set", nargs="*", default=[], help="hyperparamètres cle=valeur")
    t.add_argument("--folds", type=int, nargs="*", help="sous-ensemble de plis (essai rapide, rien n'est sauvegardé)")
    e = sp.add_parser("ensemble")
    e.add_argument("--models", nargs="*")
    e.add_argument("--name", default="ensemble")
    sp.add_parser("report")
    a = ap.parse_args()

    if a.cmd == "report":
        print(read_log().drop(columns=["params", "fold_auc"]).sort_values("cv_auc", ascending=False).to_string(index=False))
        return
    train, test, sample = load_data()
    assert (sample[ID] == test[ID]).all()
    folds, y = make_folds(train), train[TARGET].to_numpy()
    if a.cmd == "train":
        F_tr, F_te = build_features(train, test, a.features)
        params = {**(load_params(a.params) if a.params else {}), **{k: _parse(v) for k, v in (s.split("=", 1) for s in a.set)}}
        name = a.name or a.model
        r = run_cv(name, a.model, y, F_tr, F_te, folds, features=a.features, seeds=a.seeds, params=params, only_folds=a.folds)
        if a.folds is None:
            print("Soumission :", write_submission(name, r["pred"], sample))
    else:
        from hmda.ensemble import run_ensemble

        run_ensemble(a.models, y, folds, sample, a.name)


if __name__ == "__main__":
    main()
