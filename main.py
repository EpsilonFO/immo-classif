"""Point d'entrée : uv run main.py <commande> --help

    ensemble   combine les 13 modèles de recipes.py (déjà entraînés) : c'est la meilleure soumission
    reproduce  réentraîne ces 13 modèles (recipes.py)
    train      entraîne un modèle en validation croisée 5 plis (OOF + test + fichier de soumission)
    report     tableau des scores
"""

import argparse
import json

import numpy as np

from hmda.config import ID, OOF_DIR, SUB_DIR, TARGET
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


def _compare_with_saved(name, oof, y):
    """Compare une OOF recalculée à celle de outputs/oof/<name>.npy, sur les lignes recalculées."""
    from sklearn.metrics import roc_auc_score

    ref = np.load(OOF_DIR / f"{name}.npy")
    rows = ~np.isnan(oof)
    diff = float(np.abs(oof[rows] - ref[rows]).max())
    print(f"  [{name}] contrôle sur {rows.sum()} lignes : écart max {diff:.3g} ; "
          f"AUC {roc_auc_score(y[rows], oof[rows]):.5f} (recalculée) contre {roc_auc_score(y[rows], ref[rows]):.5f} (sauvegardée)")
    print(f"  [{name}] {'REPRODUIT' if diff < 1e-6 else 'DIFFÉRENT'}")


def cmd_reproduce(a):
    """Relance les recettes de recipes.py, un processus par modèle (PyTorch et LightGBM / CatBoost / XGBoost ne
    cohabitent pas toujours dans un même processus sur macOS : conflit OpenMP)."""
    import shlex
    import subprocess
    import sys

    from hmda.config import ROOT
    from hmda.recipes import RECIPES

    names = a.names or list(RECIPES)
    unknown = sorted(set(names) - set(RECIPES))
    if unknown:
        raise SystemExit(f"recettes inconnues : {unknown}")
    for n in names:
        cmd = [sys.executable, "main.py", "train", *shlex.split(RECIPES[n]), "--name", n]
        if a.check:
            cmd.append("--check")
        if a.folds:
            cmd += ["--folds", *map(str, a.folds)]
        print("$", shlex.join(cmd[1:]), flush=True)
        if not a.dry_run:
            subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train", help="validation croisée 5 plis d'un modèle")
    t.add_argument("--model", required=True)
    t.add_argument("--name")
    t.add_argument("--features", default="v1")
    t.add_argument("--seeds", type=int, default=1)
    t.add_argument("--params", help="nom d'un fichier outputs/params/<nom>.json")
    t.add_argument("--set", nargs="*", default=[], help="hyperparamètres cle=valeur")
    t.add_argument("--folds", type=int, nargs="*", help="sous-ensemble de plis (essai rapide, rien n'est sauvegardé)")
    t.add_argument("--check", action="store_true", help="ne rien écrire : comparer l'OOF obtenue à outputs/oof/<nom>.npy")
    rp = sp.add_parser("reproduce", help="relancer les recettes des 13 modèles de l'ensemble (recipes.py)")
    rp.add_argument("names", nargs="*", help="recettes à relancer (défaut : toutes)")
    rp.add_argument("--dry-run", action="store_true", help="afficher les commandes sans les lancer")
    rp.add_argument("--check", action="store_true", help="comparer aux OOF sauvegardées sans rien écrire")
    rp.add_argument("--folds", type=int, nargs="+", metavar="K", help="seulement ces plis (rien n'est sauvegardé)")
    e = sp.add_parser("ensemble", help="sélection gloutonne + stacking logistique")
    e.add_argument("--models", nargs="*", help="noms d'expériences (défaut : les 13 de recipes.ENSEMBLE)")
    e.add_argument("--all", action="store_true", help="tous les modèles du journal au lieu des 13 de recipes.ENSEMBLE")
    e.add_argument("--name", default="ensemble")
    sp.add_parser("report", help="tableau des scores")
    a = ap.parse_args()

    if a.cmd == "report":
        print(read_log().drop(columns=["params", "fold_auc"]).sort_values("cv_auc", ascending=False).to_string(index=False))
        return
    if a.cmd == "reproduce":
        cmd_reproduce(a)
        return
    train, test, sample = load_data()
    assert (sample[ID] == test[ID]).all()
    folds, y = make_folds(train), train[TARGET].to_numpy()
    if a.cmd == "train":
        F_tr, F_te = build_features(train, test, a.features)
        params = {**(load_params(a.params) if a.params else {}), **{k: _parse(v) for k, v in (s.split("=", 1) for s in a.set)}}
        name = a.name or a.model
        r = run_cv(name, a.model, y, F_tr, F_te, folds, features=a.features, seeds=a.seeds, params=params,
                   only_folds=a.folds, save=not a.check)
        if a.check:
            _compare_with_saved(name, r["oof"], y)
        elif a.folds is None:
            print("Soumission :", write_submission(name, r["pred"], sample))
    else:
        from hmda.ensemble import run_ensemble
        from hmda.recipes import ENSEMBLE

        names = None if a.all else (a.models or ENSEMBLE)
        print(f"Ensemble de {len(names)} modèles" if names else "Ensemble de tous les modèles du journal")
        run_ensemble(names, y, folds, sample, a.name)
        print(f"Soumission (stacking) : {SUB_DIR / f'{a.name}_stack.csv'}")


if __name__ == "__main__":
    main()
