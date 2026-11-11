"""Réglage Optuna (TPE) des modèles boostés, sur les 5 plis fixes. Le meilleur jeu est écrit dans
`outputs/params/<nom>.json` ; tous les essais sont gardés dans `outputs/params/<nom>_trials.csv`.

    uv run python -m hmda.tune --model lgbm --name lgbm_tuned --trials 40 --features v2 --set te=true
"""

import argparse
import json

import optuna

from hmda.config import PARAM_DIR, SEED, TARGET
from hmda.cv import make_folds, run_cv
from hmda.data import load_data
from hmda.features import build_features

SPACES = {
    "lgbm": lambda t: dict(
        learning_rate=0.05, num_leaves=t.suggest_int("num_leaves", 8, 128, log=True),
        min_child_samples=t.suggest_int("min_child_samples", 10, 300, log=True),
        colsample_bytree=t.suggest_float("colsample_bytree", 0.15, 0.8),
        subsample=t.suggest_float("subsample", 0.5, 1.0),
        reg_lambda=t.suggest_float("reg_lambda", 0.1, 100, log=True),
        reg_alpha=t.suggest_float("reg_alpha", 1e-3, 10, log=True),
        min_split_gain=t.suggest_float("min_split_gain", 0.0, 0.5),
        cat_smooth=t.suggest_float("cat_smooth", 5, 200, log=True), cat_l2=t.suggest_float("cat_l2", 1, 100, log=True),
        max_cat_to_onehot=t.suggest_categorical("max_cat_to_onehot", [4, 8, 32, 512]),
        path_smooth=t.suggest_float("path_smooth", 0.0, 50.0),
    ),
    "xgb": lambda t: dict(
        learning_rate=0.05, max_depth=t.suggest_int("max_depth", 3, 10),
        min_child_weight=t.suggest_float("min_child_weight", 1, 100, log=True),
        colsample_bytree=t.suggest_float("colsample_bytree", 0.15, 0.8),
        colsample_bynode=t.suggest_float("colsample_bynode", 0.4, 1.0),
        subsample=t.suggest_float("subsample", 0.5, 1.0), reg_lambda=t.suggest_float("reg_lambda", 0.1, 100, log=True),
        reg_alpha=t.suggest_float("reg_alpha", 1e-3, 10, log=True), gamma=t.suggest_float("gamma", 0.0, 5.0),
        max_cat_to_onehot=t.suggest_categorical("max_cat_to_onehot", [4, 8, 32, 512]),
    ),
    "cat": lambda t: dict(
        learning_rate=0.08, depth=t.suggest_int("depth", 4, 8), l2_leaf_reg=t.suggest_float("l2_leaf_reg", 1, 50, log=True),
        random_strength=t.suggest_float("random_strength", 0.1, 10, log=True),
        bagging_temperature=t.suggest_float("bagging_temperature", 0.0, 2.0),
        rsm=t.suggest_float("rsm", 0.1, 0.6),
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--features", default="v2")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    fixed = {k: json.loads(v) for k, v in (s.split("=", 1) for s in a.set)}
    train, test, _ = load_data()
    F_tr, F_te = build_features(train, test, a.features)
    folds, y = make_folds(train), train[TARGET].to_numpy()

    def objective(t):
        params = {**SPACES[a.model](t), **fixed, "patience": 100}
        r = run_cv("tune", a.model, y, F_tr, F_te, folds, params=params, save=False, verbose=False)
        t.set_user_attr("fold_auc", r["fold_auc"])
        return r["cv_auc"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=a.trials, callbacks=[lambda s, t: print(f"essai {t.number} : {t.value:.5f} (meilleur {s.best_value:.5f}) {t.params}", flush=True)])
    study.trials_dataframe().to_csv(PARAM_DIR / f"{a.name}_trials.csv", index=False)
    best = {**study.best_params, **fixed}
    (PARAM_DIR / f"{a.name}.json").write_text(json.dumps(best, indent=2))
    print("Meilleur :", study.best_value, best)


if __name__ == "__main__":
    main()
