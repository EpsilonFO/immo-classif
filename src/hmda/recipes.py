"""Recettes des 13 modèles de la meilleure soumission (`ens5_stack` : AUC 0,92282 en validation croisée, 0,92248 sur
le classement public Kaggle).

Chaque recette donne les arguments de `main.py train` qui ont produit `outputs/oof/<nom>.npy` et
`outputs/preds/<nom>.npy` ; `main.py reproduce` les relance (un processus par modèle) et `main.py ensemble` combine
les 13 modèles.

L'ordre est celui des colonnes du stacking de la soumission.
"""

_HIGH_CARD = """'high_card=["county_code","derived_msa-md"]'"""  # comté et MSA en catégories natives de CatBoost

RECIPES: dict[str, str] = {
    # -- CatBoost : meilleurs modèles seuls
    "cat_v4": f"--model cat --features v4 --set learning_rate=0.08 {_HIGH_CARD}",
    "cat_v3": f"--model cat --features v3 --set learning_rate=0.08 threads=6 {_HIGH_CARD}",
    "cat_v3_d8": f"--model cat --features v3 --set learning_rate=0.06 depth=8 l2_leaf_reg=10 {_HIGH_CARD}",
    "cat_v1": f"--model cat --features v1 --set learning_rate=0.08 {_HIGH_CARD}",
    # -- LightGBM et XGBoost (`te=true` : target encoding appris dans le pli)
    "lgbm_v3_tuned": "--model lgbm --features v3 --seeds 3 --params lgbm_tuned --set learning_rate=0.02",
    "lgbm_v2_te": "--model lgbm --features v2 --set te=true",
    "lgbm_v1": "--model lgbm --features v1",
    "xgb_v3_te": "--model xgb --features v3 --set te=true max_depth=6 min_child_weight=10 colsample_bytree=0.4",
    # -- Réseaux de neurones (pytabkit ; GPU du Mac s'il est disponible, sinon CPU)
    "realmlp_v2": "--model realmlp --features v2 --set te=true threads=2",
    "tabm_v3": "--model tabm --features v3 --set te=true threads=2",
    "tabm_v3_s1": "--model tabm --features v3 --set te=true threads=2 seed0=1",
    # -- Modèles faibles seuls mais différents des autres
    "logreg_v3": "--model logreg --features v3 --set te=true",
    "lgbm_raw": "--model lgbm --features raw",
}

ENSEMBLE: list[str] = list(RECIPES)
