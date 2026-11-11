# Classification : demandes de prêt immobilier (HMDA)

Prédire si une demande de prêt immobilier est accordée, à partir de 56 variables. Compétition Kaggle
[`demandes-de-pret-immobilier-aux-etats-unis`](https://www.kaggle.com/competitions/demandes-de-pret-immobilier-aux-etats-unis),
métrique AUC ROC.

## Installation

Il faut [uv](https://docs.astral.sh/uv/) et un token Kaggle (sur Mac, aussi `brew install libomp`).

```bash
uv sync
```

Les données sont téléchargées dans `data/` au premier lancement.

## Commandes

```bash
uv run main.py train --model cat --name cat_v3 --features v3 \
    --set learning_rate=0.08 'high_card=["county_code","derived_msa-md"]'    # un modèle, 5 plis
uv run main.py train --model lgbm --name essai --folds 0                     # essai sur un pli, rien n'est sauvegardé
uv run python -m hmda.tune --model lgbm --name lgbm_tuned --trials 30 --set te=true   # réglage Optuna
uv run main.py ensemble --models cat_v3 lgbm_v3_tuned realmlp_v2 --name ens  # ensemble (sans --models : tous)
uv run main.py report                                                        # tableau des scores
```

Modèles disponibles : `lgbm`, `xgb`, `cat`, `hgb`, `logreg`, `rf`, `et`, `realmlp`, `tabm`.
Jeux de variables : `raw`, `v1`, `v2`, `v3`, `v4` (décrits dans `src/hmda/features.py`).

Chaque entraînement écrit ses prédictions OOF dans `outputs/oof/`, ses prédictions test dans `outputs/preds/`, une
soumission dans `outputs/submissions/` et une ligne dans `outputs/experiments.csv`.

## Organisation du code

| Dossier | Contenu |
|---|---|
| `main.py` | point d'entrée |
| `src/hmda/features.py` | prétraitement (aucune imputation) et variables construites |
| `src/hmda/cv.py` | plis fixes (`data/folds.csv`), boucle de validation croisée, journal des scores |
| `src/hmda/models.py` | modèles, target encoding appris dans le pli |
| `src/hmda/tune.py` | réglage Optuna |
| `src/hmda/ensemble.py` | sélection gloutonne et stacking logistique, évalués en validation imbriquée |
| `outputs/` | prédictions, hyperparamètres, `experiments.csv` |
| `archive/` | analyse exploratoire et expériences hors pipeline |
