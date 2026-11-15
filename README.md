# Classification : demandes de prêt immobilier (HMDA)

Prédire si une demande de prêt immobilier est accordée, à partir de 56 variables. Compétition Kaggle
[`demandes-de-pret-immobilier-aux-etats-unis`](https://www.kaggle.com/competitions/demandes-de-pret-immobilier-aux-etats-unis),
métrique AUC ROC.

**Meilleure soumission : AUC 0,92248 sur Kaggle (classement public), 0,92282 en validation croisée.** Elle combine
13 modèles (CatBoost, LightGBM, XGBoost, réseaux TabM et RealMLP, régression logistique) par un stacking
logistique.

## Installation

Il faut [uv](https://docs.astral.sh/uv/) et un compte Kaggle (sur Mac, aussi `brew install libomp`).

1. Sur Kaggle, rejoindre la compétition (*Join Competition*) et créer un token sur
   https://www.kaggle.com/settings/api.
2. Installer le projet :

   ```bash
   uv sync
   ```

Les données sont téléchargées automatiquement dans `data/` au premier lancement ; le token Kaggle est demandé à ce
moment-là.

## Refaire la meilleure soumission

Les prédictions des 13 modèles sont déjà dans le dépôt (`outputs/oof/` et `outputs/preds/`). Une commande suffit
(moins d'une minute) :

```bash
uv run main.py ensemble
```

La soumission est écrite dans `outputs/submissions/ensemble_stack.csv` : c'est le fichier envoyé à Kaggle
(0,92248). La commande affiche l'AUC de chaque modèle et celle de l'ensemble en validation imbriquée ; la ligne
attendue est `Stacking logistique C=0.01 : 0.92282`.

### Tout réentraîner

Pour réentraîner les 13 modèles au lieu d'utiliser leurs prédictions (environ 6 heures sur un Mac M-series), lancer
d'abord :

```bash
uv run main.py reproduce
```

puis `uv run main.py ensemble`. Les commandes de chaque modèle sont listées dans
[`src/hmda/recipes.py`](src/hmda/recipes.py) ; `uv run main.py reproduce --dry-run` les affiche sans les lancer, et
`uv run main.py reproduce lgbm_v1 cat_v3` ne relance que les modèles nommés.

Pour contrôler un modèle sans rien écraser, `--check` le réentraîne et compare ses prédictions à celles du dépôt :

```bash
uv run main.py reproduce lgbm_v1 --check            # les 5 plis
uv run main.py reproduce cat_v3 --check --folds 0   # un seul pli (plus rapide)
```

Sur la machine d'origine (Mac M-series), ce contrôle redonne exactement les prédictions du dépôt pour les 9 modèles
contrôlés : `lgbm_raw`, `lgbm_v1`, `lgbm_v2_te`, `lgbm_v3_tuned`, `xgb_v3_te` et `logreg_v3` sur les 5 plis,
`cat_v1`, `cat_v3` et `tabm_v3` sur un pli. Sur une autre machine, de petits écarts sont possibles, surtout pour
les réseaux de neurones.

## Autres commandes

```bash
uv run main.py train --model cat --name cat_v3 --features v3 \
    --set learning_rate=0.08 'high_card=["county_code","derived_msa-md"]'    # un modèle, 5 plis
uv run main.py train --model lgbm --name essai --folds 0                     # essai sur un pli, rien n'est sauvegardé
uv run python -m hmda.tune --model lgbm --name lgbm_tuned --trials 30 --set te=true   # réglage Optuna
uv run main.py ensemble --models cat_v3 lgbm_v3_tuned realmlp_v2 --name ens  # ensemble d'autres modèles (--all : tous)
uv run main.py report                                                        # tableau des scores
```

Modèles disponibles : `lgbm`, `xgb`, `cat`, `hgb`, `logreg`, `rf`, `et`, `realmlp`, `tabm`.
Jeux de variables : `raw`, `v1`, `v2`, `v3`, `v4` (décrits dans `src/hmda/features.py`).

Chaque entraînement écrit ses prédictions OOF dans `outputs/oof/`, ses prédictions test dans `outputs/preds/`, une
soumission dans `outputs/submissions/` et une ligne dans `outputs/experiments.csv`.

## Organisation du code

| Dossier | Contenu |
|---|---|
| `main.py` | point d'entrée (`uv run main.py --help` pour toutes les commandes) |
| `src/hmda/features.py` | prétraitement (aucune imputation) et variables construites |
| `src/hmda/cv.py` | plis fixes (`data/folds.csv`), boucle de validation croisée, journal des scores |
| `src/hmda/models.py` | modèles, target encoding appris dans le pli |
| `src/hmda/recipes.py` | recettes des 13 modèles de la meilleure soumission |
| `src/hmda/tune.py` | réglage Optuna |
| `src/hmda/ensemble.py` | sélection gloutonne et stacking logistique, évalués en validation imbriquée |
| `outputs/` | prédictions, hyperparamètres, `experiments.csv` |
| `archive/` | analyse exploratoire et expériences hors pipeline |

## Résultats

| Étape | AUC en validation croisée |
|---|---|
| LightGBM sur les variables brutes, sans imputation | 0,91511 |
| LightGBM avec les variables construites (v1) | 0,91809 |
| Meilleur modèle seul (CatBoost, variables v4) | 0,92133 |
| **Stacking des 13 modèles (soumission finale)** | **0,92282** (Kaggle : 0,92248) |

La validation croisée est faite sur 5 plis fixes (`data/folds.csv`).
