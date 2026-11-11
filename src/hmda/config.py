from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMPETITION = "demandes-de-pret-immobilier-aux-etats-unis"
DATA_DIR = ROOT / "data"
OUT = ROOT / "outputs"
OOF_DIR, PRED_DIR, PARAM_DIR, SUB_DIR = OUT / "oof", OUT / "preds", OUT / "params", OUT / "submissions"
FOLDS_FILE = DATA_DIR / "folds.csv"
EXPERIMENTS_FILE = OUT / "experiments.csv"
ID, TARGET = "id", "accepted"
SEED, N_FOLDS = 42, 5
for d in (OOF_DIR, PRED_DIR, PARAM_DIR, SUB_DIR):
    d.mkdir(parents=True, exist_ok=True)
