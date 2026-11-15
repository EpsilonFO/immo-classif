import os
from pathlib import Path

from hmda.config import COMPETITION, DATA_DIR

os.environ.setdefault("KAGGLEHUB_CACHE", str(DATA_DIR))

import pandas as pd  # noqa: E402


def ensure_login() -> None:
    """Demande le token Kaggle une seule fois : il est ensuite sauvegardé dans ~/.kaggle/."""
    import kagglehub

    try:
        kagglehub.whoami(verbose=False)
    except Exception:
        print("Aucun identifiant Kaggle trouvé : création d'un token sur https://www.kaggle.com/settings/api")
        kagglehub.login()


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Renvoie train, test, sample_submission. Les fichiers déjà téléchargés sont lus sans réseau ; sinon la
    compétition est téléchargée avec kagglehub (le token Kaggle est demandé s'il est absent)."""
    path = DATA_DIR / "competitions" / COMPETITION
    if not all((path / f).exists() for f in ("train.csv", "test.csv", "sample_submission.csv")):
        import kagglehub

        ensure_login()
        path = Path(kagglehub.competition_download(COMPETITION))
    read = lambda f: pd.read_csv(path / f, low_memory=False)  # noqa: E731
    return read("train.csv"), read("test.csv"), read("sample_submission.csv")
