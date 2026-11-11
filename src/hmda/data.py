import os
from pathlib import Path

from hmda.config import COMPETITION, DATA_DIR

os.environ.setdefault("KAGGLEHUB_CACHE", str(DATA_DIR))

import pandas as pd  # noqa: E402


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Renvoie train, test, sample_submission (téléchargés avec kagglehub s'ils sont absents)."""
    path = DATA_DIR / "competitions" / COMPETITION
    if not (path / "train.csv").exists():
        import kagglehub

        path = Path(kagglehub.competition_download(COMPETITION))
    read = lambda f: pd.read_csv(path / f, low_memory=False)  # noqa: E731
    return read("train.csv"), read("test.csv"), read("sample_submission.csv")
