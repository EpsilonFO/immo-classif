import pandas as pd, numpy as np
pd.set_option("display.width", 250, "display.max_rows", 200, "display.max_columns", 50)
D = "data/competitions/demandes-de-pret-immobilier-aux-etats-unis/"
tr, te = pd.read_csv(D + "train.csv", low_memory=False), pd.read_csv(D + "test.csv", low_memory=False)
print(tr.shape, te.shape, tr.accepted.mean())
print(tr.id.describe(), te.id.describe())
rows = []
for c in tr.columns:
    if c in ("id", "accepted"): continue
    s = tr[c]
    top = s.astype(str).value_counts().head(4)
    rows.append(dict(col=c, dtype=str(s.dtype), nuniq=s.nunique(), nan_tr=round(s.isna().mean(), 3), nan_te=round(te[c].isna().mean(), 3),
                     acc_nan=round(tr.accepted[s.isna()].mean(), 3) if s.isna().any() else np.nan,
                     top=" | ".join(f"{k}:{v}" for k, v in top.items())[:90]))
print(pd.DataFrame(rows).to_string())
print(tr.head(3).T.to_string())
