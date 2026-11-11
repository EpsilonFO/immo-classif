"""Importance des variables (gain LightGBM, pli 0) et taux d'acceptation par empreinte prêteur / quartier."""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from hmda.data import load_data; from hmda.features import build_features; from hmda.cv import make_folds
pd.set_option("display.width", 250, "display.max_rows", 300)
tr, te, _ = load_data(); F, Ft = build_features(tr, te); folds = make_folds(tr); y = tr.accepted.to_numpy()
a, b = folds != 0, folds == 0
m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=63, min_child_samples=40, colsample_bytree=0.5, subsample=0.8, subsample_freq=1, reg_lambda=5, verbose=-1, n_jobs=2)
m.fit(F[a], y[a])
p = m.predict_proba(F[b])[:, 1]; print("auc", roc_auc_score(y[b], p))
imp = pd.Series(m.booster_.feature_importance("gain"), F.columns).sort_values(ascending=False)
print((imp / imp.sum()).round(4).to_string())
g = tr.assign(fp=F.fingerprint.astype(str)).groupby("fp").accepted.agg(["size", "mean"])
print("fingerprints:", len(g)); print(g.sort_values("size", ascending=False).head(40).round(3).to_string())
# erreurs : où sont les refus mal classés ?
d = tr[b].assign(p=p)
print(d.groupby(pd.qcut(d.p, 10, duplicates="drop"), observed=True).accepted.agg(["size", "mean"]).round(3))
