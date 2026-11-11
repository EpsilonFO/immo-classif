"""Quelles clés catégorielles (target encoding dans le pli) apportent de l'AUC à LightGBM ? Ajout une à une sur la
base v2+TE, puis candidates v4. LightGBM rapide (taux 0,1), 5 plis. Résultats : keys_ablation.csv"""
import sys
import numpy as np, pandas as pd
from hmda.data import load_data; from hmda.features import build_features, key_columns, HIGH_CARD; from hmda.cv import make_folds, run_cv
tr, te, _ = load_data(); F, Ft = build_features(tr, te, "v3"); folds = make_folds(tr); y = tr.accepted.to_numpy()
al = pd.concat([tr.drop(columns="accepted"), te], ignore_index=True); A = pd.concat([F, Ft], ignore_index=True)
st = lambda c: al[c].astype(object).where(al[c].notna(), "NA").astype(str)
fp = A["fingerprint"].astype(str)
cand = {
    "k_property": A["tract_id"].astype(str) + "|" + st("loan_amount") + "|" + st("property_value"),
    "k_borrower": A["tract_id"].astype(str) + "|" + st("income") + "|" + st("applicant_age") + "|" + st("derived_race") + "|" + st("derived_sex"),
    "k_fp_dti": fp + "|" + st("debt_to_income_ratio"),
    "k_fp_term": fp + "|" + st("loan_term"),
    "k_fp_ltv": fp + "|" + al["loan_to_value_ratio"].round(0).astype(str),
    "k_fp_tract": fp + "|" + A["tract_id"].astype(str),
    "k_term_ltv": st("loan_term") + "|" + al["loan_to_value_ratio"].round(1).astype(str),
    "k_lti": (al["loan_amount"] / (al["income"] * 1000)).round(1).astype(str),
    "k_amount_value": st("loan_amount") + "|" + st("property_value"),
    "k_fp_amount": fp + "|" + st("loan_amount"),
    "k_fp_income": fp + "|" + st("income"),
    "k_county_dwelling": st("county_code") + "|" + st("derived_dwelling_category"),
}
for k, v in cand.items():
    A[k] = pd.factorize(v)[0].astype(np.float32); A[k.replace("k_", "n_")] = v.map(v.value_counts()).astype(np.float32)
F, Ft = A.iloc[:len(tr)].reset_index(drop=True), A.iloc[len(tr):].reset_index(drop=True)
v2 = ["k_fp2", "k_fp_state", "k_fp_county", "k_fp2_state", "k_fp_msa", "k_state_product", "k_county_product", "k_fp_dwelling", "k_demo"]
v3 = ["k_loan_term", "k_ltv_round", "k_loan_amount", "k_property_value", "k_income", "k_ltv_dec", "k_tract"]
base = HIGH_CARD + ["fingerprint", "state_code"] + v2
P = dict(learning_rate=0.1, patience=50)
rows = []
def go(label, cols):
    r = run_cv("abl", "lgbm", y, F, Ft, folds, params={**P, "te": cols}, save=False, verbose=False)
    rows.append({"config": label, "cv_auc": r["cv_auc"], "folds": r["fold_auc"]}); print(f"{label:28s} {r['cv_auc']:.5f} {r['fold_auc']}", flush=True)
    pd.DataFrame(rows).to_csv("archive/experiments/keys_ablation.csv", index=False)
go("base v2", base)
go("base + v3", base + v3)
for k in v3: go(f"v2 + {k}", base + [k])
for k in cand: go(f"v3 + {k}", base + v3 + [k])
go("v3 + all cand", base + v3 + list(cand))
