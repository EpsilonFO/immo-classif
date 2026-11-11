import pandas as pd, numpy as np
pd.set_option("display.width", 250, "display.max_rows", 300, "display.max_columns", 50)
D = "data/competitions/demandes-de-pret-immobilier-aux-etats-unis/"
tr, te = pd.read_csv(D + "train.csv", low_memory=False), pd.read_csv(D + "test.csv", low_memory=False)
y = tr.accepted
def rate(c, df=tr, minn=150):
    g = df.groupby(df[c].astype(str) if not isinstance(c, list) else c, dropna=False).accepted.agg(["size", "mean"])
    print(f"--- {c}"); print(g[g["size"] >= minn].sort_values("mean").round(3).to_string())
for c in ["debt_to_income_ratio", "aus-1", "applicant_credit_score_type", "co-applicant_credit_score_type", "derived_loan_product_type",
          "derived_dwelling_category", "submission_of_application", "initially_payable_to_institution", "open-end_line_of_credit",
          "business_or_commercial_purpose", "interest_only_payment", "balloon_payment", "preapproval", "occupancy_type", "applicant_age",
          "derived_race", "applicant_ethnicity_observed", "manufactured_home_land_property_interest", "total_units", "conforming_loan_limit", "reverse_mortgage"]:
    rate(c)
# numeric bins
for c in ["loan_to_value_ratio", "income", "loan_amount", "property_value", "loan_term", "tract_minority_population_percent", "tract_to_msa_income_percentage"]:
    q = pd.qcut(tr[c], 12, duplicates="drop")
    print(f"--- {c}"); print(tr.groupby(q, observed=True).accepted.agg(["size", "mean"]).round(3).to_string())
# LTV decimals
s = tr.loan_to_value_ratio
dec = s.map(lambda v: -1 if pd.isna(v) else len(repr(float(v)).split(".")[1].rstrip("0")))
print("--- nb décimales LTV"); print(tr.groupby(dec).accepted.agg(["size", "mean"]).round(3))
ltv2 = 100 * tr.loan_amount / tr.property_value
print("LTV vs calc:", (s - ltv2).describe())
print("--- ltv>100, >105:", y[s > 100].agg(["size", "mean"]).values, y[s > 105].agg(["size", "mean"]).values)
print("--- income<=0:", y[tr.income <= 0].agg(["size", "mean"]).values, " income nan", y[tr.income.isna()].mean())
# exempt
ex = tr["reverse_mortgage"] == 1111
print("exempt:", ex.sum(), y[ex].mean())
# n NaN
nn = tr.isna().sum(1); print(tr.groupby(nn).accepted.agg(["size", "mean"]).round(3))
# duplicates
feats = [c for c in te.columns if c != "id"]
print("dups in train:", tr.duplicated(feats, keep=False).sum(), " test rows identical to a train row:", te.merge(tr[feats].drop_duplicates(), on=feats, how="inner").shape[0])
# pseudo tract
tk = ["tract_population", "tract_minority_population_percent", "tract_owner_occupied_units", "tract_one_to_four_family_homes", "tract_median_age_of_housing_units", "tract_to_msa_income_percentage"]
al = pd.concat([tr, te]); g = al.groupby(tk).ngroup(); vc = g.value_counts()
print("pseudo-tracts:", vc.size, "rows in tract with >=2:", (vc[vc >= 2].sum()), "max", vc.max(), " n county per tract max:", al.assign(g=g).groupby("g").county_code.nunique().max())
print("test rows whose tract appears in train:", np.isin(g.values[len(tr):], g.values[:len(tr)]).mean())
# id vs target
print("corr id-target", np.corrcoef(tr.id, y)[0, 1], tr.groupby(pd.qcut(tr.id, 10)).accepted.mean().round(3).values)
