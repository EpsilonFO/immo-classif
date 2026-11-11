"""Prétraitement et variables construites (journal, étapes 3 et 4).

Principe : ne rien imputer. Les valeurs manquantes et les codes « exempté » (1111) portent du signal (le taux
d'acceptation passe de 91 % à 54 % quand le LTV est manquant) ; les arbres les gèrent nativement, et les réseaux
reçoivent des indicatrices (voir `models.py`). Rien ici ne dépend de la cible : les statistiques de groupe sont
calculées sur train + test réunis.
"""

import numpy as np
import pandas as pd

from hmda.config import ID, TARGET

TRACT_KEYS = [
    "county_code", "tract_population", "tract_minority_population_percent", "tract_owner_occupied_units",
    "tract_one_to_four_family_homes", "tract_median_age_of_housing_units", "tract_to_msa_income_percentage",
]
NUM_RAW = [
    "loan_amount", "loan_to_value_ratio", "property_value", "income", "loan_term", "total_units", "tract_population",
    "tract_minority_population_percent", "ffiec_msa_md_median_family_income", "tract_to_msa_income_percentage",
    "tract_owner_occupied_units", "tract_one_to_four_family_homes", "tract_median_age_of_housing_units",
]
HIGH_CARD = ["county_code", "derived_msa-md", "tract_id"]  # codes entiers pour les arbres, catégories pour CatBoost
DTI_MAP = {"<20%": 15.0, "20%-<30%": 25.0, "30%-<36%": 33.0, "50%-60%": 55.0, ">60%": 70.0}
AGE_MAP = {"<25": 22.0, "25-34": 30.0, "35-44": 40.0, "45-54": 50.0, "55-64": 60.0, "65-74": 70.0, ">74": 80.0}


def _decimals(s: pd.Series) -> pd.Series:
    """Nombre de décimales saisies (–1 si manquant) : empreinte du système de saisie du prêteur."""
    return s.map(lambda v: -1 if pd.isna(v) else len(repr(float(v)).split(".")[1].rstrip("0"))).astype(np.float32)


def build_features(train: pd.DataFrame, test: pd.DataFrame, version: str = "v1") -> tuple[pd.DataFrame, pd.DataFrame]:
    """`raw` : variables d'origine seulement (repère). `v1` : + variables construites."""
    al = pd.concat([train.drop(columns=[TARGET]), test], ignore_index=True)
    raw_cols = [c for c in al.columns if c != ID]
    cat_cols = [c for c in raw_cols if c not in NUM_RAW and c not in HIGH_CARD]
    F = pd.DataFrame(index=al.index)
    for c in NUM_RAW:
        F[c] = al[c].astype(np.float32)
    for c in cat_cols:  # le manquant devient une modalité à part entière
        F[c] = al[c].astype(object).where(al[c].notna(), "NA").astype(str).astype("category")
    F["county_code"] = al["county_code"].astype(np.float32)
    F["derived_msa-md"] = al["derived_msa-md"].astype(np.float32)

    if version != "raw":
        inc = al["income"].where(al["income"] > 0) * 1000.0  # revenu en dollars ; <= 0 traité comme inconnu
        loan, prop, ltv, term = al["loan_amount"], al["property_value"], al["loan_to_value_ratio"], al["loan_term"]
        F["dti_num"] = al["debt_to_income_ratio"].map(lambda v: DTI_MAP.get(v, v)).astype(float).astype(np.float32)
        F["age_num"] = al["applicant_age"].map(AGE_MAP).astype(np.float32)
        F["co_age_num"] = al["co-applicant_age"].map(AGE_MAP).astype(np.float32)
        F["income_nonpos"] = (al["income"] <= 0).astype(np.float32)
        F["ltv_calc"] = (100 * loan / prop).astype(np.float32)
        F["ltv_diff"] = (ltv - F["ltv_calc"]).astype(np.float32)
        F["ltv_dec"] = _decimals(ltv)
        F["ltv_over100"] = (ltv - 100).clip(lower=0).astype(np.float32)
        F["down_payment"] = (prop - loan).astype(np.float32)
        F["lti"] = (loan / inc).astype(np.float32)
        F["pti"] = (prop / inc).astype(np.float32)
        r = 0.07 / 12  # mensualité approchée à 7 % : seule la forme compte pour des arbres
        pmt = loan * r / (1 - (1 + r) ** (-term.fillna(360).clip(lower=12)))
        F["pmt_to_income"] = (12 * pmt / inc).astype(np.float32)
        F["other_debt"] = (F["dti_num"] - 100 * F["pmt_to_income"]).astype(np.float32)
        F["residual_income"] = (inc / 12 * (1 - F["dti_num"] / 100)).astype(np.float32)
        F["income_rel_msa"] = (inc / al["ffiec_msa_md_median_family_income"].replace(0, np.nan)).astype(np.float32)
        F["loan_per_unit"] = (loan / al["total_units"]).astype(np.float32)
        F["owner_occ_rate"] = (al["tract_owner_occupied_units"] / al["tract_one_to_four_family_homes"].replace(0, np.nan)).astype(np.float32)
        F["tract_income"] = (al["tract_to_msa_income_percentage"] * al["ffiec_msa_md_median_family_income"] / 100).astype(np.float32)
        F["n_nan"] = al[raw_cols].isna().sum(axis=1).astype(np.float32)
        F["n_exempt"] = (al[raw_cols] == 1111).sum(axis=1).astype(np.float32)
        F["has_coapp"] = (al["co-applicant_sex"] != 5).astype(np.float32)
        # pseudo-quartier : les 6 statistiques de recensement identifient le « census tract » (58 000 valeurs)
        tract = al.groupby(TRACT_KEYS, dropna=False).ngroup()
        tract = tract.where(al["tract_population"] > 0, -1)
        F["tract_id"] = tract.astype(np.float32)
        for name, key in [("tract", tract), ("county", al["county_code"].fillna(-1)), ("msa", al["derived_msa-md"]),
                          ("state", al["state_code"].fillna("NA"))]:
            g = al.groupby(key)
            F[f"{name}_count"] = g[ID].transform("size").astype(np.float32)
            if name in ("tract", "county", "state"):
                for col, s in [("loan", loan), ("prop", prop), ("inc", inc)]:
                    F[f"{col}_rel_{name}"] = (s / s.groupby(key).transform("median")).astype(np.float32)
        # effectif de l'« empreinte prêteur » : combinaison de variables de processus propres à un établissement
        fp = ["applicant_credit_score_type", "aus-1", "submission_of_application", "initially_payable_to_institution",
              "open-end_line_of_credit", "reverse_mortgage"]
        key = al[fp].astype(str).agg("|".join, axis=1) + "|" + F["ltv_dec"].astype(str)
        F["fingerprint_count"] = key.map(key.value_counts()).astype(np.float32)
        F["fingerprint"] = key.where(F["fingerprint_count"] >= 30, "rare").astype("category")
    if version in ("v2", "v3", "v4"):
        F = pd.concat([F, _keys_v2(al, F)], axis=1)
    if version in ("v3", "v4"):  # numériques clés dupliquées en catégories : CatBoost les croise avec les autres catégories
        code = lambda v: pd.factorize(v.where(v.map(v.value_counts()) >= 20, -1).fillna(-2))[0].astype(np.float32)  # noqa: E731
        F["k_loan_term"] = code(al["loan_term"])
        F["k_ltv_round"] = code(al["loan_to_value_ratio"].round(1))
        F["k_loan_amount"] = code(al["loan_amount"])
        F["k_property_value"] = code(al["property_value"])
        F["k_income"] = code(al["income"])
        F["k_ltv_dec"] = F["ltv_dec"]
        F["k_tract"] = code(F["tract_id"])
    if version == "v4":  # ratios continus découpés en 32 tranches : nouvelles catégories à croiser pour CatBoost
        new = {}
        for c in ["lti", "pti", "pmt_to_income", "other_debt", "residual_income", "income_rel_msa", "ltv_frac", "ltv_calc",
                  "down_payment", "prop_rel_county", "inc_rel_county", "tract_minority_population_percent",
                  "tract_to_msa_income_percentage", "owner_occ_rate", "tract_median_age_of_housing_units"]:
            new[f"k_bin_{c}"] = pd.qcut(F[c].replace([np.inf, -np.inf], np.nan), 32, labels=False, duplicates="drop").fillna(-1).astype(np.float32)
        st = lambda c: al[c].astype(object).where(al[c].notna(), "NA").astype(str)  # noqa: E731
        prop = F["tract_id"].astype(str) + "|" + st("loan_amount") + "|" + st("property_value")
        new["n_property"] = prop.map(prop.value_counts()).astype(np.float32)  # dossiers portant sur le même bien
        F = pd.concat([F, pd.DataFrame(new, index=F.index)], axis=1)
    n = len(train)
    return F.iloc[:n].reset_index(drop=True), F.iloc[n:].reset_index(drop=True)


def _keys_v2(al: pd.DataFrame, F: pd.DataFrame) -> pd.DataFrame:
    """v2 : clés composites `k_*` (codes entiers) destinées au target encoding dans le pli (LightGBM, XGBoost, réseaux)
    ou aux statistiques ordonnées de CatBoost. Elles approchent l'identité du prêteur, absente des données : un
    prêteur = un système de saisie (formats, variables renseignées) × une zone géographique."""
    K = pd.DataFrame(index=al.index)
    s = lambda cols: al[cols].astype(object).where(al[cols].notna(), "NA").astype(str).agg("|".join, axis=1)  # noqa: E731
    one = lambda c: al[c].astype(object).where(al[c].notna(), "NA").astype(str)  # noqa: E731
    fp = F["fingerprint"].astype(str)
    nan_cols = ["loan_to_value_ratio", "property_value", "income", "loan_term", "debt_to_income_ratio", "applicant_age",
                "state_code", "county_code"]
    nanpat = al[nan_cols].isna().astype(int).astype(str).agg("".join, axis=1)
    obs = s(["applicant_ethnicity_observed", "applicant_race_observed", "applicant_sex_observed"])
    fp2 = fp + "|" + s(["co-applicant_credit_score_type", "loan_type"]) + "|" + obs + "|" + nanpat
    ltv, term = al["loan_to_value_ratio"], al["loan_term"]
    K["ltv_frac"] = (ltv - np.floor(ltv)).astype(np.float32)
    K["term_mod12"] = (term % 12).astype(np.float32)
    K["nan_pattern"] = nanpat.where(nanpat.map(nanpat.value_counts()) >= 30, "rare").astype("category")
    keys = {
        "k_fp2": fp2, "k_fp_state": fp + "|" + one("state_code"), "k_fp_county": fp + "|" + one("county_code"),
        "k_fp2_state": fp2 + "|" + one("state_code"), "k_fp_msa": fp + "|" + one("derived_msa-md"),
        "k_state_product": s(["state_code", "derived_loan_product_type", "derived_dwelling_category"]),
        "k_county_product": s(["county_code", "loan_type"]),
        "k_fp_dwelling": fp + "|" + s(["derived_dwelling_category", "loan_type", "occupancy_type"]),
        "k_demo": s(["derived_race", "derived_ethnicity", "derived_sex", "applicant_age"]),
    }
    for k, v in keys.items():
        K[k] = pd.factorize(v)[0].astype(np.float32)
        K[k.replace("k_", "n_")] = v.map(v.value_counts()).astype(np.float32)
    return K


def key_columns(F: pd.DataFrame) -> list[str]:
    return [c for c in F.columns if c.startswith("k_")]


def cat_columns(F: pd.DataFrame) -> list[str]:
    return [c for c in F.columns if isinstance(F[c].dtype, pd.CategoricalDtype)]
