import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Résumé classification arbres + réseaux neuronaux + SVM + AutoML

    Les dépendances principales s'installent avec `uv sync`. Deux paquets des cellules
    AutoGluon et `scikit-obliquetree` sont absents de cet environnement car ils
    entrent en conflit avec le reste du projet (`pandas>=3` / `tabm`) **et entre eux**
    (`scikit-obliquetree` tire `rich<10` → `typing-extensions<4`, incompatible avec
    AutoGluon). Les installer dans **deux** environnements séparés (désolé…) :

    ```bash
    # AutoGluon
    uv venv .venv-autogluon
    source .venv-autogluon/bin/activate
    uv pip install autogluon.tabular

    # scikit-obliquetree
    uv venv .venv-scikit-obliquetree
    source .venv-scikit-obliquetree/bin/activate
    uv pip install scikit-obliquetree numpy scikit-learn scipy
    ```
    """)
    return


@app.cell
def _():
    import re
    import time
    from typing import NamedTuple

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from scipy.special import erf
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.base import clone
    import sklearn
    from catboost import CatBoostClassifier
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.tree import DecisionTreeClassifier


    return (
        CatBoostClassifier,
        DecisionTreeClassifier,
        ExtraTreesClassifier,
        HistGradientBoostingClassifier,
        NamedTuple,
        RandomForestClassifier,
        XGBClassifier,
        clone,
        mo,
        np,
        pd,
        plt,
        roc_auc_score,
        sklearn,
        time,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Configuration
    """)
    return


@app.cell
def _(torch):
    LEARN_MISSING_VALUES = False
    TRAIN_NN = True

    BOOSTING = dict(n_estimators=1000, learning_rate=0.03)

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return LEARN_MISSING_VALUES, TRAIN_NN, device


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Read data
    """)
    return


@app.cell
def _(pd):
    train = pd.read_csv("data challenge/classification/train.csv")
    test = pd.read_csv("data challenge/classification/test.csv")
    return test, train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data preparation
    """)
    return


@app.cell
def _(np, pd):
    MISSING_CODE = 0

    # Ces colonnes ont des valeurs de type string et doivent être converties env aleurs numériques.
    STRING_CATEGORICAL = [
        "state_code",
        "conforming_loan_limit",
        "derived_loan_product_type",
        "derived_dwelling_category",
        "derived_ethnicity",
        "derived_race",
        "derived_sex",
    ]

    CODED_CATEGORICAL = [
        "derived_msa-md",
        "county_code",
        "loan_type",
        "lien_status",
        "preapproval",
        "reverse_mortgage",
        "open-end_line_of_credit",
        "business_or_commercial_purpose",
        "negative_amortization",
        "interest_only_payment",
        "balloon_payment",
        "other_nonamortizing_features",
        "construction_method",
        "occupancy_type",
        "manufactured_home_secured_property_type",
        "manufactured_home_land_property_interest",
        "applicant_credit_score_type",
        "co-applicant_credit_score_type",
        "applicant_ethnicity-1",
        "co-applicant_ethnicity-1",
        "applicant_ethnicity_observed",
        "co-applicant_ethnicity_observed",
        "applicant_race-1",
        "co-applicant_race-1",
        "applicant_race_observed",
        "co-applicant_race_observed",
        "applicant_sex",
        "co-applicant_sex",
        "applicant_sex_observed",
        "co-applicant_sex_observed",
        "submission_of_application",
        "initially_payable_to_institution",
        "aus-1",
    ]

    # pour convertir les tranches d'âge en valeurs numériques
    AGE_MIDPOINTS = {
        "<25": 22.0,
        "25-34": 29.5,
        "35-44": 39.5,
        "45-54": 49.5,
        "55-64": 59.5,
        "65-74": 69.5,
        ">74": 80.0,
    }

    def parse_age(value):
        """
        Preserve NaN
        Replace age ranges with a representative value (usually midpoint)
        Parses numerical strings into floats
        """
        if pd.isna(value):
            return -1
        text = str(value).strip()
        if text in AGE_MIDPOINTS:
            return AGE_MIDPOINTS[text]
        return float(text)

    def parse_dti(value):
        """Parse debt-to-income into number or range midpoint"""
        if pd.isna(value):
            return np.nan
        text = str(value).strip().replace("%", "").replace(" ", "").replace("-<", "-")
        try:
            return float(text)
        except ValueError:
            pass
        if text.startswith("<"):
            bound = float(text[1:])
            return bound / 2.0
        if text.startswith(">"):
            bound = float(text[1:])
            return bound + 10.0
        if "-" in text:
            left, right = text.split("-", 1)
            return (float(left) + float(right)) / 2.0
        return np.nan

    def category_labels(series):
        # numeric codes go through Float64 so that the same code gives the same label whether the column was read as int or float
        if pd.api.types.is_numeric_dtype(series):
            series = series.astype("Float64")
        return series.astype("string").fillna("__missing__")

    def build_category_mappings(*frames):
        mappings, cardinalities = {}, {}
        for col in STRING_CATEGORICAL + CODED_CATEGORICAL:
            labels = set().union(*(category_labels(df[col]).unique().tolist() for df in frames))
            categories = sorted(labels - {"__missing__"})
            mapping = {label: i + 1 for i, label in enumerate(categories)}  # we map categories into [1, ..., N] so that zero is used for missing values
            mapping["__missing__"] = 0
            mappings[col] = mapping
            cardinalities[col] = len(categories) + 2
        return mappings, cardinalities

    def encode_string_categorical(series, mapping):
        return category_labels(series).map(lambda x: mapping[x]).astype(int)

    def prepare_frame(df, category_mappings, *, take_log=False):
        out = df.copy()

        out["applicant_age"] = out["applicant_age"].map(parse_age)
        out["co-applicant_age"] = out["co-applicant_age"].map(parse_age)
        out["debt_to_income_ratio"] = out["debt_to_income_ratio"].map(parse_dti)

        for col in STRING_CATEGORICAL + CODED_CATEGORICAL:
            out[col] = encode_string_categorical(out[col], category_mappings[col])

        for col in CODED_CATEGORICAL:
            out[col] = out[col].fillna(MISSING_CODE)

        if take_log:
            out["loan_amount"] = np.log(out["loan_amount"].clip(lower=1))
            out["property_value"] = np.log(out["property_value"].clip(lower=1))

        return out

    return build_category_mappings, prepare_frame


@app.cell
def _(build_category_mappings, prepare_frame, test, train):
    category_mappings, CATEGORICAL_FIELDS = build_category_mappings(train, test)
    train_coded = prepare_frame(train, category_mappings, take_log=True)
    test_coded = prepare_frame(test, category_mappings, take_log=True)
    return CATEGORICAL_FIELDS, test_coded, train_coded


@app.cell
def _(CATEGORICAL_FIELDS, train_coded):
    CONT_COLS = [
        c for c in train_coded.columns
        if c not in CATEGORICAL_FIELDS and c not in {"id", "accepted"}
    ]
    CAT_COLS = list(CATEGORICAL_FIELDS)
    return CAT_COLS, CONT_COLS


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fit / validation split

    Découpage en jeu d'entraînement et de validation. On permet de controler le taux d'exemples négatifs pour combler la répartition très inégale dans le jeu de données fourni.
    """)
    return


@app.cell
def _(np, train_coded):
    labels = train_coded["accepted"].to_numpy(dtype=np.float32)

    _idx_0 = np.flatnonzero(labels == 0)
    _idx_1 = np.flatnonzero(labels == 1)
    np.random.shuffle(_idx_0)
    np.random.shuffle(_idx_1)

    _ratio_1 = labels.mean()

    _n_val = int(0.1 * len(labels))
    _n_val_0 = int((1 - _ratio_1) * len(_idx_0))
    _n_val_1 = max(_n_val - _n_val_0, 0)

    val_idx = np.concatenate((_idx_0[:_n_val_0], _idx_1[:_n_val_1]))
    fit_idx = np.concatenate((_idx_0[_n_val_0:], _idx_1[_n_val_1:]))
    np.random.shuffle(val_idx)
    np.random.shuffle(fit_idx)

    print(f"{len(fit_idx)} fit rows, {len(val_idx)} validation rows; ratio of accepted: {labels.mean():.3f} overall, {labels[val_idx].mean():.3f} in validation")
    return fit_idx, labels, val_idx


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Apprentissage des données manquantes

    On impute certaines des données manquantes avec xgboost. On a sélectionné un échantillon de colonnes avec beaucoup de données manquantes et pour chacune d'elles on a sélectionné des colonnes lui étant suffisament corrélées et ayant peu de données manquantes. Pour le reste, l'imputation dépendera du modèle (moyenne pour les réseaux de neurone, médiane pour les SVM dans les arbres obliques, dans les autres cas le modèle gère tout seul (éventuellement on remplace nan par une valeur numérique placeholder)).
    """)
    return


@app.cell
def _(LEARN_MISSING_VALUES, fit_idx, test_coded, train_coded):
    from xgboost import XGBRegressor

    def fill_missing_values(target, covariates, train_frame, test_frame, fit_rows):
        """Replace the missing values of `target` by a regression on `covariates`.

        The regression is fitted on the fit rows where `target` and every covariate is observed, on
        standardised columns; at prediction time a missing covariate is replaced by its fit-set mean.
        """
        _fit = train_frame.iloc[fit_rows]
        _observed = ~_fit[target].isna()
        for _c in covariates:
            _observed &= ~_fit[_c].isna()
        _complete = _fit.loc[_observed, covariates + [target]]
        if len(_complete) == 0:
            return train_frame, test_frame

        _mean = _complete.mean(axis=0)
        _std = _complete.std(axis=0).replace(0.0, 1.0)
        _model = XGBRegressor(random_state=0, n_estimators=200, max_depth=4, learning_rate=0.1)
        _model.fit(
            (_complete[covariates] - _mean[covariates]) / _std[covariates],
            (_complete[target] - _mean[target]) / _std[target],
        )

        _filled = []
        for _frame in (train_frame, test_frame):
            _frame = _frame.copy()
            _missing = _frame[target].isna()
            if _missing.any():
                _X = (_frame.loc[_missing, covariates].fillna(_mean[covariates]) - _mean[covariates]) / _std[covariates]
                _frame.loc[_missing, target] = _model.predict(_X) * _std[target] + _mean[target]
            _filled.append(_frame)
        return _filled[0], _filled[1]

    _IMPUTATIONS = [
        ("property_value", ["loan_amount", "construction_method", "ffiec_msa_md_median_family_income", "tract_to_msa_income_percentage", "income"]),
        ("income", ["property_value", "conforming_loan_limit", "loan_amount", "debt_to_income_ratio"]),
        ("loan_term", ["balloon_payment", "aus-1", "interest_only_payment", "business_or_commercial_purpose", "loan_amount"]),
        ("debt_to_income_ratio", ["loan_amount", "construction_method", "ffiec_msa_md_median_family_income", "tract_to_msa_income_percentage", "income"]),
        ("loan_to_value_ratio", ["loan_amount", "construction_method", "ffiec_msa_md_median_family_income", "tract_to_msa_income_percentage", "income"]),
    ]

    def impute_missing(train_frame, test_frame, fit_rows):
        for _target, _covariates in _IMPUTATIONS:
            train_frame, test_frame = fill_missing_values(
                _target, _covariates, train_frame, test_frame, fit_rows
            )
        return train_frame, test_frame

    train_prep, test_prep = train_coded, test_coded
    if LEARN_MISSING_VALUES:
        for _target, _covariates in _IMPUTATIONS:
            print(f"imputing {_target} ({int(train_prep[_target].isna().sum())} missing in train)")
        train_prep, test_prep = impute_missing(train_prep, test_prep, fit_idx)

    print(f"remaining missing values in train: {int(train_prep.isna().sum().sum())}")
    return impute_missing, test_prep, train_prep


@app.cell
def _(CATEGORICAL_FIELDS, CAT_COLS, CONT_COLS):
    TREE_FEATURES = CONT_COLS + CAT_COLS

    # high-cardinality columns (county, MSA) are left as numeric codes
    TREE_LOW_CARD_CATEGORICAL = [c for c, n in CATEGORICAL_FIELDS.items() if n <= 64]
    return TREE_FEATURES, TREE_LOW_CARD_CATEGORICAL


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Réseaux de neurones
    """)
    return


@app.cell
def _(torch):
    def build_mlp(in_dim, hidden_dimensions, dropout):
        """The MLP of `classification2.py` (its `residual` flag only changes `forward`)."""
        layers = [torch.nn.Linear(in_dim, hidden_dimensions[0]), torch.nn.GELU(), torch.nn.Dropout(dropout)]
        for i in range(len(hidden_dimensions) - 1):
            layers.append(torch.nn.Linear(hidden_dimensions[i], hidden_dimensions[i + 1]))
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.Dropout(dropout))
        layers.append(torch.nn.Linear(hidden_dimensions[-1], 1))
        layers.append(torch.nn.Sigmoid())
        return torch.nn.Sequential(*layers)

    class NeuralNetwork(torch.nn.Module):
        """`classification2.py`'s network, restricted to the tensor input path.

        `forward(..., return_hidden=True)` returns the activations that enter the output layer.
        """

        def __init__(self, cardinalities, n_continuous, embedding_dimension, hidden_dimensions, dropout=0.5, residual=False):
            super().__init__()
            self.embeddings = torch.nn.ModuleDict({
                cat: torch.nn.Embedding(num_embeddings=n, embedding_dim=embedding_dimension)
                for cat, n in cardinalities.items()
            })
            self.cat_cols = list(cardinalities)
            self.net = build_mlp(embedding_dimension * len(self.cat_cols) + n_continuous, hidden_dimensions, dropout)
            self.residual = []
            if residual:
                for i, u in zip(range(len(hidden_dimensions)), hidden_dimensions):
                    if i >= len(hidden_dimensions) // 2:
                        break
                    if u in self.residual:
                        continue
                    if u in hidden_dimensions[i + 1:]:
                        self.residual.append(u)

        def forward(self, continuous_features, categorical_features, return_hidden=False):
            embs = [self.embeddings[c](categorical_features[:, i]) for i, c in enumerate(self.cat_cols)]
            cont = torch.nan_to_num(continuous_features, nan=0.0)
            x = torch.cat(embs + [cont], dim=1)
            saved = {}
            for i, l in enumerate(self.net):
                if return_hidden and i == len(self.net) - 2:
                    return x
                x = l(x)
                if isinstance(l, torch.nn.Dropout) and x.size(-1) in self.residual:
                    if x.size(-1) in saved:
                        x += saved[x.size(-1)]
                    else:
                        saved[x.size(-1)] = x
            return x.squeeze(-1)

    class NeuralNetworkWithAttention(torch.nn.Module):
        """Categorical and per-feature continuous embeddings, self-attention, then the same MLP head.

        Same tensor interface as `NeuralNetwork`, so `train_network` and
        `AveragedEmbeddingNetwork` apply unchanged.
        """

        def __init__(
            self,
            cardinalities,
            n_continuous,
            embedding_dimension,
            hidden_dimensions,
            dropout=0.5,
            attn_heads=4,
        ):
            super().__init__()
            self.embeddings = torch.nn.ModuleDict({
                cat: torch.nn.Embedding(num_embeddings=n, embedding_dim=embedding_dimension)
                for cat, n in cardinalities.items()
            })
            self.cat_cols = list(cardinalities)
            heads = attn_heads
            while heads > 1 and embedding_dimension % heads != 0:
                heads -= 1
            self.continuous_embedding = torch.nn.Linear(1, embedding_dimension)
            self.attn = torch.nn.MultiheadAttention(
                embed_dim=embedding_dimension, num_heads=heads, batch_first=True, dropout=dropout,
            )
            self.net = build_mlp(
                embedding_dimension * (len(self.cat_cols) + n_continuous),
                hidden_dimensions,
                dropout,
            )
            self.residual = []

        def forward(self, continuous_features, categorical_features, return_hidden=False):
            cont = torch.nan_to_num(continuous_features, nan=0.0)
            cont = self.continuous_embedding(cont.unsqueeze(-1))
            embs = torch.stack(
                [self.embeddings[c](categorical_features[:, i]) for i, c in enumerate(self.cat_cols)],
                dim=1,
            )
            x = torch.cat([embs, cont], dim=1)
            x, _ = self.attn(x, x, x)
            x = x.flatten(1, -1)
            for i, layer in enumerate(self.net):
                if return_hidden and i == len(self.net) - 2:
                    return x
                x = layer(x)
            return x.squeeze(-1)

    return NeuralNetwork, NeuralNetworkWithAttention


@app.cell
def _(CONT_COLS, fit_idx, np, train_prep):
    _fit_cont = train_prep[CONT_COLS].to_numpy(dtype=np.float64)[fit_idx, :]
    mean = np.nan_to_num(np.nanmean(_fit_cont, axis=0), nan=0.0)
    std = np.nan_to_num(np.sqrt(np.nanmean((_fit_cont - mean) ** 2, axis=0)), nan=1.0)
    std[std < 1e-6] = 1.0
    cont_mean, cont_std = mean, std
    return cont_mean, cont_std


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Entraînement

    Les cellules suivantes permettent d'entraîner un réseau de neurones sur les données en réglant différents paramètres d'architecture et d'entraînement. `AveragedEmbeddingNetwork` est une classe qui contient plusieurs modèles entraînés et permet de calculer des embeddings ou préditions en prenant la moyenne des embeddings ou prédictions calculées par chacun des modèles qu'elle contient. Cela permet de faire du bootstrap.
    """)
    return


@app.cell
def _(np, roc_auc_score, torch):
    def train_network(
        model, cont, cat, y, fit_rows, val_rows, *, epochs, batch_size, learning_rate, patience, label="", log_rate=1, balance=False
    ):
        device = cont.device
        optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.BCELoss(reduction="none")

        fit_rows = torch.as_tensor(np.asarray(fit_rows), dtype=torch.long, device=device)
        val_rows = torch.as_tensor(np.asarray(val_rows), dtype=torch.long, device=device)

        # rééquilibrage des classes selon la méthode vue dans une annale
        alpha = y[fit_rows].mean().clamp(1e-6, 1 - 1e-6)
        weights = torch.where(y[fit_rows] == 1, 1 / alpha, 1 / (1 - alpha))
        if not balance:
            weights = torch.ones_like(weights)
        history, best_auc, best_state, stale = [], -np.inf, None, 0
        for epoch in range(epochs):
            model.train()
            order = torch.randperm(len(fit_rows), device=device)
            for start in range(0, len(order), batch_size):
                batch = order[start : start + batch_size]
                rows = fit_rows[batch]
                optimiser.zero_grad()
                loss = (weights[batch] * criterion(model(cont[rows], cat[rows]), y[rows])).mean()
                loss.backward()
                optimiser.step()

            model.eval()
            with torch.no_grad():
                _pred = torch.cat([
                    model(cont[chunk], cat[chunk]) for chunk in val_rows.split(100_000)
                ])
            auc = roc_auc_score(y[val_rows].cpu().numpy(), _pred.cpu().numpy())
            history.append(auc)
            if (epoch+1) % log_rate == 0:
                print(f"{label}epoch {epoch + 1}/{epochs}: validation roc auc {auc:.4f}")

            if auc > best_auc:
                best_auc, stale = auc, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if patience is not None and stale >= patience:
                    print(f"{label}early stopping, restoring the epoch with roc auc {best_auc:.4f}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        return history

    return (train_network,)


@app.cell
def _(np, torch):
    class AveragedEmbeddingNetwork:
        """Trained `NeuralNetwork`s behind the interface of the two checkpoint backends above.

        `embed` averages the embedding tables of the models and `forward` averages their hidden
        activations and their predicted probabilities. `weights` (one per model) are normalised
        and used as the mixing coefficients; equal weights if omitted.
        """

        def __init__(self, models, batch_size=100_000, weights=None):
            self.models = [model.eval() for model in models]
            _first = self.models[0]
            _linears = [layer for layer in _first.net if isinstance(layer, torch.nn.Linear)]

            self.cat_cols = list(_first.cat_cols)
            self.cardinalities = {col: _first.embeddings[col].num_embeddings for col in self.cat_cols}
            self.embedding_dim = _first.embeddings[self.cat_cols[0]].embedding_dim
            self.hidden_dimensions = [layer.out_features for layer in _linears[:-1]]
            self.n_continuous = _linears[0].in_features - self.embedding_dim * len(self.cat_cols)
            self.residual = _first.residual
            self.device = next(_first.parameters()).device
            self.batch_size = batch_size

            n = len(self.models)
            if weights is None:
                w = np.ones(n, dtype=np.float64)
            else:
                w = np.asarray(weights, dtype=np.float64)
                if w.shape != (n,):
                    raise ValueError(f"weights must have shape {(n,)}, got {w.shape}")
            w = w / w.sum()
            self.weights = w
            self._weight_tensor = torch.as_tensor(w, dtype=torch.float32, device=self.device)

        def _mix(self, stacked):
            w = self._weight_tensor.reshape((-1,) + (1,) * (stacked.ndim - 1))
            return (stacked * w).sum(dim=0)

        def embed(self, codes):
            """One `(n, embedding_dim)` block per categorical column, in `cat_cols` order."""
            blocks = []
            with torch.no_grad():
                for i, col in enumerate(self.cat_cols):
                    index = torch.as_tensor(codes[:, i], dtype=torch.long, device=self.device)
                    tables = torch.stack([model.embeddings[col](index) for model in self.models])
                    blocks.append(self._mix(tables).cpu().numpy())
            return blocks

        def forward(self, continuous, codes):
            cont = torch.as_tensor(continuous, dtype=torch.float32, device=self.device)
            cat = torch.as_tensor(codes, dtype=torch.long, device=self.device)
            hidden, prob = [], []
            with torch.no_grad():
                for start in range(0, len(codes), self.batch_size):
                    batch = slice(start, start + self.batch_size)
                    hidden.append(
                        self._mix(torch.stack([m(cont[batch], cat[batch], return_hidden=True) for m in self.models]))
                        .cpu().numpy()
                    )
                    prob.append(
                        self._mix(torch.stack([m(cont[batch], cat[batch]) for m in self.models]))
                        .cpu().numpy()
                    )
            return np.concatenate(hidden), np.concatenate(prob)

    return (AveragedEmbeddingNetwork,)


@app.function
def make_hidden_dims(a, b, m):
    h = [a]
    while h[-1] < m:
        h.append(2*h[-1])
    while h[-1]//2 > b:
        h.append(h[-1]//2)
    return h[1:]


@app.cell
def _():
    NN_BOOTSTRAP = 5
    NN_EMBEDDING_DIM = 8
    print(NN_EMBEDDING_DIM)
    NN_HIDDEN = make_hidden_dims(NN_EMBEDDING_DIM * 16, 16, 256 * NN_EMBEDDING_DIM) # [256, 512, 256, 128, 64, 4]
    print(NN_HIDDEN)
    NN_DROPOUT = 0.5
    NN_EPOCHS = 100
    NN_BATCH_SIZE = 2048
    NN_LEARNING_RATE = 1e-4
    NN_PATIENCE = 10
    NN_SEED = 0
    return (
        NN_BATCH_SIZE,
        NN_BOOTSTRAP,
        NN_DROPOUT,
        NN_EMBEDDING_DIM,
        NN_EPOCHS,
        NN_HIDDEN,
        NN_LEARNING_RATE,
        NN_PATIENCE,
        NN_SEED,
    )


@app.cell
def _(
    AveragedEmbeddingNetwork,
    CATEGORICAL_FIELDS,
    CAT_COLS,
    CONT_COLS,
    NN_BATCH_SIZE,
    NN_BOOTSTRAP,
    NN_DROPOUT,
    NN_EMBEDDING_DIM,
    NN_EPOCHS,
    NN_HIDDEN,
    NN_LEARNING_RATE,
    NN_PATIENCE,
    NN_SEED,
    NeuralNetwork,
    TRAIN_NN,
    cont_mean,
    cont_std,
    device,
    fit_idx,
    labels,
    np,
    roc_auc_score,
    time,
    torch,
    train_network,
    train_prep,
    val_idx,
):
    trained_nn = None
    trained_nn_history = []

    if TRAIN_NN:
        _cont = torch.as_tensor(
            (train_prep[CONT_COLS].to_numpy(dtype=np.float64) - cont_mean) / cont_std,
            dtype=torch.float32,
            device=device,
        )
        _cat = torch.as_tensor(train_prep[CAT_COLS].to_numpy(dtype=np.int64), dtype=torch.long, device=device)
        _y = torch.as_tensor(labels, dtype=torch.float32, device=device)

        _n_models = max(NN_BOOTSTRAP, 1)
        _rng = np.random.default_rng(NN_SEED)
        _models = []
        _started = time.time()
        for _b in range(_n_models):
            #torch.manual_seed(NN_SEED)
            _model = NeuralNetwork(
                CATEGORICAL_FIELDS, len(CONT_COLS), NN_EMBEDDING_DIM, NN_HIDDEN,
                dropout=NN_DROPOUT, residual=False,
            ).to(device)
            # same initial weights for every model, but its own dropout masks and shuffling
            #torch.manual_seed(NN_SEED + 1 + _b)
            _rows = fit_idx if _n_models == 1 else _rng.choice(fit_idx, size=len(fit_idx), replace=True)
            trained_nn_history.append(train_network(
                _model, _cont, _cat, _y, _rows, val_idx,
                epochs=NN_EPOCHS, batch_size=NN_BATCH_SIZE, learning_rate=NN_LEARNING_RATE,
                patience=NN_PATIENCE, label="" if _n_models == 1 else f"model {_b + 1}/{_n_models}, ", log_rate=20
            ))
            _model.eval()
            _models.append(_model)

        _aucs = np.array([max(_h) for _h in trained_nn_history], dtype=np.float64)
        trained_nn = AveragedEmbeddingNetwork(_models, weights=_aucs / _aucs.sum())
        _preds = trained_nn.forward(_cont[val_idx], _cat[val_idx])[1]
        _final_roc_auc = roc_auc_score(_y[val_idx].detach().cpu(), _preds)
        print("final roc auc", _final_roc_auc)
    else:
        _models = [NeuralNetwork(
                CATEGORICAL_FIELDS, len(CONT_COLS), NN_EMBEDDING_DIM, NN_HIDDEN,
                dropout=NN_DROPOUT, residual=True,
            ).to(device) for _ in range(NN_BOOTSTRAP)]
        for _i, _m in enumerate(_models):
            _m.load_state_dict(torch.load(f"mixture{_i}.pt", weights_only=True))
        trained_nn = AveragedEmbeddingNetwork(_models)
    return trained_nn, trained_nn_history


@app.cell
def _(trained_nn_history):
    with open("2.txt", "w") as f:
        f.write("\n".join([",".join(map(str, t)) for t in trained_nn_history]))
    return


@app.cell
def _():
    #for _i, _m in enumerate(trained_nn.models):
    #    torch.save(_m.state_dict(), f"mixture{_i}.pt")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Réseau avec attention

    `NeuralNetworkWithAttention` reprend l'architecture de `classification2.py`: chaque variable
    continue est projetée dans le même espace que les embeddings catégoriels, un
    `MultiheadAttention` mélange la séquence, puis le MLP habituel produit une probabilité.
    L'entraînement utilise `train_network` (poids de classe, early-stopping sur `val_idx`) et le
    même bootstrap que le réseau sans attention.
    """)
    return


@app.cell
def _(
    AveragedEmbeddingNetwork,
    CATEGORICAL_FIELDS,
    CAT_COLS,
    CONT_COLS,
    NN_BATCH_SIZE,
    NN_BOOTSTRAP,
    NN_DROPOUT,
    NN_EMBEDDING_DIM,
    NN_EPOCHS,
    NN_HIDDEN,
    NN_PATIENCE,
    NeuralNetworkWithAttention,
    TRAIN_NN,
    cont_mean,
    cont_std,
    device,
    fit_idx,
    labels,
    np,
    roc_auc_score,
    torch,
    train_network,
    train_prep,
    val_idx,
):
    trained_nn_attn = None
    trained_nn_attn_history = []
    if TRAIN_NN:
        _cont = torch.as_tensor(
            (train_prep[CONT_COLS].to_numpy(dtype=np.float64) - cont_mean) / cont_std,
            dtype=torch.float32,
            device=device,
        )
        _cat = torch.as_tensor(train_prep[CAT_COLS].to_numpy(dtype=np.int64), dtype=torch.long, device=device)
        _y = torch.as_tensor(labels, dtype=torch.float32, device=device)
        _n_models = max(NN_BOOTSTRAP, 1)
        _rng = np.random.default_rng(1)
        _models = []
        for _b in range(_n_models):
            _model = NeuralNetworkWithAttention(
                CATEGORICAL_FIELDS, len(CONT_COLS), NN_EMBEDDING_DIM, NN_HIDDEN,
                dropout=NN_DROPOUT,
            ).to(device)
            _rows = fit_idx if _n_models == 1 else _rng.choice(fit_idx, size=len(fit_idx), replace=True)
            trained_nn_attn_history.append(train_network(
                _model, _cont, _cat, _y, _rows, val_idx,
                epochs=NN_EPOCHS, batch_size=NN_BATCH_SIZE, learning_rate=1e-5,
                patience=NN_PATIENCE,
                label="" if _n_models == 1 else f"attn model {_b + 1}/{_n_models}, ",
                log_rate=50,
            ))
            _model.eval()
            _models.append(_model)
        _aucs = np.array([max(_h) for _h in trained_nn_attn_history], dtype=np.float64)
        trained_nn_attn = AveragedEmbeddingNetwork(_models, weights=_aucs / _aucs.sum())
        _preds = trained_nn_attn.forward(_cont[val_idx], _cat[val_idx])[1]
        print("attention nn val roc auc", roc_auc_score(_y[val_idx].detach().cpu(), _preds))
    trained_nn_attn
    return (trained_nn_attn,)


@app.cell
def _(
    CAT_COLS,
    CONT_COLS,
    cont_mean,
    cont_std,
    np,
    pd,
    test_prep,
    train_prep,
    trained_nn,
):
    def embedded_frame(frame, with_output=False, network=trained_nn):
        """Continuous features plus the embedding of every categorical feature, one column per
        embedding coordinate; optionally plus the last hidden layer and the network's prediction."""
        _codes = frame[CAT_COLS].to_numpy(dtype=np.int64)
        columns = [frame[CONT_COLS].reset_index(drop=True)]
        for _c, _emb in zip(CAT_COLS, network.embed(_codes)):
            columns.append(pd.DataFrame(_emb, columns=[f"{_c}_emb{_k}" for _k in range(_emb.shape[1])]))

        if with_output:
            _cont = (frame[CONT_COLS].to_numpy(dtype=np.float64) - cont_mean) / cont_std
            _hidden, _prob = network.forward(_cont, _codes)
            columns.append(pd.DataFrame(_hidden, columns=[f"nn_hidden{_k}" for _k in range(_hidden.shape[1])]))
            columns.append(pd.DataFrame({"nn_prob": _prob}))

        return pd.concat(columns, axis=1)

    def clustered_embedded_frame(frame, clustered_nn, with_output=False):
        """Like `embedded_frame`, routing each row to the network of its cluster.

        `clustered_nn` is the 6-tuple returned by `train_clustered_networks`.
        """
        _, nns, clusterer, clustering_features, _, _ = clustered_nn
        first = next((nn for nn in nns if nn is not None), None)
        if first is None:
            return embedded_frame(frame, with_output=with_output)

        codes = frame[CAT_COLS].to_numpy(dtype=np.int64)
        clusters = clusterer.predict(clustering_features(
            frame[CONT_COLS].to_numpy(dtype=np.float32),
            frame[CAT_COLS].to_numpy(dtype=np.float32),
        ))
        n = len(frame)
        emb_dim = first.embedding_dim
        blocks = [np.zeros((n, emb_dim), dtype=np.float32) for _ in CAT_COLS]
        hidden, prob = None, np.full(n, np.nan, dtype=np.float32)
        cont = (frame[CONT_COLS].to_numpy(dtype=np.float64) - cont_mean) / cont_std

        for i, nn in enumerate(nns):
            if nn is None:
                continue
            mask = clusters == i
            if not np.any(mask):
                continue
            for j, emb in enumerate(nn.embed(codes[mask])):
                blocks[j][mask] = emb
            if with_output:
                h, p = nn.forward(cont[mask], codes[mask])
                if hidden is None:
                    hidden = np.zeros((n, h.shape[1]), dtype=np.float32)
                hidden[mask] = h
                prob[mask] = p

        columns = [frame[CONT_COLS].reset_index(drop=True)]
        for col, block in zip(CAT_COLS, blocks):
            columns.append(pd.DataFrame(block, columns=[f"{col}_emb{k}" for k in range(emb_dim)]))
        if with_output:
            if hidden is None:
                hidden = np.zeros((n, 1), dtype=np.float32)
            columns.append(pd.DataFrame(hidden, columns=[f"nn_hidden{k}" for k in range(hidden.shape[1])]))
            columns.append(pd.DataFrame({"nn_prob": np.nan_to_num(prob, nan=0.5)}))
        return pd.concat(columns, axis=1)

    nn_emb_train = embedded_frame(train_prep)
    nn_emb_test = embedded_frame(test_prep)
    nn_out_train = embedded_frame(train_prep, with_output=True)
    nn_out_test = embedded_frame(test_prep, with_output=True)
    return (
        clustered_embedded_frame,
        embedded_frame,
        nn_emb_test,
        nn_emb_train,
        nn_out_test,
        nn_out_train,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sur des clusters

    On entraîne des réseaux de neurones sur des clusters. On définit d'abord `InductiveClusterer` qui combine un modèle de clustering et un modèle de classification (typiquement plus-proche voisin) pour associer des points à un cluster quand ils n'ont pas été utilisés pour fitter le modèle de clustering.
    """)
    return


@app.class_definition
class InductiveClusterer:
    def __init__(self, clusterer, classifier):
        self.clusterer = clusterer
        self.classifier = classifier

    def fit(self, X, cluster_X=None):
        labels = self.clusterer.fit_predict(X if cluster_X is None else cluster_X)
        self.classifier.fit(X, labels)
        return labels

    def predict(self, X_new):
        return self.classifier.predict(X_new)


@app.cell
def _(
    AveragedEmbeddingNetwork,
    CATEGORICAL_FIELDS,
    CAT_COLS,
    CONT_COLS,
    NN_BATCH_SIZE,
    NN_BOOTSTRAP,
    NN_DROPOUT,
    NN_EMBEDDING_DIM,
    NN_EPOCHS,
    NN_HIDDEN,
    NN_PATIENCE,
    NeuralNetwork,
    cont_mean,
    cont_std,
    device,
    fit_idx,
    labels,
    np,
    pd,
    roc_auc_score,
    sklearn,
    test_prep,
    time,
    torch,
    train_network,
    train_prep,
    val_idx,
):
    def write_submission(y, tag):
        y = np.asarray(y, dtype=np.float64).ravel()
        path = f"submission_{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}_{tag}.csv"
        pd.DataFrame({"id": test_prep["id"].to_numpy(), "accepted": y}).set_index("id").to_csv(
            path, float_format="%.4f"
        )
        print("wrote", path)
        return path

    def write_cluster_nn_submission(nns, clusterer, clustering_features, tag):
        test_cont = test_prep[CONT_COLS].to_numpy(dtype=np.float32)
        test_cat = test_prep[CAT_COLS].to_numpy(dtype=np.float32)
        clusters = clusterer.predict(clustering_features(test_cont, test_cat))
        cont = torch.as_tensor(
            (test_prep[CONT_COLS].to_numpy(dtype=np.float64) - cont_mean) / cont_std,
            dtype=torch.float32,
        )
        cat = torch.as_tensor(test_prep[CAT_COLS].to_numpy(dtype=np.int64), dtype=torch.long)
        preds = np.full(len(test_prep), np.nan, dtype=np.float32)
        for i, nn in enumerate(nns):
            if nn is None:
                continue
            mask = clusters == i
            if not np.any(mask):
                continue
            preds[mask] = nn.forward(cont[mask], cat[mask])[1]
        if np.isnan(preds).any():
            preds = np.where(np.isfinite(preds), preds, np.nanmean(preds))
        return write_submission(preds, tag)

    def _positive_proba(model, X):
        proba = model.predict_proba(X)
        proba = np.asarray(proba)
        return proba[:, 1] if proba.ndim == 2 else proba.ravel()

    def write_tree_submission(models, X_test, tag, test_clusters=None):
        if hasattr(X_test, "fillna"):
            X_test = X_test.fillna(-1)
        if not isinstance(models, dict):
            y = _positive_proba(models, X_test)
        else:
            y = np.full(len(X_test), np.nan, dtype=np.float64)
            test_clusters = np.asarray(test_clusters)
            for cid, model in models.items():
                mask = test_clusters == cid
                if not mask.any():
                    continue
                rows = X_test.iloc[mask] if hasattr(X_test, "iloc") else X_test[mask]
                y[mask] = _positive_proba(model, rows)
            if np.isnan(y).any():
                y = np.where(np.isfinite(y), y, np.nanmean(y))
        return write_submission(y, tag)

    def train_clustered_networks(n_clusters, n_components, reduce=None, n_epochs=NN_EPOCHS, n_bootstraps=NN_BOOTSTRAP):
        cluster_fit_cap = min(50_000, len(fit_idx))
        train_cont = train_prep[CONT_COLS].to_numpy(dtype=np.float32)
        train_cat = train_prep[CAT_COLS].to_numpy(dtype=np.float32)
        birch_idx = fit_idx[:cluster_fit_cap]
        fit_cont, fit_cat = train_cont[birch_idx], train_cat[birch_idx]

        cont_m, cont_s = np.nanmean(fit_cont, axis=0), np.nanstd(fit_cont, axis=0)
        cont_s[cont_s < 1e-6] = 1
        cat_m, cat_s = np.nanmean(fit_cat, axis=0), np.nanstd(fit_cat, axis=0)
        cat_s[cat_s < 1e-6] = 1

        def raw_clustering_features(cont, cat):
            cont = np.nan_to_num((cont - cont_m) / cont_s, 0)
            cat = np.nan_to_num((cat - cat_m) / cat_s, -1)
            return np.concat((cont, cat), axis=1)

        X_birch = raw_clustering_features(fit_cont, fit_cat)
        method = None if reduce is None else str(reduce).lower()
        dim = min(int(n_components), X_birch.shape[0], X_birch.shape[1])
        reducer, X_cluster, assign_reduced = None, X_birch, False

        if method == "pca":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=dim, random_state=0).fit(X_birch)
            X_cluster = reducer.transform(X_birch)
            assign_reduced = True
        elif method == "umap":
            import umap
            reducer = umap.UMAP(n_components=dim, random_state=0).fit(X_birch)
            X_cluster = reducer.transform(X_birch)
            assign_reduced = True
        elif method == "tsne":
            from sklearn.manifold import TSNE
            X_cluster = TSNE(n_components=min(dim, 3), random_state=0).fit_transform(X_birch)
        elif method is not None:
            raise ValueError(f"reduce must be None, 'pca', 'umap' or 'tsne', got {reduce!r}")

        print(
            f"clustering on {X_cluster.shape[1]} dimensions"
            + (f" after {method}" if method else " (no reduction)")
        )

        def clustering_features(cont, cat):
            X = raw_clustering_features(cont, cat)
            return reducer.transform(X) if assign_reduced else X

        clusterer = InductiveClusterer(
            clusterer=sklearn.cluster.Birch(n_clusters=n_clusters),
            classifier=sklearn.neighbors.KNeighborsClassifier(n_neighbors=1),
        )
        clusterer.fit(clustering_features(fit_cont, fit_cat), cluster_X=X_cluster)

        fit_clusters = clusterer.predict(clustering_features(train_cont[fit_idx], train_cat[fit_idx]))
        val_clusters = clusterer.predict(clustering_features(train_cont[val_idx], train_cat[val_idx]))

        cont = torch.as_tensor(
            (train_prep[CONT_COLS].to_numpy(dtype=np.float64) - cont_mean) / cont_std,
            dtype=torch.float32,
            device=device,
        )
        cat = torch.as_tensor(train_prep[CAT_COLS].to_numpy(dtype=np.int64), dtype=torch.long, device=device)
        y = torch.as_tensor(labels, dtype=torch.float32, device=device)

        n_models = max(n_bootstraps, 1)
        trained_cluster_nns = [None] * n_clusters
        cluster_nn_aucs = np.full(n_clusters, np.nan, dtype=np.float64)
        for i in range(n_clusters):
            indices = fit_idx[(fit_clusters == i).nonzero()[0]].copy()
            rng = np.random.default_rng(i)
            rng.shuffle(indices)
            n_fit = int(0.9 * len(indices))
            cluster_fit_idx, cluster_holdout_idx = indices[:n_fit], indices[n_fit:]
            print(
                f"for cluster {i}, training on {len(cluster_fit_idx)} samples, "
                f"early-stopping on {len(cluster_holdout_idx)} (from fit_idx only)."
            )
            if len(cluster_fit_idx) == 0 or len(cluster_holdout_idx) == 0:
                print(f"skipping cluster {i}: not enough samples for a 90/10 split.")
                continue

            models, histories = [], []
            for b in range(n_models):
                model = NeuralNetwork(
                    CATEGORICAL_FIELDS, len(CONT_COLS), NN_EMBEDDING_DIM, NN_HIDDEN,
                    dropout=NN_DROPOUT, residual=True,
                ).to(device)
                rows = (
                    cluster_fit_idx
                    if n_models == 1
                    else rng.choice(cluster_fit_idx, size=len(cluster_fit_idx), replace=True)
                )
                histories.append(train_network(
                    model, cont, cat, y, rows, cluster_holdout_idx,
                    epochs=n_epochs, batch_size=NN_BATCH_SIZE, learning_rate=2e-5,
                    patience=NN_PATIENCE,
                    label="" if n_models == 1 else f"cluster {i + 1}/{n_clusters}, model {b + 1}/{n_models}, ",
                    log_rate=25,
                ))
                model.eval()
                models.append(model)

            aucs = np.array([max(h) for h in histories], dtype=np.float64)
            cluster_nn = AveragedEmbeddingNetwork(models, weights=aucs / aucs.sum())
            trained_cluster_nns[i] = cluster_nn
            holdout_preds = cluster_nn.forward(cont[cluster_holdout_idx], cat[cluster_holdout_idx])[1]
            holdout_auc = roc_auc_score(y[cluster_holdout_idx].detach().cpu(), holdout_preds)
            cluster_nn_aucs[i] = holdout_auc
            print(f"cluster {i + 1}/{n_clusters} holdout roc auc", holdout_auc)

        preds = np.full(len(val_idx), np.nan, dtype=np.float32)
        for i, nn in enumerate(trained_cluster_nns):
            if nn is None:
                continue
            mask = val_clusters == i
            if not np.any(mask):
                continue
            preds[mask] = nn.forward(cont[val_idx[mask]], cat[val_idx[mask]])[1]

        scored = np.isfinite(preds)
        global_auc = roc_auc_score(labels[val_idx][scored], preds[scored])
        print("final roc auc", global_auc)
        method = None if reduce is None else str(reduce).lower()
        write_cluster_nn_submission(
            trained_cluster_nns,
            clusterer,
            clustering_features,
            tag=f"cluster-nn-{n_clusters}-{method or 'none'}-{int(n_components)}",
        )
        clusterer.nn_val_aucs = cluster_nn_aucs
        return global_auc, trained_cluster_nns, clusterer, clustering_features, fit_clusters, val_clusters

    return train_clustered_networks, write_submission, write_tree_submission


@app.cell
def _(
    CAT_COLS,
    CONT_COLS,
    CatBoostClassifier,
    TREE_FEATURES,
    cont_mean,
    cont_std,
    fit_idx,
    labels,
    np,
    roc_auc_score,
    test_prep,
    torch,
    train_clustered_networks,
    train_prep,
    val_idx,
    write_submission,
):
    def train_clustered_nn_and_trees(
        n_clusters,
        n_components,
        reduce=None,
        nn_auc_threshold=0.9,
        tree_params=None,
        **nn_kwargs,
    ):
        """Train per-cluster networks and CatBoost trees, then predict on `test_prep`.

        A cluster uses its neural net on the test set if that net's holdout ROC AUC is
        strictly above `nn_auc_threshold`; otherwise it uses the cluster tree.
        """
        tree_params = dict(
            iterations=500,
            depth=8,
            learning_rate=0.05,
            verbose=0,
            random_seed=0,
            allow_writing_files=False,
        ) | (tree_params or {})

        clustered = train_clustered_networks(
            n_clusters, n_components, reduce=reduce, **nn_kwargs
        )
        _, nns, clusterer, clustering_features, fit_clusters, val_clusters = clustered
        nn_aucs = np.asarray(clusterer.nn_val_aucs, dtype=np.float64)

        X_fit = train_prep[TREE_FEATURES].iloc[fit_idx].fillna(-1)
        y_fit = labels[fit_idx]
        X_val = train_prep[TREE_FEATURES].iloc[val_idx].fillna(-1)
        y_val = labels[val_idx]

        trees = {}
        for cid in range(n_clusters):
            tr, va = fit_clusters == cid, val_clusters == cid
            if not np.any(tr):
                print(f"cluster {cid}: no fit rows, skipping tree")
                continue
            model = CatBoostClassifier(**tree_params)
            model.fit(X_fit.iloc[tr], y_fit[tr])
            trees[cid] = model
            if np.any(va) and np.unique(y_val[va]).size > 1:
                tree_auc = roc_auc_score(y_val[va], model.predict_proba(X_val.iloc[va])[:, 1])
                print(f"cluster {cid} tree val roc auc {tree_auc:.4f}")

        test_clusters = clusterer.predict(clustering_features(
            test_prep[CONT_COLS].to_numpy(dtype=np.float32),
            test_prep[CAT_COLS].to_numpy(dtype=np.float32),
        ))
        cont = torch.as_tensor(
            (test_prep[CONT_COLS].to_numpy(dtype=np.float64) - cont_mean) / cont_std,
            dtype=torch.float32,
        )
        cat = torch.as_tensor(test_prep[CAT_COLS].to_numpy(dtype=np.int64), dtype=torch.long)
        X_test = test_prep[TREE_FEATURES].fillna(-1)
        preds = np.full(len(test_prep), np.nan, dtype=np.float64)

        for cid in np.unique(test_clusters):
            cid = int(cid)
            mask = test_clusters == cid
            nn_auc = nn_aucs[cid] if 0 <= cid < len(nn_aucs) else np.nan
            use_nn = (
                0 <= cid < len(nns)
                and nns[cid] is not None
                and np.isfinite(nn_auc)
                and nn_auc > nn_auc_threshold
            )
            print(
                f"cluster {cid}: nn holdout auc {nn_auc:.4f}, "
                f"using {'neural net' if use_nn else 'tree'} on {int(mask.sum())} test rows"
            )
            if use_nn:
                preds[mask] = nns[cid].forward(cont[mask], cat[mask])[1]
            elif cid in trees:
                preds[mask] = trees[cid].predict_proba(X_test.iloc[mask])[:, 1]

        if np.isnan(preds).any():
            preds = np.where(np.isfinite(preds), preds, np.nanmean(preds))

        path = write_submission(
            preds, f"cluster-nn-or-tree-{n_clusters}-{reduce or 'none'}-{int(n_components)}"
        )
        return clustered, trees, preds, path

    return (train_clustered_nn_and_trees,)


@app.cell
def _(train_clustered_nn_and_trees):
    train_clustered_nn_and_trees(3, 4, "pca", tree_params={
          "iterations": 1000,
          "depth": 8,
          "learning_rate": 0.1,
          "l2_leaf_reg": 0.19
        })
    return


@app.cell
def _(train_clustered_networks):
    _N_CLUSTERS = 3
    _CLUSTER_REDUCE = "tsne"          # None | "pca" | "umap" | "tsne"
    _CLUSTER_REDUCE_DIMS = 2

    _clustered_nn = train_clustered_networks(
        _N_CLUSTERS, _CLUSTER_REDUCE_DIMS, reduce=_CLUSTER_REDUCE, n_bootstraps=15, n_epochs=1000
    )
    (
        _cluster_nn_auc,
        _trained_cluster_nns,
        _clusterer,
        _clustering_features,
        _training_set_clusters,
        _validation_set_clusters,
    ) = _clustered_nn

    _cluster_nn_auc
    return


@app.cell
def _(train_clustered_networks):
    _N_CLUSTERS = 2
    _CLUSTER_REDUCE = "tsne"          # None | "pca" | "umap" | "tsne"
    _CLUSTER_REDUCE_DIMS = 2

    (
        _cluster_nn_auc,
        _trained_cluster_nns,
        _clusterer,
        _clustering_features,
        _training_set_clusters,
        _validation_set_clusters,
    ) = train_clustered_networks(_N_CLUSTERS, _CLUSTER_REDUCE_DIMS, reduce=_CLUSTER_REDUCE, n_bootstraps=2)

    _cluster_nn_auc
    return


@app.cell
def _(train_clustered_networks):
    _N_CLUSTERS = 2
    _CLUSTER_REDUCE = "umap"          # None | "pca" | "umap" | "tsne"
    _CLUSTER_REDUCE_DIMS = 2

    (
        _cluster_nn_auc,
        _trained_cluster_nns,
        _clusterer,
        _clustering_features,
        _training_set_clusters,
        _validation_set_clusters,
    ) = train_clustered_networks(_N_CLUSTERS, _CLUSTER_REDUCE_DIMS, reduce=_CLUSTER_REDUCE, n_bootstraps=2)

    _cluster_nn_auc
    return


@app.cell
def _(train_clustered_networks):
    _N_CLUSTERS = 2
    _CLUSTER_REDUCE = "pca"          # None | "pca" | "umap" | "tsne"
    _CLUSTER_REDUCE_DIMS = 2

    (
        _cluster_nn_auc,
        _trained_cluster_nns,
        _clusterer,
        _clustering_features,
        _training_set_clusters,
        _validation_set_clusters,
    ) = train_clustered_networks(_N_CLUSTERS, _CLUSTER_REDUCE_DIMS, reduce=_CLUSTER_REDUCE, n_bootstraps=2)

    _cluster_nn_auc
    return


@app.cell
def _(train_clustered_networks):
    _N_CLUSTERS = 4
    _CLUSTER_REDUCE = "tsne"          # None | "pca" | "umap" | "tsne"
    _CLUSTER_REDUCE_DIMS = 2

    (
        _cluster_nn_auc,
        _trained_cluster_nns,
        _clusterer,
        _clustering_features,
        _training_set_clusters,
        _validation_set_clusters,
    ) = train_clustered_networks(_N_CLUSTERS, _CLUSTER_REDUCE_DIMS, reduce=_CLUSTER_REDUCE, n_bootstraps=2)

    _cluster_nn_auc
    return


@app.cell
def _(train_clustered_networks):
    _N_CLUSTERS = 5
    _CLUSTER_REDUCE = "tsne"          # None | "pca" | "umap" | "tsne"
    _CLUSTER_REDUCE_DIMS = 2

    (
        _cluster_nn_auc,
        _trained_cluster_nns,
        _clusterer,
        _clustering_features,
        _training_set_clusters,
        _validation_set_clusters,
    ) = train_clustered_networks(_N_CLUSTERS, _CLUSTER_REDUCE_DIMS, reduce=_CLUSTER_REDUCE, n_bootstraps=2)

    _cluster_nn_auc
    return


@app.cell
def _(train_clustered_networks):
    _N_CLUSTERS = 5
    _CLUSTER_REDUCE = "umap"          # None | "pca" | "umap" | "tsne"
    _CLUSTER_REDUCE_DIMS = 2

    (
        _cluster_nn_auc,
        _trained_cluster_nns,
        _clusterer,
        _clustering_features,
        _training_set_clusters,
        _validation_set_clusters,
    ) = train_clustered_networks(_N_CLUSTERS, _CLUSTER_REDUCE_DIMS, reduce=_CLUSTER_REDUCE, n_bootstraps=2)

    _cluster_nn_auc
    return


@app.cell
def _(train_clustered_networks):
    _N_CLUSTERS = 5
    _CLUSTER_REDUCE = "pca"          # None | "pca" | "umap" | "tsne"
    _CLUSTER_REDUCE_DIMS = 2

    (
        _cluster_nn_auc,
        _trained_cluster_nns,
        _clusterer,
        _clustering_features,
        _training_set_clusters,
        _validation_set_clusters,
    ) = train_clustered_networks(_N_CLUSTERS, _CLUSTER_REDUCE_DIMS, reduce=_CLUSTER_REDUCE, n_bootstraps=2)

    _cluster_nn_auc
    return


@app.cell
def _(train_clustered_networks):
    _N_CLUSTERS = 2
    _CLUSTER_REDUCE = "tsne"          # None | "pca" | "umap" | "tsne"
    _CLUSTER_REDUCE_DIMS = 5

    (
        _cluster_nn_auc,
        _trained_cluster_nns,
        _clusterer,
        _clustering_features,
        _training_set_clusters,
        _validation_set_clusters,
    ) = train_clustered_networks(_N_CLUSTERS, _CLUSTER_REDUCE_DIMS, reduce=_CLUSTER_REDUCE, n_bootstraps=2)

    _cluster_nn_auc
    return


@app.cell
def _(train_clustered_networks):
    _N_CLUSTERS = 2
    _CLUSTER_REDUCE = "umap"          # None | "pca" | "umap" | "tsne"
    _CLUSTER_REDUCE_DIMS = 5

    (
        _cluster_nn_auc,
        _trained_cluster_nns,
        _clusterer,
        _clustering_features,
        _training_set_clusters,
        _validation_set_clusters,
    ) = train_clustered_networks(_N_CLUSTERS, _CLUSTER_REDUCE_DIMS, reduce=_CLUSTER_REDUCE, n_bootstraps=2)

    _cluster_nn_auc
    return


@app.cell
def _(train_clustered_networks):
    _N_CLUSTERS = 2
    _CLUSTER_REDUCE = "pca"          # None | "pca" | "umap" | "tsne"
    _CLUSTER_REDUCE_DIMS = 5

    (
        _cluster_nn_auc,
        trained_cluster_nns_,
        _clusterer,
        _clustering_features,
        _training_set_clusters,
        _validation_set_clusters,
    ) = train_clustered_networks(_N_CLUSTERS, _CLUSTER_REDUCE_DIMS, reduce=_CLUSTER_REDUCE, n_bootstraps=15, n_epochs=1000)

    _cluster_nn_auc
    return


@app.cell
def _(train_clustered_networks):
    _N_CLUSTERS = 2
    _CLUSTER_REDUCE = "pca"          # None | "pca" | "umap" | "tsne"
    _CLUSTER_REDUCE_DIMS = 4

    clustered_nn = train_clustered_networks(_N_CLUSTERS, _CLUSTER_REDUCE_DIMS, reduce=_CLUSTER_REDUCE, n_bootstraps=20, n_epochs=1000)

    (
        cluster_nn_auc,
        trained_cluster_nns,
        clusterer,
        clustering_features,
        training_set_clusters,
        validation_set_clusters,
    ) = clustered_nn
    return (
        clustered_nn,
        clusterer,
        clustering_features,
        training_set_clusters,
        validation_set_clusters,
    )


@app.cell
def _(train_clustered_networks):
    _N_CLUSTERS = 2
    _CLUSTER_REDUCE = "pca"          # None | "pca" | "umap" | "tsne"
    _CLUSTER_REDUCE_DIMS = 6

    (
        _cluster_nn_auc,
        _trained_cluster_nns,
        _clusterer,
        _clustering_features,
        _training_set_clusters,
        _validation_set_clusters,
    ) = train_clustered_networks(_N_CLUSTERS, _CLUSTER_REDUCE_DIMS, reduce=_CLUSTER_REDUCE, n_bootstraps=2)

    _cluster_nn_auc
    return


@app.cell
def _(train_clustered_networks):
    _N_CLUSTERS = 2
    _CLUSTER_REDUCE = "pca"          # None | "pca" | "umap" | "tsne"
    _CLUSTER_REDUCE_DIMS = 7

    (
        _cluster_nn_auc,
        _trained_cluster_nns,
        _clusterer,
        _clustering_features,
        _training_set_clusters,
        _validation_set_clusters,
    ) = train_clustered_networks(_N_CLUSTERS, _CLUSTER_REDUCE_DIMS, reduce=_CLUSTER_REDUCE, n_bootstraps=2)

    _cluster_nn_auc
    return


@app.cell
def _(train_clustered_networks):
    _N_CLUSTERS = 2
    _CLUSTER_REDUCE = "pca"          # None | "pca" | "umap" | "tsne"
    _CLUSTER_REDUCE_DIMS = 8

    (
        _cluster_nn_auc,
        _trained_cluster_nns,
        _clusterer,
        _clustering_features,
        _training_set_clusters,
        _validation_set_clusters,
    ) = train_clustered_networks(_N_CLUSTERS, _CLUSTER_REDUCE_DIMS, reduce=_CLUSTER_REDUCE, n_bootstraps=2)

    _cluster_nn_auc
    return


@app.cell
def _(train_clustered_networks):
    _N_CLUSTERS = 4
    _CLUSTER_REDUCE = "tsne"          # None | "pca" | "umap" | "tsne"
    _CLUSTER_REDUCE_DIMS = 5

    (
        _cluster_nn_auc,
        _trained_cluster_nns,
        _clusterer,
        _clustering_features,
        _training_set_clusters,
        _validation_set_clusters,
    ) = train_clustered_networks(_N_CLUSTERS, _CLUSTER_REDUCE_DIMS, reduce=_CLUSTER_REDUCE, n_bootstraps=2)

    _cluster_nn_auc
    return


@app.cell
def _(train_clustered_networks):
    _N_CLUSTERS = 2
    _CLUSTER_REDUCE = "tsne"          # None | "pca" | "umap" | "tsne"
    _CLUSTER_REDUCE_DIMS = 10

    (
        _cluster_nn_auc,
        _trained_cluster_nns,
        _clusterer,
        _clustering_features,
        _training_set_clusters,
        _validation_set_clusters,
    ) = train_clustered_networks(_N_CLUSTERS, _CLUSTER_REDUCE_DIMS, reduce=_CLUSTER_REDUCE, n_bootstraps=2)

    _cluster_nn_auc
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Encodage des colonnes catégorielles par mean-target encoding et variantes

    Cette section utilise des méthodes d'encodage déterministes, construites à partir de statistiques du jeu d'entraînement et résumées dans https://arxiv.org/pdf/2307.09191v2.
    """)
    return


@app.cell
def _(np, pd):
    def level_counts(codes, n_levels, y=None):
        counts = np.bincount(codes, minlength=n_levels).astype(np.float64)
        sums = np.bincount(codes, weights=y, minlength=n_levels) if y is not None else None
        return counts, sums

    def mean_target(counts, sums, prior):
        """MT, falling back on the prior for the levels absent from the training rows."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(counts > 0, sums / np.maximum(counts, 1), prior)

    def pre_binned_levels(counts, threshold):
        """PBMT bins: as many bins as possible with relative frequency at least `threshold`.

        Accumulating the levels greedily by increasing frequency is optimal for that objective.
        """
        total = counts.sum()
        bin_of_level = np.zeros(len(counts), dtype=np.int64)
        current, mass, complete = 0, 0.0, []
        for level in np.argsort(counts, kind="stable"):
            bin_of_level[level] = current
            mass += counts[level]
            if total > 0 and mass / total >= threshold:
                complete.append(current)
                current += 1
                mass = 0.0
        if mass > 0 and complete:  # the leftover levels join the last complete bin
            bin_of_level[bin_of_level == current] = complete[-1]
        return bin_of_level

    class TargetEncoder:
        """Mean-target encoding and the variants of section 3.3 of arXiv:2307.09191v2.

        `fit_transform` returns the in-sample encoding, which differs from `transform` for the
        data-constraining encoders CB and CVMT.
        """

        KINDS = ("MT", "WoE", "DMT", "PBMT", "ME", "GLMM", "CB", "CVMT", "BUMT")

        def __init__(self, kind, cardinalities, *, bins=5, threshold=0.01, weight=1.0, folds=5, smoothing=1.0, random_state=0):
            if kind not in self.KINDS:
                raise ValueError(f"unknown encoder {kind!r}")
            self.kind = kind
            self.cardinalities = cardinalities
            self.bins = bins
            self.threshold = threshold
            self.weight = weight
            self.folds = folds
            self.smoothing = smoothing
            self.random_state = random_state

        def fit(self, frame, y):
            y = np.asarray(y, dtype=np.float64)
            self.prior_ = float(y.mean())
            self.columns_ = list(frame.columns)
            self.fold_of_row_ = np.random.default_rng(self.random_state).integers(0, self.folds, size=len(y))
            self.tables_ = {}
            for col in self.columns_:
                codes = frame[col].to_numpy(dtype=np.int64)
                n_levels = int(max(self.cardinalities[col], codes.max() + 1))
                counts, sums = level_counts(codes, n_levels, y)
                self.tables_[col] = self._table(codes, y, counts, sums, n_levels)
            return self

        def _table(self, codes, y, counts, sums, n_levels):
            """The level -> encoding lookup table of one column, with one column per output."""
            prior = self.prior_
            mt = mean_target(counts, sums, prior)
            if self.kind in ("MT", "CVMT"):
                return mt[:, None]
            if self.kind == "WoE":
                p = (sums + 0.5) / (counts + 1.0)  # Laplace smoothing keeps the logit finite
                return np.log(p / (1.0 - p))[:, None]
            if self.kind == "ME":
                return ((counts * mt + self.weight * prior) / (self.weight + counts))[:, None]
            if self.kind == "GLMM":
                return self._empirical_bayes(counts, mt)[:, None]
            if self.kind == "CB":
                return ((sums + self.smoothing * prior) / (counts + self.smoothing))[:, None]
            if self.kind == "DMT":
                seen = counts > 0
                low, high = (mt[seen].min(), mt[seen].max()) if seen.any() else (prior, prior)
                width = (high - low) / self.bins
                if width <= 0:
                    return np.full((n_levels, 1), low)
                index = np.clip(np.floor((mt - low) / width), 0, self.bins - 1)
                return (low + index * width)[:, None]
            if self.kind == "PBMT":
                bin_of_level = pre_binned_levels(counts, self.threshold)
                bin_counts, bin_sums = level_counts(bin_of_level[codes], bin_of_level.max() + 1, y)
                return mean_target(bin_counts, bin_sums, prior)[bin_of_level][:, None]
            if self.kind == "BUMT":
                tables = []
                for j in range(self.folds):
                    mask = self.fold_of_row_ == j
                    fold_counts, fold_sums = level_counts(codes[mask], n_levels, y[mask])
                    tables.append(mean_target(fold_counts, fold_sums, prior))
                return np.stack(tables, axis=1)
            raise AssertionError(self.kind)

        def _empirical_bayes(self, counts, mt):
            prior = self.prior_
            seen = counts > 0
            sigma2 = max(prior * (1.0 - prior), 1e-12)
            if seen.sum() < 2:
                return np.full(len(counts), prior)
            weights = counts[seen] / counts[seen].sum()
            between = float((weights * (mt[seen] - prior) ** 2).sum())
            tau2 = max(between - sigma2 * float((weights / counts[seen]).sum()), 0.0)
            shrink = np.zeros(len(counts))
            if tau2 > 0:
                shrink[seen] = tau2 / (tau2 + sigma2 / counts[seen])
            return prior + shrink * (mt - prior)

        def _names(self, col, width):
            return [f"{col}_{self.kind}"] if width == 1 else [f"{col}_{self.kind}{j}" for j in range(width)]

        def transform(self, frame):
            out = {}
            for col in self.columns_:
                table = self.tables_[col]
                codes = np.clip(frame[col].to_numpy(dtype=np.int64), 0, len(table) - 1)
                for name, column in zip(self._names(col, table.shape[1]), table[codes].T):
                    out[name] = column
            return pd.DataFrame(out, index=pd.RangeIndex(len(frame)))

        def fit_transform(self, frame, y):
            self.fit(frame, y)
            if self.kind not in ("CB", "CVMT"):
                return self.transform(frame)

            y = np.asarray(y, dtype=np.float64)
            out = {}
            for col in self.columns_:
                codes = frame[col].to_numpy(dtype=np.int64)
                n_levels = len(self.tables_[col])
                out[self._names(col, 1)[0]] = (
                    self._catboost_encoding(codes, y, n_levels)
                    if self.kind == "CB"
                    else self._cross_validated_encoding(codes, y, n_levels)
                )
            return pd.DataFrame(out, index=pd.RangeIndex(len(frame)))

        def _catboost_encoding(self, codes, y, n_levels):
            """Every occurrence of a level is encoded with the average target of its earlier
            occurrences in a random permutation of the rows."""
            permutation = np.random.default_rng(self.random_state + 1).permutation(len(codes))
            running_count, running_sum = np.zeros(n_levels), np.zeros(n_levels)
            values = np.empty(len(codes))
            a, prior = self.smoothing, self.prior_
            for i in permutation:
                level = codes[i]
                values[i] = (running_sum[level] + a * prior) / (running_count[level] + a)
                running_count[level] += 1.0
                running_sum[level] += y[i]
            return values

        def _cross_validated_encoding(self, codes, y, n_levels):
            """Every fold is encoded with a mean-target fitted on the other folds."""
            total_counts, total_sums = level_counts(codes, n_levels, y)
            values = np.empty(len(codes))
            for j in range(self.folds):
                mask = self.fold_of_row_ == j
                if not mask.any():
                    continue
                fold_counts, fold_sums = level_counts(codes[mask], n_levels, y[mask])
                values[mask] = mean_target(total_counts - fold_counts, total_sums - fold_sums, self.prior_)[codes[mask]]
            return values

    return (TargetEncoder,)


@app.cell
def _():
    TARGET_ENCODERS = {
        "target MT": ("MT", {}),
        "target WoE": ("WoE", {}),
        "target DMT(B=2)": ("DMT", dict(bins=2)),
        "target DMT(B=5)": ("DMT", dict(bins=5)),
        "target DMT(B=10)": ("DMT", dict(bins=10)),
        "target PBMT(0.001)": ("PBMT", dict(threshold=0.001)),
        "target PBMT(0.01)": ("PBMT", dict(threshold=0.01)),
        "target PBMT(0.1)": ("PBMT", dict(threshold=0.1)),
        "target ME(w=0.1)": ("ME", dict(weight=0.1)),
        "target ME(w=1)": ("ME", dict(weight=1.0)),
        "target ME(w=10)": ("ME", dict(weight=10.0)),
        "target GLMM": ("GLMM", {}),
        "target CB": ("CB", {}),
        "target CV2MT": ("CVMT", dict(folds=2)),
        "target CV5MT": ("CVMT", dict(folds=5)),
        "target CV10MT": ("CVMT", dict(folds=10)),
        "target BU2MT": ("BUMT", dict(folds=2)),
        "target BU5MT": ("BUMT", dict(folds=5)),
    }
    return (TARGET_ENCODERS,)


@app.cell
def _(
    embedded_frame,
    nn_emb_test,
    nn_emb_train,
    nn_out_test,
    nn_out_train,
    test_prep,
    train_prep,
    trained_nn_attn,
):
    if trained_nn_attn is not None:
        nn_attn_emb_train = embedded_frame(train_prep, network=trained_nn_attn)
        nn_attn_emb_test = embedded_frame(test_prep, network=trained_nn_attn)
        nn_attn_out_train = embedded_frame(train_prep, with_output=True, network=trained_nn_attn)
        nn_attn_out_test = embedded_frame(test_prep, with_output=True, network=trained_nn_attn)
    else:
        nn_attn_emb_train, nn_attn_emb_test = nn_emb_train, nn_emb_test
        nn_attn_out_train, nn_attn_out_test = nn_out_train, nn_out_test
    return (
        nn_attn_emb_test,
        nn_attn_emb_train,
        nn_attn_out_test,
        nn_attn_out_train,
    )


@app.cell
def _(
    CATEGORICAL_FIELDS,
    CAT_COLS,
    CONT_COLS,
    NamedTuple,
    TARGET_ENCODERS,
    TREE_FEATURES,
    TREE_LOW_CARD_CATEGORICAL,
    TargetEncoder,
    clustered_embedded_frame,
    labels,
    nn_attn_emb_test,
    nn_attn_emb_train,
    nn_attn_out_test,
    nn_attn_out_train,
    nn_emb_test,
    nn_emb_train,
    nn_out_test,
    nn_out_train,
    np,
    pd,
    test_prep,
    train_prep,
    val_idx,
):
    class Dataset(NamedTuple):
        X_train: pd.DataFrame
        y_train: np.ndarray
        X_val: pd.DataFrame
        y_val: np.ndarray
        X_test: pd.DataFrame
        categorical_features: list

    UNSUPERVISED_INPUTS = {
        "default codes": (lambda: train_prep[TREE_FEATURES], lambda: test_prep[TREE_FEATURES], TREE_LOW_CARD_CATEGORICAL),
        "nn embeddings": (lambda: nn_emb_train, lambda: nn_emb_test, []),
        "nn embeddings + output": (lambda: nn_out_train, lambda: nn_out_test, []),
        "nn attention embeddings": (lambda: nn_attn_emb_train, lambda: nn_attn_emb_test, []),
        "nn attention embeddings + output": (lambda: nn_attn_out_train, lambda: nn_attn_out_test, []),
    }

    def _as_float(frame):
        return np.nan_to_num(frame.to_numpy(dtype=np.float64), nan=0.0)

    def _reduce_splits(X_train, X_val, X_test, method, n_components):
        """Fit a reducer on train, then map val and test. t-SNE has no transform, so val/test
        coordinates are the embedding of the nearest training row."""
        method = method.lower()
        Xtr = _as_float(X_train)
        n_components = min(int(n_components), Xtr.shape[0], Xtr.shape[1])
        if n_components < 1:
            raise ValueError("n_components must be at least 1")

        if method == "pca":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=n_components, random_state=0).fit(Xtr)
            Z_train = reducer.transform(Xtr)
            Z_val = reducer.transform(_as_float(X_val))
            Z_test = reducer.transform(_as_float(X_test))
        elif method == "umap":
            import umap
            reducer = umap.UMAP(n_components=n_components, random_state=0).fit(Xtr)
            Z_train = reducer.transform(Xtr)
            Z_val = reducer.transform(_as_float(X_val))
            Z_test = reducer.transform(_as_float(X_test))
        elif method == "tsne":
            from sklearn.manifold import TSNE
            from sklearn.neighbors import KNeighborsRegressor
            n_components = min(n_components, 3)
            Z_train = TSNE(n_components=n_components, random_state=0).fit_transform(Xtr)
            _knn = KNeighborsRegressor(n_neighbors=1).fit(Xtr, Z_train)
            Z_val = _knn.predict(_as_float(X_val))
            Z_test = _knn.predict(_as_float(X_test))
        else:
            raise ValueError(f"reduce must be None, 'pca', 'umap' or 'tsne', got {method!r}")

        names = [f"{method}{j}" for j in range(Z_train.shape[1])]
        return (
            pd.DataFrame(Z_train, columns=names),
            pd.DataFrame(Z_val, columns=names),
            pd.DataFrame(Z_test, columns=names),
        )

    def make_dataset(kind, train_idx, reduce=None, n_components=2, clustered_nn=None):
        """Fit / validation / test matrices for one input representation.

        A supervised encoder only ever sees `train_idx`, so no validation label reaches a feature.
        If `reduce` is `"pca"`, `"umap"` or `"tsne"`, the three matrices are projected to
        `n_components` dimensions; the reducer is fitted on the training rows only.
        If `clustered_nn` is the tuple returned by `train_clustered_networks`, the nn-embedding
        inputs are built by routing each row to that cluster's network.
        """
        y_train = labels[train_idx]

        if clustered_nn is not None and kind in ("nn embeddings", "nn embeddings + output"):
            _with_output = kind == "nn embeddings + output"
            _frame = clustered_embedded_frame(train_prep, clustered_nn, with_output=_with_output)
            X_train, X_val, X_test = (
                _frame.iloc[train_idx],
                _frame.iloc[val_idx],
                clustered_embedded_frame(test_prep, clustered_nn, with_output=_with_output),
            )
            _cats = []
        elif kind in UNSUPERVISED_INPUTS:
            _build_train, _build_test, _cats = UNSUPERVISED_INPUTS[kind]
            _frame = _build_train()
            X_train, X_val, X_test = _frame.iloc[train_idx], _frame.iloc[val_idx], _build_test()
        elif kind in TARGET_ENCODERS:
            _encoder_kind, _params = TARGET_ENCODERS[kind]
            _encoder = TargetEncoder(_encoder_kind, CATEGORICAL_FIELDS, **_params)
            _codes, _cont = train_prep[CAT_COLS], train_prep[CONT_COLS]

            X_train = pd.concat(
                [_cont.iloc[train_idx].reset_index(drop=True), _encoder.fit_transform(_codes.iloc[train_idx], y_train)],
                axis=1,
            )
            X_val = pd.concat(
                [_cont.iloc[val_idx].reset_index(drop=True), _encoder.transform(_codes.iloc[val_idx])],
                axis=1,
            )
            X_test = pd.concat(
                [test_prep[CONT_COLS].reset_index(drop=True), _encoder.transform(test_prep[CAT_COLS])],
                axis=1,
            )
            _cats = []
        else:
            raise ValueError(f"unknown input {kind!r}")

        X_train = X_train.reset_index(drop=True)
        X_val = X_val.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        if reduce is not None:
            X_train, X_val, X_test = _reduce_splits(X_train, X_val, X_test, reduce, n_components)
            _cats = []

        return Dataset(
            X_train,
            y_train,
            X_val,
            labels[val_idx],
            X_test,
            list(_cats),
        )

    return (make_dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Test sur un arbre de décision seul
    """)
    return


@app.cell
def _(
    DecisionTreeClassifier,
    TREE_FEATURES,
    fit_idx,
    labels,
    plt,
    roc_auc_score,
    train_prep,
    val_idx,
):
    _fit_X = train_prep[TREE_FEATURES].iloc[fit_idx].fillna(-1)
    _val_X = train_prep[TREE_FEATURES].iloc[val_idx].fillna(-1)
    _fit_y, _val_y = labels[fit_idx], labels[val_idx]

    _depths = list(range(2, 31, 2))
    _fit_aucs, _val_aucs = [], []
    for _d in _depths:
        _tree = DecisionTreeClassifier(max_depth=_d, min_samples_leaf=20, random_state=0)
        _tree.fit(_fit_X, _fit_y)
        _fit_aucs.append(roc_auc_score(_fit_y, _tree.predict_proba(_fit_X)[:, 1]))
        _val_aucs.append(roc_auc_score(_val_y, _tree.predict_proba(_val_X)[:, 1]))

    best_tree_depth = _depths[max(range(len(_depths)), key=lambda i: _val_aucs[i])]
    print(f"best depth: {best_tree_depth}, validation roc auc: {max(_val_aucs):.4f}")

    _fig, _ax = plt.subplots(figsize=(8, 5))
    _ax.plot(_depths, _fit_aucs, marker="o", label="fit")
    _ax.plot(_depths, _val_aucs, marker="o", label="validation")
    _ax.axvline(best_tree_depth, color="grey", linestyle=":")
    _ax.set_xlabel("max depth")
    _ax.set_ylabel("ROC AUC")
    _ax.set_title("single decision tree on the default codes")
    _ax.legend()
    _ax
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Réseaux de neurones sur les feuilles d'un arbre

    Un arbre de décision simple (profondeur et taille de feuille limitées) est fitté uniquement
    sur `fit_idx`. Chaque feuille reçoit ensuite un ensemble de réseaux entraînés par bootstrap,
    avec early-stopping sur un hold-out **interne à la feuille** (jamais `val_idx`). À
    l'inférence, chaque ligne de validation est envoyée au réseau de la feuille que l'arbre lui
    assigne. On affiche le ROC AUC de l'arbre seul, le ROC AUC de hold-out par feuille, et le
    ROC AUC global sur `val_idx`.
    """)
    return


@app.cell
def _(
    AveragedEmbeddingNetwork,
    CATEGORICAL_FIELDS,
    CAT_COLS,
    CONT_COLS,
    DecisionTreeClassifier,
    NN_BATCH_SIZE,
    NN_DROPOUT,
    NN_EMBEDDING_DIM,
    NN_EPOCHS,
    NN_HIDDEN,
    NN_PATIENCE,
    NeuralNetwork,
    TREE_FEATURES,
    cont_mean,
    cont_std,
    device,
    fit_idx,
    labels,
    np,
    roc_auc_score,
    torch,
    train_network,
    train_prep,
    val_idx,
):
    def train_leaf_networks(
        max_depth=3,
        min_samples_leaf=500,
        n_bootstraps=10,
        n_epochs=NN_EPOCHS,
        learning_rate=2e-5,
    ):
        _fit_X = train_prep[TREE_FEATURES].iloc[fit_idx].fillna(-1)
        _val_X = train_prep[TREE_FEATURES].iloc[val_idx].fillna(-1)
        _tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=0,
        )
        _tree.fit(_fit_X, labels[fit_idx])
        _tree_val_auc = roc_auc_score(labels[val_idx], _tree.predict_proba(_val_X)[:, 1])
        print(
            f"decision tree max_depth={max_depth}, min_samples_leaf={min_samples_leaf}: "
            f"val roc auc {_tree_val_auc:.4f}"
        )

        _fit_leaves = _tree.apply(_fit_X)
        _val_leaves = _tree.apply(_val_X)
        _leaf_ids = np.unique(_fit_leaves)
        print(f"{len(_leaf_ids)} leaves")

        _cont = torch.as_tensor(
            (train_prep[CONT_COLS].to_numpy(dtype=np.float64) - cont_mean) / cont_std,
            dtype=torch.float32,
            device=device,
        )
        _cat = torch.as_tensor(train_prep[CAT_COLS].to_numpy(dtype=np.int64), dtype=torch.long, device=device)
        _y = torch.as_tensor(labels, dtype=torch.float32, device=device)

        _n_models = max(n_bootstraps, 1)
        _leaf_nns = {}
        _leaf_holdout_aucs = {}
        for _k, _leaf in enumerate(_leaf_ids):
            _indices = fit_idx[_fit_leaves == _leaf].copy()
            _rng = np.random.default_rng(int(_leaf))
            _rng.shuffle(_indices)
            _n_fit = int(0.9 * len(_indices))
            _leaf_fit, _leaf_holdout = _indices[:_n_fit], _indices[_n_fit:]
            _holdout_y = labels[_leaf_holdout]
            print(
                f"leaf {_leaf} ({_k + 1}/{len(_leaf_ids)}): train {len(_leaf_fit)}, "
                f"holdout {len(_leaf_holdout)}"
            )
            if (
                len(_leaf_fit) == 0
                or len(_leaf_holdout) < 20
                or np.unique(_holdout_y).size < 2
                or np.unique(labels[_leaf_fit]).size < 2
            ):
                print(f"  skip: not enough samples or only one class; tree leaf probability will be used")
                continue

            _models, _histories = [], []
            for _b in range(_n_models):
                _model = NeuralNetwork(
                    CATEGORICAL_FIELDS, len(CONT_COLS), NN_EMBEDDING_DIM, NN_HIDDEN,
                    dropout=NN_DROPOUT, residual=True,
                ).to(device)
                _rows = (
                    _leaf_fit
                    if _n_models == 1
                    else _rng.choice(_leaf_fit, size=len(_leaf_fit), replace=True)
                )
                _histories.append(train_network(
                    _model, _cont, _cat, _y, _rows, _leaf_holdout,
                    epochs=n_epochs, batch_size=NN_BATCH_SIZE, learning_rate=learning_rate,
                    patience=NN_PATIENCE,
                    label="" if _n_models == 1 else (
                        f"leaf {_leaf} ({_k + 1}/{len(_leaf_ids)}), "
                        f"model {_b + 1}/{_n_models}, "
                    ),
                    log_rate=25,
                ))
                _model.eval()
                _models.append(_model)

            _aucs = np.array([max(_h) for _h in _histories], dtype=np.float64)
            _ens = AveragedEmbeddingNetwork(_models, weights=_aucs / _aucs.sum())
            _holdout_preds = _ens.forward(_cont[_leaf_holdout], _cat[_leaf_holdout])[1]
            _holdout_auc = roc_auc_score(_y[_leaf_holdout].detach().cpu(), _holdout_preds)
            _leaf_nns[int(_leaf)] = _ens
            _leaf_holdout_aucs[int(_leaf)] = _holdout_auc
            print(f"  holdout roc auc {_holdout_auc:.4f}  (max individual {float(_aucs.max()):.4f})")

        _tree_proba = _tree.predict_proba(_val_X)[:, 1]
        _preds = np.array(_tree_proba, dtype=np.float64, copy=True)
        for _leaf, _ens in _leaf_nns.items():
            _mask = _val_leaves == _leaf
            if not np.any(_mask):
                continue
            _preds[_mask] = _ens.forward(_cont[val_idx[_mask]], _cat[val_idx[_mask]])[1]

        _global_auc = roc_auc_score(labels[val_idx], _preds)
        print(f"\nleaf NNs covering {len(_leaf_nns)}/{len(_leaf_ids)} leaves")
        print(f"tree val roc auc          {_tree_val_auc:.4f}")
        print(f"leaf-NN global val roc auc {_global_auc:.4f}")
        for _leaf in _leaf_ids:
            _mask = _val_leaves == _leaf
            if not np.any(_mask) or np.unique(labels[val_idx[_mask]]).size < 2:
                continue
            _leaf_val_auc = roc_auc_score(labels[val_idx[_mask]], _preds[_mask])
            _src = "nn" if int(_leaf) in _leaf_nns else "tree"
            print(f"  leaf {_leaf} val roc auc {_leaf_val_auc:.4f} ({_src})")
        return _global_auc, _tree_val_auc, _leaf_nns, _leaf_holdout_aucs, _tree

    leaf_nn_global_auc, leaf_tree_val_auc, leaf_nns, leaf_holdout_aucs, leaf_tree = train_leaf_networks()
    return


@app.cell
def _(np, roc_auc_score):
    def _rows(frame, mask):
        return frame.iloc[mask] if hasattr(frame, "iloc") else frame[mask]

    def train_and_evaluate(
        model_class, X_train, y_train, X_val, y_val, params,
        train_clusters=None, val_clusters=None,
    ):
        """Fit `model_class` and print validation ROC AUC. With cluster labels, fit one model
        per cluster and print per-cluster and overall scores.

        Returns `(score, models)` where `models` is a fitted estimator or a dict cluster id -> estimator.
        """
        name = getattr(model_class, "__name__", str(model_class))
        if train_clusters is None:
            model = model_class(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_pred)
            print(f"{name} ROC AUC Score: {score:.4f}")
            return score, model

        train_clusters = np.asarray(train_clusters)
        val_clusters = np.asarray(val_clusters)
        y_pred = np.full(len(y_val), np.nan, dtype=np.float64)
        models = {}
        print(f"{name}, one model per cluster")
        for cid in np.unique(np.concatenate([train_clusters, val_clusters])):
            tr, va = train_clusters == cid, val_clusters == cid
            if not tr.any():
                print(f"  cluster {cid}: no training rows, skipped")
                continue
            model = model_class(**params)
            model.fit(_rows(X_train, tr), y_train[tr])
            models[int(cid)] = model
            if not va.any():
                print(f"  cluster {cid}: train {int(tr.sum())}, no validation rows")
                continue
            y_pred[va] = model.predict_proba(_rows(X_val, va))[:, 1]
            if np.unique(y_val[va]).size < 2:
                print(f"  cluster {cid}: train {int(tr.sum())}, val {int(va.sum())}, ROC AUC n/a (one class)")
            else:
                score = roc_auc_score(y_val[va], y_pred[va])
                print(f"  cluster {cid}: train {int(tr.sum())}, val {int(va.sum())}, ROC AUC {score:.4f}")

        mask = np.isfinite(y_pred)
        overall = roc_auc_score(y_val[mask], y_pred[mask])
        print(f"{name} overall ROC AUC Score: {overall:.4f}")
        return overall, models

    return (train_and_evaluate,)


@app.cell
def _(
    CatBoostClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    XGBClassifier,
    lgb,
    roc_auc_score,
):
    import optuna

    def optimize_model_hyperparameters(model_name, X_train, y_train, X_val, y_val, n_trials=50):
        def objective(trial):
            if model_name in ["random forest", "extra trees"]:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 600, 1400, step=100),
                    "max_depth": trial.suggest_int("max_depth", 3, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
                    "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 1.0]),
                    "random_state": 0,
                    "n_jobs": -1
                }
                model = RandomForestClassifier(**params) if model_name == "random forest" else ExtraTreesClassifier(**params)

            elif model_name in ["XGBoost", "LightGBM"]:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "random_state": 0
                }
                if model_name == "XGBoost":
                    model = XGBClassifier(**params, tree_method="hist", n_jobs=-1)
                elif model_name == "LightGBM":
                    model = lgb.LGBMClassifier(**params, verbose=-1)
                else:
                    model = HistGradientBoostingClassifier(**params, early_stopping=True)

            elif model_name == "CatBoost":
                params = {
                    "iterations": trial.suggest_int("iterations", 600, 1400, step=100),
                    "depth": trial.suggest_int("depth", 2, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
                    "random_seed": 0,
                    "verbose": 0,
                    "allow_writing_files": False,
                    "auto_class_weights": trial.suggest_categorical("auto_class_weights", ["Balanced", "SqrtBalanced", None])
                }
                model = CatBoostClassifier(**params)

            else:
                raise ValueError(f"Unsupported model name: {model_name}")

            # Fit and evaluate
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, y_pred)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        print(f"Best ROC AUC for {model_name}: {study.best_value:.4f}")
        return study.best_params

    return (optimize_model_hyperparameters,)


@app.cell
def _(
    CatBoostClassifier,
    fit_idx,
    make_dataset,
    optimize_model_hyperparameters,
    train_and_evaluate,
    training_set_clusters,
    validation_set_clusters,
):
    _TUNE = False
    _PER_CLUSTER = True

    _ds = make_dataset("nn embeddings + output", fit_idx)

    _fit_X = _ds.X_train.fillna(-1)
    _fit_y = _ds.y_train
    _val_X = _ds.X_val.fillna(-1)
    _val_y = _ds.y_val
    _cluster_kw = (
        dict(train_clusters=training_set_clusters, val_clusters=validation_set_clusters)
        if _PER_CLUSTER else {}
    )

    tree_nn_out_models = None
    tree_nn_out_X_test = _ds.X_test
    if _TUNE:
        for _m in ["random forest", "extra trees", "XGBoost", "LightGBM", "CatBoost"]:
            optimize_model_hyperparameters(_m, _fit_X, _fit_y, _val_X, _val_y, n_trials=10)
    else:
        # result after running:
        # best: catboost
        _best_param = {
          "iterations": 1000,
          "depth": 5,
          "learning_rate": 0.10098919313017121,
          "l2_leaf_reg": 0.18767037119892388
        }
        _, tree_nn_out_models = train_and_evaluate(
            CatBoostClassifier, _fit_X, _fit_y, _val_X, _val_y, _best_param, **_cluster_kw
        )
    return tree_nn_out_X_test, tree_nn_out_models


@app.cell
def _(
    CAT_COLS,
    CONT_COLS,
    clusterer,
    clustering_features,
    np,
    test_prep,
    tree_nn_out_X_test,
    tree_nn_out_models,
    write_tree_submission,
):
    if tree_nn_out_models is not None:
        _test_clusters = clusterer.predict(clustering_features(
            test_prep[CONT_COLS].to_numpy(dtype=np.float32),
            test_prep[CAT_COLS].to_numpy(dtype=np.float32),
        )) if isinstance(tree_nn_out_models, dict) else None
        write_tree_submission(
            tree_nn_out_models, tree_nn_out_X_test, "tree-catboost-nn-out",
            test_clusters=_test_clusters,
        )
    return


@app.cell
def _(
    CatBoostClassifier,
    clustered_nn,
    fit_idx,
    make_dataset,
    optimize_model_hyperparameters,
    train_and_evaluate,
    training_set_clusters,
    validation_set_clusters,
):
    _TUNE = False
    _PER_CLUSTER = True

    _ds = make_dataset("nn embeddings + output", fit_idx, clustered_nn=clustered_nn)

    _fit_X = _ds.X_train.fillna(-1)
    _fit_y = _ds.y_train
    _val_X = _ds.X_val.fillna(-1)
    _val_y = _ds.y_val
    _cluster_kw = (
        dict(train_clusters=training_set_clusters, val_clusters=validation_set_clusters)
        if _PER_CLUSTER else {}
    )

    tree_nn_out_models = None
    tree_nn_out_X_test = _ds.X_test
    if _TUNE:
        for _m in ["random forest", "extra trees", "XGBoost", "LightGBM", "CatBoost"]:
            optimize_model_hyperparameters(_m, _fit_X, _fit_y, _val_X, _val_y, n_trials=10)
    else:
        # result after running:
        # best: catboost
        _best_param = {
          "iterations": 1000,
          "depth": 5,
          "learning_rate": 0.10098919313017121,
          "l2_leaf_reg": 0.18767037119892388
        }
        _, tree_nn_out_models = train_and_evaluate(
            CatBoostClassifier, _fit_X, _fit_y, _val_X, _val_y, _best_param, **_cluster_kw
        )
    return tree_nn_out_X_test, tree_nn_out_models


@app.cell
def _(
    CatBoostClassifier,
    fit_idx,
    make_dataset,
    optimize_model_hyperparameters,
    train_and_evaluate,
    training_set_clusters,
    validation_set_clusters,
):
    _TUNE = False
    _PER_CLUSTER = True

    _ds = make_dataset("nn embeddings + output", fit_idx)

    _fit_X = _ds.X_train.fillna(-1)
    _fit_y = _ds.y_train
    _val_X = _ds.X_val.fillna(-1)
    _val_y = _ds.y_val
    _cluster_kw = (
        dict(train_clusters=training_set_clusters, val_clusters=validation_set_clusters)
        if _PER_CLUSTER else {}
    )

    _tree_nn_out_models = None
    _tree_nn_out_X_test = _ds.X_test
    if _TUNE:
        for _m in ["random forest", "extra trees", "XGBoost", "LightGBM", "CatBoost"]:
            optimize_model_hyperparameters(_m, _fit_X, _fit_y, _val_X, _val_y, n_trials=10)
    else:
        # result after running:
        # best: catboost
        _best_param = {
          "iterations": 1000,
          "depth": 5,
          "learning_rate": 0.10098919313017121,
          "l2_leaf_reg": 0.18767037119892388
        }
        _, _tree_nn_out_models = train_and_evaluate(
            CatBoostClassifier, _fit_X, _fit_y, _val_X, _val_y, _best_param, **_cluster_kw
        )
    return


@app.cell
def _(
    CatBoostClassifier,
    XGBClassifier,
    fit_idx,
    lgb,
    make_dataset,
    optimize_model_hyperparameters,
    train_and_evaluate,
    training_set_clusters,
    validation_set_clusters,
):
    _TUNE = False
    _PER_CLUSTER = True

    _ds = make_dataset("nn embeddings", fit_idx)

    _fit_X = _ds.X_train.fillna(-1)
    _fit_y = _ds.y_train
    _val_X = _ds.X_val.fillna(-1)
    _val_y = _ds.y_val
    _cluster_kw = (
        dict(train_clusters=training_set_clusters, val_clusters=validation_set_clusters)
        if _PER_CLUSTER else {}
    )

    tree_nn_emb_fitted = []
    tree_nn_emb_X_test = _ds.X_test
    if _TUNE:
        for _m in ["random forest", "extra trees", "XGBoost", "LightGBM", "CatBoost"]:
            optimize_model_hyperparameters(_m, _fit_X, _fit_y, _val_X, _val_y, n_trials=10)
    else:
        # result after running: best are XGBoost, LightGBM and CatBoost with parameters:
        _best_params = {
            XGBClassifier: {'n_estimators': 700, 'max_depth': 6, 'learning_rate': 0.026714616722410156, 'subsample': 0.7754555491994554, 'colsample_bytree': 0.9720160067875027},
            lgb.LGBMClassifier: {'n_estimators': 1000, 'max_depth': 7, 'learning_rate': 0.013601724502393548, 'subsample': 0.9517460066112589, 'colsample_bytree': 0.6976125801099711},
            CatBoostClassifier: {'iterations': 700, 'depth': 8, 'learning_rate': 0.02705719756516648, 'l2_leaf_reg': 0.027000453718482205} 
        }
        for _m in [XGBClassifier, lgb.LGBMClassifier, CatBoostClassifier]:
            _, _fitted = train_and_evaluate(_m, _fit_X, _fit_y, _val_X, _val_y, _best_params[_m], **_cluster_kw)
            tree_nn_emb_fitted.append((_m.__name__, _fitted))
    return tree_nn_emb_X_test, tree_nn_emb_fitted


@app.cell
def _(
    CAT_COLS,
    CONT_COLS,
    clusterer,
    clustering_features,
    np,
    test_prep,
    tree_nn_emb_X_test,
    tree_nn_emb_fitted,
    write_tree_submission,
):
    _test_clusters = clusterer.predict(clustering_features(
        test_prep[CONT_COLS].to_numpy(dtype=np.float32),
        test_prep[CAT_COLS].to_numpy(dtype=np.float32),
    ))
    for _name, _models in tree_nn_emb_fitted:
        write_tree_submission(
            _models, tree_nn_emb_X_test, f"tree-{_name}-nn-emb",
            test_clusters=_test_clusters if isinstance(_models, dict) else None,
        )
    return


@app.cell
def _(
    CatBoostClassifier,
    TREE_FEATURES,
    XGBClassifier,
    fit_idx,
    labels,
    optimize_model_hyperparameters,
    test_prep,
    train_and_evaluate,
    train_prep,
    training_set_clusters,
    val_idx,
    validation_set_clusters,
):
    _TUNE = False
    _PER_CLUSTER = True

    _fit_X = train_prep[TREE_FEATURES].iloc[fit_idx].fillna(-1)
    _val_X = train_prep[TREE_FEATURES].iloc[val_idx].fillna(-1)
    _fit_y, _val_y = labels[fit_idx], labels[val_idx]
    _cluster_kw = (
        dict(train_clusters=training_set_clusters, val_clusters=validation_set_clusters)
        if _PER_CLUSTER else {}
    )

    tree_default_fitted = []
    tree_default_X_test = test_prep[TREE_FEATURES]
    if _TUNE:
        for _m in ["random forest", "extra trees", "XGBoost", "LightGBM", "CatBoost"]:
            optimize_model_hyperparameters(_m, _fit_X, _fit_y, _val_X, _val_y, n_trials=10)
    else:
        _best_params = {
            XGBClassifier: {'n_estimators': 700, 'max_depth': 9, 'learning_rate': 0.01135962971122079, 'subsample': 0.8342352304254633, 'colsample_bytree': 0.8170993435755725},
            CatBoostClassifier: {'iterations': 500, 'depth': 8, 'learning_rate': 0.05595418680834222, 'l2_leaf_reg': 5.998938298775239}
        }

        for _m in [XGBClassifier, CatBoostClassifier]:
            _, _fitted = train_and_evaluate(_m, _fit_X, _fit_y, _val_X, _val_y, _best_params[_m], **_cluster_kw)
            tree_default_fitted.append((_m.__name__, _fitted))
    return tree_default_X_test, tree_default_fitted


@app.cell
def _(
    CAT_COLS,
    CONT_COLS,
    clusterer,
    clustering_features,
    np,
    test_prep,
    tree_default_X_test,
    tree_default_fitted,
    write_tree_submission,
):
    _test_clusters = clusterer.predict(clustering_features(
        test_prep[CONT_COLS].to_numpy(dtype=np.float32),
        test_prep[CAT_COLS].to_numpy(dtype=np.float32),
    ))
    for _name, _models in tree_default_fitted:
        write_tree_submission(
            _models, tree_default_X_test, f"tree-{_name}-default-codes",
            test_clusters=_test_clusters if isinstance(_models, dict) else None,
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    On cherche le meilleur encodage déterministe en faisant tourner CatBoost sur chaque. On choisit CV5MT. Bon, ça se joue à peu.
    """)
    return


@app.cell
def _(TARGET_ENCODERS, fit_idx, make_dataset, optimize_model_hyperparameters):
    _TUNE = False

    if _TUNE:
        for _k in TARGET_ENCODERS:
            print(_k)
            _ds = make_dataset(_k, fit_idx)

            _fit_X = _ds.X_train.fillna(-1)
            _fit_y = _ds.y_train
            _val_X = _ds.X_val.fillna(-1)
            _val_y = _ds.y_val

            for _m in ["CatBoost"]:
                optimize_model_hyperparameters(_m, _fit_X, _fit_y, _val_X, _val_y, n_trials=10)
    else:
        _ds = make_dataset("target CV5MT", fit_idx)

        _fit_X = _ds.X_train.fillna(-1)
        _fit_y = _ds.y_train
        _val_X = _ds.X_val.fillna(-1)
        _val_y = _ds.y_val

        for _m in ["XGBoost", "LightGBM", "CatBoost"]:
            optimize_model_hyperparameters(_m, _fit_X, _fit_y, _val_X, _val_y, n_trials=10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # obliquetree

    https://www.reddit.com/r/MachineLearning/comments/1hxa6u6/r_obliquetree_advanced_decision_tree/
    """)
    return


@app.cell
def _():
    from obliquetree import Classifier, Regressor
    from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
    from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
    from sklearn.utils.multiclass import unique_labels
    from obliquetree import Classifier as ObliqueTreeClassifier
    from obliquetree import Regressor as ObliqueTreeRegressor

    return (
        BaseEstimator,
        ClassifierMixin,
        ObliqueTreeClassifier,
        ObliqueTreeRegressor,
        RegressorMixin,
        unique_labels,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    La classe suivante contient un oblique tree et les méthodes de l'interface de SKLearn lui permettant d'être utilié avec des modèles augmentés type adaboost.
    """)
    return


@app.cell
def _(
    BaseEstimator,
    CatBoostClassifier,
    ClassifierMixin,
    ObliqueTreeClassifier,
    ObliqueTreeRegressor,
    RegressorMixin,
    np,
    unique_labels,
):
    class SklearnObliqueTreeClassifier(BaseEstimator, ClassifierMixin):
        def __init__(
            self, 
            use_oblique=True, 
            max_depth=3, 
            n_pair=2, 
            random_state=None, 
            categories=None,
            # Add any other specific obliquetree parameters you need here
        ):
            # Scikit-learn requires all parameters to be explicitly defined in __init__
            # and stored as attributes with the exact same name.
            self.use_oblique = use_oblique
            self.max_depth = max_depth
            self.n_pair = n_pair
            self.random_state = random_state
            self.categories = categories

        def fit(self, X, y, sample_weight=None):
            # Validate data. force_all_finite='allow-nan' ensures we don't break 
            # obliquetree's native missing value handling.
            #X, y = check_X_y(X, y)

            # Scikit-learn requires these attributes for classifiers
            self.classes_ = unique_labels(y)
            self.n_classes_ = len(self.classes_)

            # Initialize the underlying obliquetree model
            self.estimator_ = ObliqueTreeClassifier(
                use_oblique=self.use_oblique,
                max_depth=self.max_depth,
                n_pair=self.n_pair,
                random_state=self.random_state,
                categories=self.categories
            )

            # AdaBoost relies heavily on sample weights during boosting iterations.
            if sample_weight is not None:
                self.estimator_.fit(X, y, sample_weight=sample_weight)
            else:
                self.estimator_.fit(X, y)

            return self

        def predict(self, X):
            return self.estimator_.predict(X)

        def predict_proba(self, X):        
            # Standard AdaBoost algorithm (SAMME.R) requires probability estimates.
            if hasattr(self.estimator_, "predict_proba"):
                return self.estimator_.predict_proba(X)

            # Fallback if obliquetree only supports hard predictions
            preds = self.predict(X)
            proba = np.zeros((len(X), self.n_classes_))
            for i, p in enumerate(preds):
                class_idx = np.where(self.classes_ == p)[0]
                if len(class_idx) > 0:
                    proba[i, class_idx[0]] = 1.0
            return proba


    class SklearnObliqueTreeRegressor(BaseEstimator, RegressorMixin):
        def __init__(
            self, 
            use_oblique=True, 
            max_depth=3, 
            n_pair=2, 
            random_state=None, 
            categories=None,
        ):
            self.use_oblique = use_oblique
            self.max_depth = max_depth
            self.n_pair = n_pair
            self.random_state = random_state
            self.categories = categories

        def fit(self, X, y, sample_weight=None):
            self.estimator_ = ObliqueTreeRegressor(
                use_oblique=self.use_oblique,
                max_depth=self.max_depth,
                n_pair=self.n_pair,
                random_state=self.random_state,
                categories=self.categories
            )

            if sample_weight is not None:
                self.estimator_.fit(X, y, sample_weight=sample_weight)
            else:
                self.estimator_.fit(X, y)

            return self

        def predict(self, X):
            return self.estimator_.predict(X)


    class SklearnCatBoostClassifier(BaseEstimator, ClassifierMixin):
        """CatBoost with sklearn clone semantics. `cat_features` is stored as given and
        only copied into a fresh CatBoostClassifier at `fit` time, so AdaBoost can clone it.
        """

        def __init__(
            self,
            iterations=1000,
            learning_rate=0.03,
            depth=6,
            eval_metric="AUC",
            verbose=0,
            random_seed=0,
            allow_writing_files=False,
            cat_features=None,
            auto_class_weights=None,
        ):
            self.iterations = iterations
            self.learning_rate = learning_rate
            self.depth = depth
            self.eval_metric = eval_metric
            self.verbose = verbose
            self.random_seed = random_seed
            self.allow_writing_files = allow_writing_files
            self.cat_features = cat_features
            self.auto_class_weights = auto_class_weights

        def fit(self, X, y, sample_weight=None):
            self.classes_ = unique_labels(y)
            self.n_classes_ = len(self.classes_)
            cat_features = None if self.cat_features is None else list(self.cat_features)
            self.estimator_ = CatBoostClassifier(
                iterations=self.iterations,
                learning_rate=self.learning_rate,
                depth=self.depth,
                eval_metric=self.eval_metric,
                verbose=self.verbose,
                random_seed=self.random_seed,
                allow_writing_files=self.allow_writing_files,
                cat_features=cat_features,
                auto_class_weights=self.auto_class_weights,
            )
            if sample_weight is not None:
                self.estimator_.fit(X, y, sample_weight=sample_weight)
            else:
                self.estimator_.fit(X, y)
            return self

        def predict(self, X):
            return self.estimator_.predict(X)

        def predict_proba(self, X):
            return self.estimator_.predict_proba(X)

    return (
        SklearnCatBoostClassifier,
        SklearnObliqueTreeClassifier,
        SklearnObliqueTreeRegressor,
    )


@app.cell
def _(
    SklearnObliqueTreeClassifier,
    TREE_FEATURES,
    fit_idx,
    labels,
    roc_auc_score,
    sklearn,
    train_prep,
    val_idx,
):
    _d = 6

    _fit_X = train_prep[TREE_FEATURES].iloc[fit_idx].fillna(-1)
    _val_X = train_prep[TREE_FEATURES].iloc[val_idx].fillna(-1)
    _fit_y, _val_y = labels[fit_idx], labels[val_idx]

    # On peut donner au modèle la liste des colonnes catégorielles. Ça améliore à peine les performances et rend l'entraînement très très très très très très long.
    _cols = list(_fit_X.columns)
    _categories = None # [_cols.index(_f) for _f in TREE_LOW_CARD_CATEGORICAL]

    _rng = range(1, 10) if _d == "sweep" else [_d]
    for _d in _rng:
        print(_d)
        adaboosted_class = sklearn.ensemble.AdaBoostClassifier(SklearnObliqueTreeClassifier(
                use_oblique=True,
                max_depth=_d,
                n_pair=2,
                random_state=42,
                categories=_categories,
            ), n_estimators=60)
        adaboosted_class.fit(_fit_X, _fit_y)
        _preds = adaboosted_class.predict_proba(_val_X)
        print(roc_auc_score(_val_y, _preds[:, 1]))
    return (adaboosted_class,)


@app.cell
def _(TREE_FEATURES, adaboosted_class, test_prep, write_tree_submission):
    write_tree_submission(
        adaboosted_class, test_prep[TREE_FEATURES], "tree-adaboost-oblique-default-codes"
    )
    return


@app.cell
def _(
    SklearnObliqueTreeClassifier,
    fit_idx,
    make_dataset,
    roc_auc_score,
    sklearn,
):
    _TUNE = False

    _ds = make_dataset("target CV5MT", fit_idx)

    _fit_X = _ds.X_train.fillna(-1)
    _fit_y = _ds.y_train
    _val_X = _ds.X_val.fillna(-1)
    _val_y = _ds.y_val

    _cols = list(_fit_X.columns)

    _rng = range(1, 10) if _TUNE else [8]

    for _d in range(1, 10):
        adaboosted_class_mt = sklearn.ensemble.AdaBoostClassifier(SklearnObliqueTreeClassifier(  # Replace "Classifier" with "Regressor" if performing regression
                use_oblique=True,       # Enable oblique splits
                max_depth=_d,            # Set the maximum depth of the tree
                n_pair=2,               # Number of feature pairs for optimization
                random_state=42,        # Set a random state for reproducibility
                categories=None, # Specify which features are categorical
            ), n_estimators=50)
        adaboosted_class_mt.fit(_fit_X, _fit_y)
        _preds = adaboosted_class_mt.predict_proba(_val_X)
        print(_d, roc_auc_score(_val_y, _preds[:, 1]))
    adaboosted_mt_X_test = _ds.X_test
    return adaboosted_class_mt, adaboosted_mt_X_test


@app.cell
def _(adaboosted_class_mt, adaboosted_mt_X_test, write_tree_submission):
    write_tree_submission(adaboosted_class_mt, adaboosted_mt_X_test, "tree-adaboost-oblique-cv5mt")
    return


@app.cell
def _(
    SklearnCatBoostClassifier,
    TREE_FEATURES,
    TREE_LOW_CARD_CATEGORICAL,
    fit_idx,
    labels,
    roc_auc_score,
    sklearn,
    train_prep,
    val_idx,
):
    _fit_X = train_prep[TREE_FEATURES].iloc[fit_idx].fillna(-1)
    _val_X = train_prep[TREE_FEATURES].iloc[val_idx].fillna(-1)
    _fit_y, _val_y = labels[fit_idx], labels[val_idx]

    _cols = list(_fit_X.columns)
    _categories = [_cols.index(_f) for _f in TREE_LOW_CARD_CATEGORICAL]

    adaboosted_class_cat = sklearn.ensemble.AdaBoostClassifier(
        SklearnCatBoostClassifier(
            iterations=1000,
            learning_rate=0.03,
            depth=6,
            eval_metric="AUC",
            verbose=0,
            random_seed=0,
            allow_writing_files=False,
            cat_features=_categories,
        ),
        n_estimators=50,
    )
    adaboosted_class_cat.fit(_fit_X, _fit_y)
    _preds = adaboosted_class_cat.predict_proba(_val_X)
    roc_auc_score(_val_y, _preds[:, 1])
    return (adaboosted_class_cat,)


@app.cell
def _(TREE_FEATURES, adaboosted_class_cat, test_prep, write_tree_submission):
    write_tree_submission(
        adaboosted_class_cat, test_prep[TREE_FEATURES], "tree-adaboost-catboost-default-codes"
    )
    return


@app.cell
def _(clone, np):
    class BoostedClassification:
        '''
        Classe pour Gradient Boosting adapté à la classification binaire
        Supporte les classifieurs et les régresseurs comme modèles de base.
        '''
        def __init__(self, base_model, n_estimators=100, learning_rate=0.1):
            self.base_model = base_model
            self.n_estimators = n_estimators
            self.learning_rate = learning_rate
            self.restrees = []

            # On détermine la nature du modèle en cherchant l'attribut 'predict_proba'
            self.is_classifier = hasattr(self.base_model, 'predict_proba')

        def _get_preds(self, model, X):
            if self.is_classifier:
                try:
                    return model.predict_proba(X)[:, 1]
                except IndexError:
                    return model.predict_proba(X)[:, 0]
            else:
                return model.predict(X)

        def predict_proba(self, X):
            '''
            Args :
            - X : vecteur(s) de covariables pour le(s)quel(s) on souhaite avoir la prédiction
            '''
            pred = self._get_preds(self.base_model, X)

            for restree in self.restrees:
                pred += self.learning_rate * self._get_preds(restree, X)

            # On s'assure que les probabilités finies restent bien strictement entre 0 et 1
            pred = np.clip(pred, 0, 1)
            return pred

        def predict(self, X):
            '''
            Retourne la classe prédite (0 ou 1) avec un seuil de 0.5
            '''
            probas = self.predict_proba(X)
            return (probas >= 0.5).astype(int)

        def fit(self, X, Y, X_val=None, Y_val=None):
            '''
            Args :
            - X : vecteurs de covariables pour l'entraînement
            - Y : targets pour l'entraînement (0 ou 1)
            - X_val : vecteurs de covariables pour la validation
            - Y_val : targets pour la validation
            '''
            self.restrees = []
            restrees = [clone(self.base_model) for _ in range(self.n_estimators)]

            # Entraînement du modèle de base
            self.base_model.fit(X, Y)

            # Récupération des prédictions initiales
            preds = self._get_preds(self.base_model, X)

            for i in range(self.n_estimators):
                # Gradient négatif de la perte Cross-Entropy
                res = Y - preds
                preds = np.clip(preds, 0, 1)
                #res = (Y * (1-preds) + (1-Y) * preds) / (preds * (1-preds))
                res_tree = restrees[i]

                if self.is_classifier:
                    # Un classifieur a besoin de cibles discrètes (on utilise le signe du gradient)
                    res_tree.fit(X, (res > 0).astype(int))
                else:
                    # Un régresseur (ex: DecisionTreeRegressor) peut directement fitter le gradient continu
                    res_tree.fit(X, res)

                # Mise à jour des prédictions
                _preds = self._get_preds(res_tree, X)
                preds += self.learning_rate * _preds
                preds = np.clip(preds, 0, 1)

                self.restrees.append(res_tree)

    return (BoostedClassification,)


@app.cell
def _(
    BoostedClassification,
    DecisionTreeClassifier,
    SklearnObliqueTreeRegressor,
    TREE_FEATURES,
    TREE_LOW_CARD_CATEGORICAL,
    fit_idx,
    labels,
    roc_auc_score,
    train_prep,
    val_idx,
):
    _fit_X = train_prep[TREE_FEATURES].iloc[fit_idx].fillna(-1)
    _val_X = train_prep[TREE_FEATURES].iloc[val_idx].fillna(-1)
    _fit_y, _val_y = labels[fit_idx], labels[val_idx]

    _cols = list(_fit_X.columns)
    _categories = [_cols.index(_f) for _f in TREE_LOW_CARD_CATEGORICAL]

    _base_model = SklearnObliqueTreeRegressor( #SklearnObliqueTreeClassifier(
        use_oblique=True,
        max_depth=2,
        n_pair=2,
        random_state=42,
        categories=_categories,
    )

    _base_model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=0)
    xgb_clf_oblique = BoostedClassification(_base_model, n_estimators=20)
    xgb_clf_oblique.fit(_fit_X, _fit_y)

    _preds = xgb_clf_oblique.predict_proba(_val_X)
    roc_auc_score(_val_y, _preds)
    return (xgb_clf_oblique,)


@app.cell
def _(TREE_FEATURES, test_prep, write_tree_submission, xgb_clf_oblique):
    write_tree_submission(
        xgb_clf_oblique, test_prep[TREE_FEATURES], "tree-boosted-oblique-default-codes"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # SVM

    sklearn `SVC` RBF avec `class_weight="balanced"` sur les différentes représentations
    (`make_dataset`). La seule implémentation custom est `DualSVM`, qui apprend **en même
    temps** les coefficients duals $\alpha$ et la largeur de noyau $\gamma$.
    """)
    return


@app.cell
def _(device, np, roc_auc_score, sklearn, torch):
    @torch.no_grad()
    def project_alpha_dykstra(
        alpha_data,
        idx,
        batch_y,
        weights,
        C,
        tol=1e-8,
        max_iter=20,
    ):
        x = alpha_data[idx].clone()
        lo = torch.zeros_like(x)
        hi = C * weights
        p = torch.zeros_like(x)
        q = torch.zeros_like(x)
        y_norm_sq = torch.dot(batch_y, batch_y)

        for k in range(max_iter):
            x_old = x.clone()
            z = x + p
            if y_norm_sq > 1e-12:
                y = z - (torch.dot(z, batch_y) / y_norm_sq) * batch_y
            else:
                y = z
            p = z - y
            z = y + q
            x = torch.minimum(torch.maximum(z, lo), hi)
            q = z - x
            diff = torch.norm(x - x_old)
            scale = 1.0 + torch.norm(x)
            if y_norm_sq > 1e-12:
                feasibility = torch.abs(torch.dot(x, batch_y)) / torch.sqrt(y_norm_sq)
            else:
                feasibility = torch.tensor(0.0, device=x.device)

            if diff <= tol * scale and feasibility <= tol * scale:
                break
        alpha_new = alpha_data.clone()
        alpha_new[idx] = x
        return alpha_new


    def dataset_to_svm_arrays(ds, n_fit=None, fill=-1.0):
        """Standardise a `Dataset` on its training rows and map labels to ±1."""
        X_train = ds.X_train.fillna(fill).to_numpy(dtype=np.float64)
        X_val = ds.X_val.fillna(fill).to_numpy(dtype=np.float64)
        X_test = ds.X_test.fillna(fill).to_numpy(dtype=np.float64)
        y_train = np.asarray(ds.y_train, dtype=np.float64).ravel()
        y_val = np.asarray(ds.y_val, dtype=np.float64).ravel()
        if n_fit is not None:
            X_train, y_train = X_train[:n_fit], y_train[:n_fit]
        mean = np.nanmean(X_train, axis=0)
        std = np.nanstd(X_train, axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        X_train = np.nan_to_num((X_train - mean) / std, nan=0.0)
        X_val = np.nan_to_num((X_val - mean) / std, nan=0.0)
        X_test = np.nan_to_num((X_test - mean) / std, nan=0.0)
        y_train = np.where(y_train > 0, 1.0, -1.0)
        y_val = np.where(y_val > 0, 1.0, -1.0)
        return X_train, y_train, X_val, y_val, X_test, mean, std

    def sklearn_rbf_svm(ds, C=24.2, gamma=0.004, n_fit=60_000, probability=False):
        X_train, y_train, X_val, y_val, X_test, _, _ = dataset_to_svm_arrays(ds, n_fit=n_fit)
        clf = sklearn.svm.SVC(
            C=C,
            kernel="rbf",
            gamma=gamma,
            class_weight="balanced",
            probability=probability,
        )
        clf.fit(X_train, y_train)
        scores = clf.decision_function(X_val)
        auc = roc_auc_score(y_val, scores)
        print(f"sklearn RBF SVC C={C} gamma={gamma} n={len(X_train)}: val roc auc {auc:.4f}")
        return auc, clf, X_test

    class DualSVM:
        """Dual kernel SVM. For `kernel='rbf'`, `gamma` is a tensor updated by the same
        optimiser as the dual coefficients, so the kernel width and the SVM objective
        are trained jointly (as in `classification2.py`).
        """

        def __init__(self, C=1.0, kernel="rbf", gamma=0.01, kappa=2.0, delta=0.0, lr=1e-5):
            self.C = C
            self.kernel = kernel
            self.gamma0 = float(gamma)
            self.kappa = float(kappa)
            self.delta = float(delta)
            self.lr = lr
            self.device = device
            self.alpha = None
            self.support_vectors = None
            self.support_vector_labels = None
            self.b = 0.0
            self.gamma = self.gamma0
            self.mlp = None

        def _rbf(self, dist, gamma, kappa, delta):
            return torch.exp(-gamma * dist ** kappa + delta)

        def _kernel_function(self, X1, X2, gamma=None):
            gamma = self.gamma0 if gamma is None else gamma
            if self.kernel == "linear":
                return X1 @ X2.T
            dist = torch.cdist(X1, X2)
            return self._rbf(dist, gamma, self.kappa, self.delta)

        def fit(self, X, y, n_epochs=100, batch_size=0, log_every=10, X_val=None, y_val=None):
            X = torch.as_tensor(np.asarray(X, dtype=np.float32), device=self.device)
            y = torch.as_tensor(np.asarray(y, dtype=np.float32), device=self.device)
            n_samples = X.shape[0]
            alpha = torch.empty(n_samples, device=self.device, requires_grad=True)
            torch.nn.init.normal_(alpha, 0.0, 1.0)
            gamma = torch.tensor(self.gamma0, device=self.device, requires_grad=True)
            opt = torch.optim.AdamW([alpha, gamma], lr=self.lr, weight_decay=1.0)
            pos_weight = n_samples / (2 * (y == 1).sum().clamp(min=1))
            neg_weight = n_samples / (2 * (y == -1).sum().clamp(min=1))
            if batch_size <= 0 or batch_size >= n_samples:
                batch_size = n_samples
            cdist = torch.cdist(X, X) if self.kernel == "rbf" else None
            history = []
            for epoch in range(n_epochs):
                perm = torch.randperm(n_samples, device=self.device)
                total = 0.0
                for start in range(0, n_samples, batch_size):
                    idx = perm[start:start + batch_size]
                    batch_y = y[idx]
                    weights = torch.where(batch_y == 1, pos_weight, neg_weight)
                    if self.kernel == "rbf":
                        K = self._rbf(cdist[idx][:, idx], gamma, self.kappa, self.delta)
                    else:
                        K = X[idx] @ X[idx].T
                    opt.zero_grad()
                    term1 = (alpha[idx] * weights).sum()
                    term2 = 0.5 * ((alpha[idx] * batch_y).unsqueeze(1) * (alpha[idx] * batch_y).unsqueeze(0) * K).sum()
                    loss = -term1 + term2
                    loss.backward()
                    opt.step()
                    #with torch.no_grad():
                    #    alpha_sum = (alpha.data[idx] * batch_y).sum()
                    #    alpha.data[idx] -= (alpha_sum / (batch_y ** 2).sum().clamp(min=1e-8)) * batch_y
                    #    lo = torch.zeros(len(idx), device=self.device)
                    #    hi = self.C * weights
                    #    alpha.data[idx] = torch.clamp(alpha.data[idx], lo, hi)
                    #    gamma.data.clamp_(1e-6, 10.0)
                    alpha.data.copy_(
                        project_alpha_dykstra(
                            alpha.data,
                            idx,
                            batch_y,
                            weights,
                            self.C,
                            tol=1e-8,
                            max_iter=10000
                        )
                    )
                    gamma.data.clamp_(1e-6, 10.0)
                    total += float(loss.detach())
                if (epoch % log_every == 0 or epoch == n_epochs - 1) and X_val is not None:
                    self._store(alpha, X, y, gamma)
                    scores = self.decision_function(X_val)
                    auc = roc_auc_score(np.asarray(y_val).ravel(), scores)
                    history.append(auc)
                    print(
                        f"DualSVM epoch {epoch}/{n_epochs}: loss {total:.4f} "
                        f"gamma {float(gamma.detach()):.5f} val roc auc {auc:.4f}"
                    )
            if cdist is not None:
                del cdist
            self._store(alpha, X, y, gamma)
            return history

        def _store(self, alpha, X, y, gamma):
            with torch.no_grad():
                sv = alpha > 1e-5
                self.alpha = alpha.detach()[sv]
                self.support_vectors = X[sv]
                self.support_vector_labels = y[sv]
                self.gamma = float(gamma.detach())
                if len(self.alpha) == 0:
                    self.b = 0.0
                    return
                K_sv = self._kernel_function(
                    self.support_vectors, self.support_vectors, gamma=gamma.detach(),
                )
                contrib = (self.alpha * self.support_vector_labels) @ K_sv
                self.b = float((self.support_vector_labels - contrib).mean())

        def decision_function(self, X):
            X = torch.as_tensor(np.asarray(X, dtype=np.float32), device=self.device)
            if self.support_vectors is None or len(self.alpha) == 0:
                return np.zeros(len(X), dtype=np.float64)
            K = self._kernel_function(X, self.support_vectors, gamma=self.gamma)
            scores = K @ (self.alpha * self.support_vector_labels) + self.b
            return scores.detach().cpu().numpy()

        def predict(self, X):
            return np.sign(self.decision_function(X))

    def train_dual_svm(ds, C=0.1, gamma=0.01, n_fit=4000, n_epochs=80, log_every=10, **kwargs):
        X_train, y_train, X_val, y_val, X_test, _, _ = dataset_to_svm_arrays(ds, n_fit=n_fit)
        model = DualSVM(C=C, kernel="rbf", gamma=gamma, **kwargs)
        history = model.fit(X_train, y_train, n_epochs=n_epochs, log_every=log_every, X_val=X_val, y_val=y_val)
        scores = model.decision_function(X_val)
        auc = roc_auc_score(y_val, scores)
        print(
            f"DualSVM C={C} learned gamma={model.gamma:.5f} n={len(X_train)}: "
            f"val roc auc {auc:.4f}"
        )
        return auc, model, X_test, history

    return sklearn_rbf_svm, train_dual_svm


@app.cell
def _(fit_idx, make_dataset, sklearn_rbf_svm):
    SVM_KINDS = [
        "default codes",
        "nn embeddings",
        "nn embeddings + output",
        "nn attention embeddings",
        "nn attention embeddings + output",
        "target CV5MT",
    ]
    sklearn_svm_results = {}
    for _kind in SVM_KINDS:
        _ds = make_dataset(_kind, fit_idx)
        _n = 60_000 if _kind == "default codes" else 20_000
        _auc, _clf, _ = sklearn_rbf_svm(
            _ds, C=24.2, gamma=0.004, n_fit=_n,
        )
        sklearn_svm_results[_kind] = dict(val_roc_auc=_auc, model=_clf)
    sklearn_svm_results
    return


@app.cell
def _(fit_idx, make_dataset, train_dual_svm):
    _ds = make_dataset("default codes", fit_idx)
    dual_svm_auc, dual_svm, _, dual_svm_history = train_dual_svm(
        _ds, C=0.1, gamma=0.01, n_fit=4000, n_epochs=80, log_every=10,
    )
    print("jointly learned kernel gamma", dual_svm.gamma)
    dual_svm_auc
    return


@app.cell
def _(fit_idx, make_dataset, train_dual_svm):
    _ds = make_dataset("nn embeddings", fit_idx)
    dual_svm_emb_auc, dual_svm_emb, _, _ = train_dual_svm(
        _ds, C=0.1, gamma=0.01, n_fit=4000, n_epochs=60, log_every=10,
    )
    dual_svm_emb_auc
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # AutoGluon
    """)
    return


@app.cell
def _(
    CATEGORICAL_FIELDS,
    fit_idx,
    pd,
    roc_auc_score,
    test_prep,
    time,
    train_prep,
    val_idx,
):
    from autogluon.tabular import TabularPredictor

    AUTOGLUON_TIME_LIMIT = 1000  # seconds
    AUTOGLUON_PRESETS = "best_quality"
    AUTOGLUON_PATH = "autogluon_models"

    def autogluon_frame(df):
        out = df.drop(columns=["id"])
        for _c in CATEGORICAL_FIELDS:
            out[_c] = out[_c].astype("category")
        return out

    autogluon_fit_df = autogluon_frame(train_prep.iloc[fit_idx])
    autogluon_val_df = autogluon_frame(train_prep.iloc[val_idx])
    autogluon_test_df = autogluon_frame(test_prep)

    autogluon_predictor = TabularPredictor(
        label="accepted",
        problem_type="binary",
        eval_metric="roc_auc",
        path=AUTOGLUON_PATH,
    ).fit(
        autogluon_fit_df,
        presets=AUTOGLUON_PRESETS,
        time_limit=AUTOGLUON_TIME_LIMIT,
    )

    _proba = autogluon_predictor.predict_proba(autogluon_val_df, as_multiclass=False)
    print(f"AutoGluon validation roc auc: {roc_auc_score(autogluon_val_df['accepted'], _proba):.4f}")
    autogluon_predictor.leaderboard(autogluon_val_df, silent=True)

    _y = autogluon_predictor.predict_proba(autogluon_test_df, as_multiclass=False).to_numpy()
    _submission = pd.DataFrame({"id": test_prep.id.to_numpy(), "accepted": _y}).set_index("id")
    _submission.to_csv(f"submission_{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}.csv", float_format="%.4f")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # CatBoost: comparaison encodages et imputation

    Meilleure config CatBoost déjà retenue pour chaque représentation, fittée une fois sur
    les codes sans imputation XGBoost et une fois avec.
    """)
    return


@app.cell
def _(
    CATEGORICAL_FIELDS,
    CAT_COLS,
    CONT_COLS,
    CatBoostClassifier,
    TREE_FEATURES,
    TargetEncoder,
    embedded_frame,
    fit_idx,
    impute_missing,
    labels,
    np,
    pd,
    plt,
    roc_auc_score,
    test_coded,
    train_coded,
    trained_nn,
    val_idx,
):
    _CATBOOST_BEST = {
        "nn embeddings": {
            "iterations": 700, "depth": 8,
            "learning_rate": 0.02705719756516648, "l2_leaf_reg": 0.027000453718482205,
        },
        "nn embeddings + output": {
            "iterations": 1000, "depth": 8,
            "learning_rate": 0.10098919313017121, "l2_leaf_reg": 0.18767037119892388,
        },
        "default encoding": {
            "iterations": 500, "depth": 8,
            "learning_rate": 0.05595418680834222, "l2_leaf_reg": 5.998938298775239,
        },
        "CV5MT": {
            "iterations": 500, "depth": 8,
            "learning_rate": 0.05595418680834222, "l2_leaf_reg": 5.998938298775239,
        },
    }
    _EXTRA = dict(verbose=0, allow_writing_files=False, random_seed=0, eval_metric="AUC")

    def _xy_default(frame):
        feat = frame[TREE_FEATURES]
        return feat.iloc[fit_idx], feat.iloc[val_idx]

    def _xy_nn(frame, with_output):
        feat = embedded_frame(frame, with_output=with_output, network=trained_nn)
        return feat.iloc[fit_idx], feat.iloc[val_idx]

    def _xy_cv5mt(frame):
        encoder = TargetEncoder("CVMT", CATEGORICAL_FIELDS, folds=5)
        codes, cont = frame[CAT_COLS], frame[CONT_COLS]
        X_train = pd.concat(
            [cont.iloc[fit_idx].reset_index(drop=True), encoder.fit_transform(codes.iloc[fit_idx], labels[fit_idx])],
            axis=1,
        )
        X_val = pd.concat(
            [cont.iloc[val_idx].reset_index(drop=True), encoder.transform(codes.iloc[val_idx])],
            axis=1,
        )
        return X_train, X_val

    _builders = [
        ("nn embeddings", lambda fr: _xy_nn(fr, False)),
        ("nn embeddings + output", lambda fr: _xy_nn(fr, True)),
        ("default encoding", _xy_default),
        ("CV5MT", _xy_cv5mt),
    ]

    _y_fit, _y_val = labels[fit_idx], labels[val_idx]
    catboost_auc_rows = []
    for _learned in (False, True):
        _train_f, _test_f = train_coded.copy(), test_coded.copy()
        _missing_label = "learned missing values" if _learned else "no learned missing values"
        if _learned:
            _train_f, _test_f = impute_missing(_train_f, _test_f, fit_idx)
        for _name, _build in _builders:
            _X_fit, _X_val = _build(_train_f)
            _params = {**_CATBOOST_BEST[_name], **_EXTRA}
            _model = CatBoostClassifier(**_params)
            _model.fit(_X_fit.fillna(-1), _y_fit)
            _auc = roc_auc_score(_y_val, _model.predict_proba(_X_val.fillna(-1))[:, 1])
            print(f"CatBoost {_name} / {_missing_label}: {_auc:.4f}")
            catboost_auc_rows.append(
                {"encoding": _name, "missing_values": _missing_label, "val_roc_auc": _auc}
            )

    catboost_auc_df = pd.DataFrame(catboost_auc_rows)
    _encodings = [name for name, _ in _builders]
    _x = np.arange(len(_encodings))
    _w = 0.35
    _raw = catboost_auc_df.set_index(["encoding", "missing_values"])["val_roc_auc"]
    _fig, _ax = plt.subplots(figsize=(10, 5))
    _ax.bar(
        _x - _w / 2,
        [_raw.loc[e, "no learned missing values"] for e in _encodings],
        width=_w,
        label="Pas d'imputation",
    )
    _ax.bar(
        _x + _w / 2,
        [_raw.loc[e, "learned missing values"] for e in _encodings],
        width=_w,
        label="imputation par XGBoost",
    )
    _ax.set_xticks(_x)
    _ax.set_xticklabels(_encodings, rotation=15, ha="right")
    _ax.set_ylabel("val ROC AUC")
    _ymin = float(catboost_auc_df["val_roc_auc"].min()) - 0.01
    _ymax = float(catboost_auc_df["val_roc_auc"].max()) + 0.01
    _ax.set_ylim(_ymin, _ymax)
    _ax.legend()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # SVM: comparaison encodages et imputation

    Meilleure config sklearn RBF (`C=24.2`, \(\gamma=0.004\), `class_weight="balanced"`), on fait varier le type d'encodage et l'imputation des données et on fit sur 20k échantillons pour comparer.
    """)
    return


@app.cell
def _(
    CATEGORICAL_FIELDS,
    CAT_COLS,
    CONT_COLS,
    TREE_FEATURES,
    TargetEncoder,
    embedded_frame,
    fit_idx,
    impute_missing,
    labels,
    np,
    pd,
    plt,
    roc_auc_score,
    sklearn,
    test_coded,
    train_coded,
    trained_nn,
    val_idx,
):
    _SVM_C, _SVM_GAMMA, _SVM_N_FIT = 24.2, 0.004, 20_000

    def _xy_default(frame):
        feat = frame[TREE_FEATURES]
        return feat.iloc[fit_idx], feat.iloc[val_idx]

    def _xy_nn(frame, with_output):
        feat = embedded_frame(frame, with_output=with_output, network=trained_nn)
        return feat.iloc[fit_idx], feat.iloc[val_idx]

    def _xy_cv5mt(frame):
        encoder = TargetEncoder("CVMT", CATEGORICAL_FIELDS, folds=5)
        codes, cont = frame[CAT_COLS], frame[CONT_COLS]
        X_train = pd.concat(
            [cont.iloc[fit_idx].reset_index(drop=True), encoder.fit_transform(codes.iloc[fit_idx], labels[fit_idx])],
            axis=1,
        )
        X_val = pd.concat(
            [cont.iloc[val_idx].reset_index(drop=True), encoder.transform(codes.iloc[val_idx])],
            axis=1,
        )
        return X_train, X_val

    _builders = [
        ("nn embeddings", lambda fr: _xy_nn(fr, False)),
        ("nn embeddings + output", lambda fr: _xy_nn(fr, True)),
        ("default encoding", _xy_default),
        ("CV5MT", _xy_cv5mt),
    ]

    _y_fit_all, _y_val = labels[fit_idx], labels[val_idx]
    svm_auc_rows = []
    for _learned in (False, True):
        _train_f, _test_f = train_coded.copy(), test_coded.copy()
        _missing_label = "learned missing values" if _learned else "no learned missing values"
        if _learned:
            _train_f, _test_f = impute_missing(_train_f, _test_f, fit_idx)
        for _name, _build in _builders:
            _X_fit, _X_val = _build(_train_f)
            _X_fit = _X_fit.fillna(-1).to_numpy(dtype=np.float64)[:_SVM_N_FIT]
            _X_val = _X_val.fillna(-1).to_numpy(dtype=np.float64)
            _y_fit = np.where(_y_fit_all[:_SVM_N_FIT] > 0, 1.0, -1.0)
            _y_val_pm = np.where(_y_val > 0, 1.0, -1.0)
            _mean = np.nanmean(_X_fit, axis=0)
            _std = np.nanstd(_X_fit, axis=0)
            _std = np.where(_std < 1e-6, 1.0, _std)
            _X_fit = np.nan_to_num((_X_fit - _mean) / _std, nan=0.0)
            _X_val = np.nan_to_num((_X_val - _mean) / _std, nan=0.0)
            _clf = sklearn.svm.SVC(
                C=_SVM_C, kernel="rbf", gamma=_SVM_GAMMA, class_weight="balanced",
            )
            _clf.fit(_X_fit, _y_fit)
            _auc = roc_auc_score(_y_val_pm, _clf.decision_function(_X_val))
            print(f"SVM {_name} / {_missing_label}: {_auc:.4f}")
            svm_auc_rows.append(
                {"encoding": _name, "missing_values": _missing_label, "val_roc_auc": _auc}
            )

    svm_auc_df = pd.DataFrame(svm_auc_rows)
    _encodings = [name for name, _ in _builders]
    _x = np.arange(len(_encodings))
    _w = 0.35
    _raw = svm_auc_df.set_index(["encoding", "missing_values"])["val_roc_auc"]
    _fig, _ax = plt.subplots(figsize=(10, 5))
    _ax.bar(
        _x - _w / 2,
        [_raw.loc[e, "no learned missing values"] for e in _encodings],
        width=_w,
        label="Pas d'imputation",
    )
    _ax.bar(
        _x + _w / 2,
        [_raw.loc[e, "learned missing values"] for e in _encodings],
        width=_w,
        label="imputation par XGBoost",
    )
    _ax.set_xticks(_x)
    _ax.set_xticklabels(_encodings, rotation=15, ha="right")
    _ax.set_ylabel("val ROC AUC")
    _ymin = float(svm_auc_df["val_roc_auc"].min()) - 0.01
    _ymax = float(svm_auc_df["val_roc_auc"].max()) + 0.01
    _ax.set_ylim(_ymin, _ymax)
    _ax.legend()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Arbres obliques: comparaison
    """)
    return


@app.cell
def _(
    BoostedClassification,
    DecisionTreeClassifier,
    SklearnObliqueTreeClassifier,
    SklearnObliqueTreeRegressor,
    fit_idx,
    make_dataset,
    np,
    pd,
    plt,
    roc_auc_score,
    sklearn,
):
    from scikit_obliquetree.HHCART import HouseHolderCART
    from scikit_obliquetree.segmentor import MSE, MeanSegmentor

    _N_FIT = 20_000
    _KINDS = [
        "default codes",
        "nn embeddings",
        "nn embeddings + output",
        "target CV5MT",
    ]

    def _xy(ds):
        X_train = ds.X_train.fillna(-1).to_numpy(dtype=np.float64)[:_N_FIT]
        y_train = np.asarray(ds.y_train, dtype=np.float64).ravel()[:_N_FIT]
        X_val = ds.X_val.fillna(-1).to_numpy(dtype=np.float64)
        y_val = np.asarray(ds.y_val, dtype=np.float64).ravel()
        return X_train, y_train, X_val, y_val

    def _auc_proba(model, X_val, y_val):
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_val)
            scores = scores[:, 1] if getattr(scores, "ndim", 1) == 2 else scores
        else:
            scores = model.predict(X_val)
        return roc_auc_score(y_val, scores)

    def _fit_cart(X, y):
        clf = DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=20, class_weight="balanced", random_state=0,
        )
        clf.fit(X, y)
        return clf

    def _fit_oblique(X, y):
        clf = SklearnObliqueTreeClassifier(
            use_oblique=True, max_depth=8, n_pair=2, random_state=42,
        )
        clf.fit(X, y)
        return clf

    def _fit_hhcart(X, y):
        tree = HouseHolderCART(
            MSE(), MeanSegmentor(msl=20), method="eig", max_depth=6, min_samples=200,
        )
        tree.fit(X, y)
        return tree

    def _fit_adaboost(X, y):
        clf = sklearn.ensemble.AdaBoostClassifier(
            SklearnObliqueTreeClassifier(
                use_oblique=True, max_depth=3, n_pair=2, random_state=42,
            ),
            n_estimators=20,
        )
        clf.fit(X, y)
        return clf

    def _fit_gb(X, y):
        base = SklearnObliqueTreeRegressor(
            use_oblique=True, max_depth=3, n_pair=2, random_state=42,
        )
        model = BoostedClassification(base, n_estimators=20, learning_rate=0.1)
        model.fit(X, y)
        return model

    _METHODS = [
        ("CART", _fit_cart),
        ("obliquetree", _fit_oblique),
        ("HHCART", _fit_hhcart),
        ("AdaBoost + obliquetree", _fit_adaboost),
        ("GB + obliquetree", _fit_gb),
    ]

    oblique_auc_rows = []
    for _kind in _KINDS:
        _ds = make_dataset(_kind, fit_idx)
        _X_fit, _y_fit, _X_val, _y_val = _xy(_ds)
        for _name, _fit in _METHODS:
            _model = _fit(_X_fit, _y_fit)
            _auc = _auc_proba(_model, _X_val, _y_val)
            print(f"{_name} / {_kind}: {_auc:.4f}")
            oblique_auc_rows.append(
                {"encoding": _kind, "method": _name, "val_roc_auc": _auc}
            )

    oblique_auc_df = pd.DataFrame(oblique_auc_rows)
    _encodings = _KINDS
    _methods = [name for name, _ in _METHODS]
    _x = np.arange(len(_encodings))
    _n = len(_methods)
    _w = 0.8 / _n
    _raw = oblique_auc_df.set_index(["encoding", "method"])["val_roc_auc"]
    _fig, _ax = plt.subplots(figsize=(12, 5))
    for _i, _name in enumerate(_methods):
        _ax.bar(
            _x + (_i - (_n - 1) / 2) * _w,
            [_raw.loc[e, _name] for e in _encodings],
            width=_w,
            label=_name,
        )
    _ax.set_xticks(_x)
    _ax.set_xticklabels(_encodings, rotation=15, ha="right")
    _ax.set_ylabel("val ROC AUC")
    _ymin = float(oblique_auc_df["val_roc_auc"].min()) - 0.01
    _ymax = float(oblique_auc_df["val_roc_auc"].max()) + 0.01
    _ax.set_ylim(_ymin, _ymax)
    _ax.legend()
    _fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
