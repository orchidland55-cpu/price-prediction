"""
============================================================
Entraînement générique des modèles de prédiction de prix
GateOne.immo | Marrakech
============================================================
Ce script lit les fichiers Excel réels, détecte automatiquement
les colonnes catégorielles / numériques / équipements et entraîne
un modèle XGBoost + LightGBM par catégorie de bien.

Les catégories suivent FIDÈLEMENT les colonnes réellement présentes
dans les fichiers de données (pas les valeurs codées en dur dans les
anciens notebooks, qui ne correspondaient plus aux données actuelles).
============================================================
"""
import json
import re
import warnings
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import lightgbm as lgb
from category_encoders import TargetEncoder
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

SEED = 42
N_FOLDS = 5
IQR_FACTOR = 2.5
DATA_DIR = "data"
MODEL_DIR = "models"

# Ordres ordinaux connus — appliqués UNIQUEMENT si les valeurs détectées
# dans les données correspondent (sinon fallback sur l'ordre alphabétique/détecté).
KNOWN_ORDERS = {
    "condition": ["à rénover", "Bon état", "Neuf"],
    "standing": ["Economique", "Moyen standing", "Haut standing"],
    "Classement touristique": ["Economique", "Moyen standing", "Haut standing"],
    "age_du_bien": ["Moins de 1 an", "1-5 ans", "6-10 ans", "11-20 ans", "21+ ans"],
    "disponibilite": ["Immédiate", "2 mois", "3 mois", "+4 mois"],
    "zoning": ["Agricole", "Maison", "Villa", "Immeuble", "Industriel", "Service public"],
}

# Configuration par catégorie : fichier, colonne prix, colonnes id à exclure
CATEGORY_CONFIG = {
    "appartements": {
        "file": "appartements.xlsx",
        "price_col": "prix",
        "price_is_numeric": False,
        "exclude": ["categorie", "titre", "prix_m2"],
    },
    "bureaux": {
        "file": "bureaux.xlsx",
        "price_col": "prix",
        "price_is_numeric": False,
        "exclude": ["categorie", "titre", "prix_m2"],
    },
    "magasins": {
        "file": "magasins.xlsx",
        "price_col": "prix",
        "price_is_numeric": False,
        "exclude": ["categorie", "titre", "prix_m2"],
    },
    "maisons": {
        "file": "maisons.xlsx",
        "price_col": "prix",
        "price_is_numeric": False,
        "exclude": ["categorie", "titre", "prix_m2"],
    },
    "riads": {
        "file": "riads_final.xlsx",
        "price_col": "prix",
        "price_is_numeric": False,
        "exclude": ["categorie", "titre", "prix_m2"],
    },
    "terrains": {
        "file": "terrains.xlsx",
        "price_col": "prix",
        "price_is_numeric": False,
        "exclude": ["categorie", "titre", "prix_m2"],
    },
    "villas": {
        "file": "villas.xlsx",
        "price_col": "prix",
        "price_is_numeric": False,
        "exclude": ["categorie", "titre", "prix_m2"],
    },
    "maisons_dhotes": {
        "file": "maison_dhotes_7.xlsx",
        "price_col": "prix DH",
        "price_is_numeric": True,
        "exclude": ["N°", "titre", "prix_m2"],
    },
}


def parse_prix(val):
    if isinstance(val, (int, float)):
        return float(val)
    val = (
        str(val)
        .replace("\xa0", "")
        .replace(" ", "")
        .replace("DH", "")
        .replace(",", ".")
    )
    try:
        return float(val)
    except ValueError:
        return np.nan


def sanitize_colname(c):
    """Nom de colonne sûr pour JSON / clés Python (utilisé pour les noms exposés à l'API)."""
    c2 = re.sub(r"\s+", "_", str(c).strip())
    return c2


def detect_feature_types(df, price_col, exclude_cols):
    """Détecte automatiquement les colonnes numériques, catégorielles (texte non-binaire)
    et binaires (équipements 0/1) à partir du dataframe réel."""
    candidate_cols = [c for c in df.columns if c not in exclude_cols + [price_col]]

    num_features, cat_features, equip_features = [], [], []

    for c in candidate_cols:
        series = df[c]
        if pd.api.types.is_numeric_dtype(series):
            uniq = series.dropna().unique()
            # Binaire (0/1 uniquement) -> équipement
            if set(pd.unique(series.dropna())).issubset({0, 1}) and len(uniq) <= 2:
                equip_features.append(c)
            else:
                num_features.append(c)
        else:
            # Texte : localisation traitée séparément (target encoding à forte cardinalité)
            if c == "localisation":
                continue
            nuniq = series.dropna().nunique()
            if nuniq <= 10:
                cat_features.append(c)
            else:
                # Texte à forte cardinalité non prévu -> ignoré (ex: descriptions libres)
                continue

    return num_features, cat_features, equip_features


def build_ordinal_categories(df, cat_features):
    """Construit la liste ordonnée de catégories pour OrdinalEncoder, en utilisant
    KNOWN_ORDERS quand les valeurs correspondent, sinon un ordre basé sur la
    fréquence (du moins cher au plus cher serait idéal mais on n'a pas le prix
    encore ; on utilise donc l'ordre alphabétique trié comme fallback neutre)."""
    categories = []
    for c in cat_features:
        present_vals = set(df[c].dropna().unique().tolist())
        if c in KNOWN_ORDERS and present_vals.issubset(set(KNOWN_ORDERS[c])):
            order = [v for v in KNOWN_ORDERS[c] if v in present_vals]
            # Ajoute les valeurs présentes non couvertes par KNOWN_ORDERS (sécurité)
            order += sorted(present_vals - set(order))
        else:
            order = sorted(present_vals)
        order.append("Inconnu")
        categories.append(order)
    return categories


def mdape(y_true, y_pred):
    return float(np.median(np.abs((y_true - y_pred) / y_true)) * 100)


def train_category(name, config):
    print(f"\n{'='*60}\n CATEGORIE: {name}\n{'='*60}")
    path = f"{DATA_DIR}/{config['file']}"
    df = pd.read_excel(path)
    print(f"[INFO] Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")

    price_col = config["price_col"]

    # ── Nettoyage prix ──
    if config["price_is_numeric"]:
        df["prix_num"] = pd.to_numeric(df[price_col], errors="coerce")
    else:
        df["prix_num"] = df[price_col].apply(parse_prix)

    df.dropna(subset=["prix_num"], inplace=True)
    df = df[df["prix_num"] > 0]

    Q1, Q3 = df["prix_num"].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    df = df[
        (df["prix_num"] >= Q1 - IQR_FACTOR * IQR)
        & (df["prix_num"] <= Q3 + IQR_FACTOR * IQR)
    ]
    print(f"[INFO] Après suppression outliers : {len(df)} lignes")

    # ── Détection des features ──
    exclude_cols = config["exclude"] + ["prix_num", price_col]
    num_features, cat_features, equip_features = detect_feature_types(
        df, price_col, exclude_cols
    )
    has_localisation = "localisation" in df.columns

    print(f"[INFO] num_features ({len(num_features)}): {num_features}")
    print(f"[INFO] cat_features ({len(cat_features)}): {cat_features}")
    print(f"[INFO] equip_features ({len(equip_features)}): {equip_features}")
    print(f"[INFO] localisation présente: {has_localisation}")

    for col in cat_features:
        df[col] = df[col].fillna("Inconnu")
    if num_features:
        df[num_features] = df[num_features].fillna(0)
    if equip_features:
        df[equip_features] = df[equip_features].fillna(0)
    if has_localisation:
        df["localisation"] = df["localisation"].fillna("Inconnue")

    feature_cols = (
        num_features
        + cat_features
        + equip_features
        + (["localisation"] if has_localisation else [])
    )

    X = df[feature_cols].copy()
    y = np.log1p(df["prix_num"])

    # ── Encodage ordinal des catégorielles ──
    ord_enc = None
    if cat_features:
        categories = build_ordinal_categories(df, cat_features)
        ord_enc = OrdinalEncoder(
            categories=categories, handle_unknown="use_encoded_value", unknown_value=-1
        )
        X[cat_features] = ord_enc.fit_transform(X[cat_features])

    # ── Target encoding localisation ──
    te = None
    if has_localisation:
        smoothing = 10 if len(df) > 500 else (15 if len(df) > 150 else 20)
        te = TargetEncoder(cols=["localisation"], smoothing=smoothing)
        X_enc = te.fit_transform(X, y)
    else:
        X_enc = X

    print(f"[INFO] Shape features finale : {X_enc.shape}")

    # ── Régularisation adaptée à la taille du dataset ──
    n = len(df)
    if n < 150:
        params = dict(
            n_estimators=300, learning_rate=0.04, max_depth=3,
            subsample=0.7, colsample_bytree=0.7,
            reg_alpha=1.0, reg_lambda=3.0,
        )
        lgb_extra = dict(min_child_samples=8)
        xgb_extra = dict(min_child_weight=5)
    elif n < 400:
        params = dict(
            n_estimators=400, learning_rate=0.04, max_depth=4,
            subsample=0.75, colsample_bytree=0.75,
            reg_alpha=0.5, reg_lambda=2.0,
        )
        lgb_extra = dict(min_child_samples=8)
        xgb_extra = dict(min_child_weight=5)
    else:
        params = dict(
            n_estimators=600, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
        )
        lgb_extra = {}
        xgb_extra = {}

    xgb_model = xgb.XGBRegressor(
        **params, **xgb_extra, random_state=SEED, verbosity=0, n_jobs=-1
    )
    lgb_model = lgb.LGBMRegressor(
        **params, **lgb_extra, random_state=SEED, verbose=-1, n_jobs=-1
    )

    # ── Cross-validation ──
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    cv_results = {}
    for mname, model in [("xgb", xgb_model), ("lgb", lgb_model)]:
        r2 = cross_val_score(model, X_enc, y, cv=kf, scoring="r2", n_jobs=-1)
        mae = cross_val_score(
            model, X_enc, y, cv=kf, scoring="neg_mean_absolute_error", n_jobs=-1
        )
        cv_results[mname] = {"r2_mean": float(r2.mean()), "r2_std": float(r2.std()),
                              "mae_log_mean": float(-mae.mean())}
        print(f"{mname:5s} CV | R²={r2.mean():.4f} ± {r2.std():.4f} | MAE(log)={-mae.mean():.4f}")

    # ── Entraînement final ──
    xgb_model.fit(X_enc, y)
    lgb_model.fit(X_enc, y)

    metrics = {"cv": cv_results, "final": {}}
    y_true = np.expm1(y)
    for mname, model in [("xgb", xgb_model), ("lgb", lgb_model)]:
        preds = np.expm1(model.predict(X_enc))
        m = mdape(y_true.values, preds)
        r2v = r2_score(y_true, preds)
        metrics["final"][mname] = {"mdape": m, "r2": float(r2v)}
        print(f"{mname:5s} FINAL | MdAPE={m:.2f}%  R²={r2v:.4f}")

    # Choix du meilleur modèle par défaut (meilleur R² en CV moyen)
    best_model = "xgb" if cv_results["xgb"]["r2_mean"] >= cv_results["lgb"]["r2_mean"] else "lgb"
    print(f"[INFO] Meilleur modèle (CV) : {best_model}")

    # ── Statistiques pour validation des inputs côté API ──
    stats = {
        "n_rows_trained": int(len(df)),
        "price_min": float(df["prix_num"].min()),
        "price_max": float(df["prix_num"].max()),
        "price_median": float(df["prix_num"].median()),
        "localisations": sorted(df["localisation"].dropna().unique().tolist()) if has_localisation else [],
        "cat_values": {c: sorted(df[c].dropna().unique().tolist()) for c in cat_features},
        "num_ranges": {
            c: {"min": float(df[c].min()), "max": float(df[c].max()), "median": float(df[c].median())}
            for c in num_features
        },
    }

    artefacts = {
        "xgb_model": xgb_model,
        "lgb_model": lgb_model,
        "ord_enc": ord_enc,
        "target_enc": te,
        "feature_cols": feature_cols,
        "cat_features": cat_features,
        "num_features": num_features,
        "equip_features": equip_features,
        "has_localisation": has_localisation,
        "categorie": name,
        "best_model": best_model,
        "metrics": metrics,
        "stats": stats,
    }
    out_path = f"{MODEL_DIR}/model_{name}.pkl"
    joblib.dump(artefacts, out_path)
    print(f"[OK] Modèle sauvegardé → {out_path}")

    return {
        "categorie": name,
        "metrics": metrics,
        "best_model": best_model,
        "feature_cols": feature_cols,
        "cat_features": cat_features,
        "num_features": num_features,
        "equip_features": equip_features,
        "has_localisation": has_localisation,
        "stats": stats,
    }


def main():
    summary = {}
    for name, config in CATEGORY_CONFIG.items():
        try:
            summary[name] = train_category(name, config)
        except Exception as e:
            print(f"[ERREUR] {name}: {e}")
            raise

    with open(f"{MODEL_DIR}/registry.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Registre des modèles sauvegardé → {MODEL_DIR}/registry.json")


if __name__ == "__main__":
    main()
