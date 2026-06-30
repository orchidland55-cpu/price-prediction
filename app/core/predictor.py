"""
Module central de gestion des modèles ML : chargement, cache, prédiction.
"""
import json
import os
from functools import lru_cache
from typing import Optional

import joblib
import numpy as np
import pandas as pd

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")

VALID_CATEGORIES = [
    "appartements",
    "bureaux",
    "magasins",
    "maisons",
    "riads",
    "terrains",
    "villas",
    "maisons_dhotes",
]


class CategoryNotFoundError(Exception):
    pass


class PredictionInputError(Exception):
    pass


@lru_cache(maxsize=1)
def load_registry() -> dict:
    path = os.path.join(MODEL_DIR, "registry.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=None)
def load_artefacts(categorie: str) -> dict:
    if categorie not in VALID_CATEGORIES:
        raise CategoryNotFoundError(f"Catégorie inconnue: {categorie}")
    path = os.path.join(MODEL_DIR, f"model_{categorie}.pkl")
    if not os.path.exists(path):
        raise CategoryNotFoundError(f"Modèle introuvable pour la catégorie: {categorie}")
    return joblib.load(path)


def get_category_schema(categorie: str) -> dict:
    """Retourne la définition complète des champs attendus pour une catégorie
    (utilisé pour générer dynamiquement la doc / le formulaire front)."""
    registry = load_registry()
    if categorie not in registry:
        raise CategoryNotFoundError(f"Catégorie inconnue: {categorie}")
    return registry[categorie]


def list_categories() -> list:
    registry = load_registry()
    return [
        {
            "categorie": k,
            "n_rows_trained": v["stats"]["n_rows_trained"],
            "best_model": v["best_model"],
            "r2_final": v["metrics"]["final"][v["best_model"]]["r2"],
            "mdape_final": v["metrics"]["final"][v["best_model"]]["mdape"],
        }
        for k, v in registry.items()
    ]


def _build_feature_row(art: dict, payload: dict) -> pd.DataFrame:
    """Construit une ligne de features prête à être encodée, à partir d'un payload
    utilisateur (dict). Applique les valeurs par défaut pour les champs manquants."""
    row = {}

    # Numériques
    for c in art["num_features"]:
        val = payload.get(c, 0)
        if val is None:
            val = 0
        row[c] = float(val)

    # Catégorielles
    for c in art["cat_features"]:
        val = payload.get(c)
        row[c] = val if val not in (None, "") else "Inconnu"

    # Équipements (0/1)
    equipements = payload.get("equipements") or {}
    for ef in art["equip_features"]:
        v = equipements.get(ef, payload.get(ef, 0))
        row[ef] = int(bool(v))

    # Localisation
    if art["has_localisation"]:
        loc = payload.get("localisation")
        row["localisation"] = loc if loc not in (None, "") else "Inconnue"

    return pd.DataFrame([row])[art["feature_cols"]]


def predict_price(categorie: str, payload: dict, use_model: Optional[str] = None) -> dict:
    """Prédit le prix d'un bien immobilier pour une catégorie donnée.

    payload: dict contenant les champs numériques, catégoriels, et la clé
    optionnelle 'equipements' (dict nom_equip -> 0/1) ou directement les
    clés d'équipement au niveau racine.
    """
    art = load_artefacts(categorie)

    model_choice = use_model or art["best_model"]
    if model_choice not in ("xgb", "lgb"):
        raise PredictionInputError("use_model doit être 'xgb' ou 'lgb'")

    X_pred = _build_feature_row(art, payload)

    if art["cat_features"]:
        X_pred[art["cat_features"]] = art["ord_enc"].transform(X_pred[art["cat_features"]])

    if art["has_localisation"] and art["target_enc"] is not None:
        X_pred = art["target_enc"].transform(X_pred)

    model = art["xgb_model"] if model_choice == "xgb" else art["lgb_model"]
    pred_log = model.predict(X_pred)[0]
    price = float(np.expm1(pred_log))
    price = max(price, 0.0)

    stats = art["stats"]
    # Marge d'incertitude basée sur le MdAPE du modèle choisi
    mdape = art["metrics"]["final"][model_choice]["mdape"] / 100.0
    low = price * (1 - mdape)
    high = price * (1 + mdape)

    return {
        "categorie": categorie,
        "prix_estime_dh": round(price),
        "prix_estime_min_dh": round(low),
        "prix_estime_max_dh": round(high),
        "modele_utilise": model_choice,
        "marge_erreur_pct": round(art["metrics"]["final"][model_choice]["mdape"], 2),
        "r2_modele": round(art["metrics"]["final"][model_choice]["r2"], 4),
        "prix_median_dataset_dh": stats["price_median"],
    }


def get_required_fields(categorie: str) -> dict:
    art = load_artefacts(categorie)
    return {
        "num_features": art["num_features"],
        "cat_features": art["cat_features"],
        "cat_values": art["stats"]["cat_values"],
        "equip_features": art["equip_features"],
        "has_localisation": art["has_localisation"],
        "localisations": art["stats"]["localisations"] if art["has_localisation"] else [],
        "num_ranges": art["stats"]["num_ranges"],
    }
