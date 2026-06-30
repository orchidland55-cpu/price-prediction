"""
Schémas Pydantic — requêtes et réponses de l'API GateOne.immo
"""
from typing import Dict, Optional

from pydantic import BaseModel, Field, field_validator

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


class PredictionRequest(BaseModel):
    """Requête générique de prédiction de prix.

    Tous les champs numériques/catégoriels sont optionnels au niveau du schéma
    (des valeurs par défaut neutres sont appliquées côté serveur) mais il est
    recommandé de fournir un maximum d'informations pour une estimation fiable.
    Consultez GET /api/v1/categories/{categorie}/champs pour la liste exacte
    des champs pertinents pour chaque catégorie.
    """

    # Champs numériques génériques (communs à plusieurs catégories)
    surface_m2_clean: Optional[float] = Field(None, description="Surface totale en m²")
    surface_habitable: Optional[float] = Field(None, description="Surface habitable en m²")
    surface_terrain: Optional[float] = Field(None, description="Surface du terrain en m²")
    chambres: Optional[float] = Field(None, description="Nombre de chambres")
    salle_de_bain: Optional[float] = Field(None, description="Nombre de salles de bain")
    salles_bain: Optional[float] = Field(None, description="Nombre de salles de bain (maisons d'hôtes)")
    salons: Optional[float] = Field(None, description="Nombre de salons")
    etage: Optional[float] = Field(None, description="Étage (appartements)")
    etages: Optional[float] = Field(None, description="Nombre d'étages (maisons d'hôtes)")
    nb_etage: Optional[float] = Field(None, description="Nombre d'étages (bureaux)")
    nb_etages: Optional[float] = Field(None, description="Nombre d'étages (riads/villas)")
    nombre_detage: Optional[float] = Field(None, description="Nombre d'étages (maisons)")
    nb_pieces: Optional[float] = Field(None, description="Nombre de pièces (bureaux)")
    frais_syndic: Optional[float] = Field(None, description="Frais de syndic mensuels (DH)")
    distance_route_m: Optional[float] = Field(
        None, alias="distance_route m", description="Distance à la route en mètres (terrains)"
    )

    # Champs catégoriels génériques
    condition: Optional[str] = Field(None, description="État du bien: 'à rénover', 'Bon état', 'Neuf'")
    standing: Optional[str] = Field(None, description="Standing: 'Economique', 'Moyen standing', 'Haut standing'")
    age_du_bien: Optional[str] = Field(None, description="Âge du bien")
    disponibilite: Optional[str] = Field(None, description="Disponibilité")
    type_appartement: Optional[str] = Field(None, description="Type: 'Studio', 'Duplex', 'Appartement'")
    zoning: Optional[str] = Field(None, description="Zonage du terrain")
    type_bien: Optional[str] = Field(None, description="Type de bien (maisons d'hôtes)")
    classement_touristique: Optional[str] = Field(
        None, alias="Classement touristique", description="Classement touristique (maisons d'hôtes)"
    )

    # Localisation
    localisation: Optional[str] = Field(None, description="Quartier / zone de Marrakech")

    # Équipements — dict libre {nom_equipement: 0 ou 1}
    equipements: Optional[Dict[str, int]] = Field(
        default=None,
        description="Dictionnaire des équipements présents, ex: {'Piscine': 1, 'Jardin': 0}",
    )

    # Choix du modèle
    use_model: Optional[str] = Field(
        None, description="Forcer le modèle utilisé : 'xgb' ou 'lgb' (sinon le meilleur modèle est utilisé)"
    )

    class Config:
        populate_by_name = True
        extra = "allow"  # autorise des champs spécifiques à une catégorie non listés ici


class PredictionResponse(BaseModel):
    categorie: str
    prix_estime_dh: int
    prix_estime_min_dh: int
    prix_estime_max_dh: int
    modele_utilise: str
    marge_erreur_pct: float
    r2_modele: float
    prix_median_dataset_dh: float


class CategoryInfo(BaseModel):
    categorie: str
    n_rows_trained: int
    best_model: str
    r2_final: float
    mdape_final: float


class ErrorResponse(BaseModel):
    detail: str
