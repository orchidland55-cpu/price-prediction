"""
Router /api/v1 — Estimation de prix immobiliers GateOne.immo
"""
from fastapi import APIRouter, Depends, HTTPException, Path

from app.core.predictor import (
    CategoryNotFoundError,
    PredictionInputError,
    VALID_CATEGORIES,
    get_required_fields,
    list_categories,
    predict_price,
)
from app.core.security import verify_api_key
from app.schemas.prediction import (
    CategoryInfo,
    ErrorResponse,
    PredictionRequest,
    PredictionResponse,
)

router = APIRouter(prefix="/api/v1", tags=["estimation"])


@router.get(
    "/categories",
    response_model=list[CategoryInfo],
    summary="Lister les catégories de biens disponibles",
    description="Retourne la liste des catégories de biens immobiliers pour lesquelles "
    "un modèle d'estimation de prix est disponible, avec leurs métriques de performance.",
)
async def get_categories(api_key: str = Depends(verify_api_key)):
    return list_categories()


@router.get(
    "/categories/{categorie}/champs",
    summary="Décrire les champs attendus pour une catégorie",
    description="Retourne la liste des champs numériques, catégoriels et équipements "
    "pertinents pour la catégorie demandée, ainsi que les valeurs/plages valides "
    "observées dans les données d'entraînement (utile pour générer un formulaire).",
    responses={404: {"model": ErrorResponse}},
)
async def get_category_fields(
    categorie: str = Path(..., description="Nom de la catégorie", examples=VALID_CATEGORIES),
    api_key: str = Depends(verify_api_key),
):
    try:
        return get_required_fields(categorie)
    except CategoryNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post(
    "/predict/{categorie}",
    response_model=PredictionResponse,
    summary="Estimer le prix d'un bien immobilier",
    description="Calcule une estimation de prix en DH pour un bien de la catégorie "
    "spécifiée, à partir de ses caractéristiques (surface, état, équipements, "
    "localisation, etc.).",
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
async def predict(
    categorie: str = Path(..., description="Catégorie du bien", examples=VALID_CATEGORIES),
    payload: PredictionRequest = ...,
    api_key: str = Depends(verify_api_key),
):
    if categorie not in VALID_CATEGORIES:
        raise HTTPException(
            status_code=404,
            detail=f"Catégorie inconnue: '{categorie}'. Catégories valides: {VALID_CATEGORIES}",
        )
    try:
        data = payload.model_dump(by_alias=False, exclude_none=True)
        use_model = data.pop("use_model", None)
        result = predict_price(categorie, data, use_model=use_model)
        return result
    except CategoryNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PredictionInputError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de la prédiction: {e}")
