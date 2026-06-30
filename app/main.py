"""
============================================================
GateOne.immo — API d'estimation de prix immobiliers
Marrakech, Maroc
============================================================
API REST permettant d'intégrer les modèles de prédiction de
prix (appartements, villas, riads, maisons, bureaux, magasins,
terrains, maisons d'hôtes) dans une plateforme immobilière.
============================================================
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routers import estimation

app = FastAPI(
    title="GateOne.immo — API d'estimation de prix",
    description=(
        "API de prédiction du prix de biens immobiliers à Marrakech, basée sur des "
        "modèles XGBoost / LightGBM entraînés par catégorie de bien.\n\n"
        "Authentification : en-tête `X-API-Key` requis sur tous les endpoints `/api/v1/*`.\n\n"
        "Catégories disponibles : appartements, bureaux, magasins, maisons, riads, "
        "terrains, villas, maisons_dhotes."
    ),
    version="1.0.0",
    contact={"name": "GateOne.immo", "url": "https://gateone.immo"},
)

# CORS — à restreindre aux domaines de la plateforme en production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(estimation.router)


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Erreur interne du serveur: {str(exc)}"},
    )


@app.get("/", tags=["health"], summary="Vérification de l'état de l'API")
async def root():
    return {
        "service": "GateOne.immo API",
        "status": "online",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["health"], summary="Healthcheck")
async def health():
    return {"status": "ok"}
