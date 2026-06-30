"""
Sécurité de l'API — authentification par clé API (header X-API-Key).

Pour une plateforme de production, remplacez SECRET_API_KEYS par une
vérification en base de données / variable d'environnement par client.
"""
import os
from fastapi import Header, HTTPException, status

# Clés API valides : chargées depuis la variable d'environnement GATEONE_API_KEYS
# (séparées par des virgules), avec une clé de démo par défaut pour le développement.
_default_keys = "demo-key-gateone-2026"
_env_keys = os.getenv("GATEONE_API_KEYS", _default_keys)
VALID_API_KEYS = {k.strip() for k in _env_keys.split(",") if k.strip()}


async def verify_api_key(x_api_key: str = Header(..., description="Clé API d'accès")) -> str:
    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Clé API invalide ou manquante. Fournissez un en-tête X-API-Key valide.",
        )
    return x_api_key
