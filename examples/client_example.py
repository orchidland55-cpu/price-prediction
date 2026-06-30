"""
Exemple de client Python pour intégrer l'API GateOne.immo dans une plateforme.

Usage:
    python3 client_example.py
"""
import requests

BASE_URL = "http://localhost:8000"  # remplacer par l'URL de production
API_KEY = "demo-key-gateone-2026"

HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}


def lister_categories():
    r = requests.get(f"{BASE_URL}/api/v1/categories", headers=HEADERS)
    r.raise_for_status()
    return r.json()


def obtenir_champs(categorie: str):
    r = requests.get(f"{BASE_URL}/api/v1/categories/{categorie}/champs", headers=HEADERS)
    r.raise_for_status()
    return r.json()


def estimer_prix(categorie: str, caracteristiques: dict):
    r = requests.post(
        f"{BASE_URL}/api/v1/predict/{categorie}",
        json=caracteristiques,
        headers=HEADERS,
    )
    r.raise_for_status()
    return r.json()


if __name__ == "__main__":
    print("Catégories disponibles :")
    for cat in lister_categories():
        print(f"  - {cat['categorie']} (R²={cat['r2_final']:.3f}, erreur médiane={cat['mdape_final']:.1f}%)")

    print("\nChamps requis pour 'riads' :")
    champs = obtenir_champs("riads")
    print(f"  Numériques  : {champs['num_features']}")
    print(f"  Catégoriels : {champs['cat_features']}")
    print(f"  Équipements : {champs['equip_features']}")

    print("\nEstimation d'un riad à la Kasbah :")
    resultat = estimer_prix(
        "riads",
        {
            "surface_m2_clean": 100,
            "surface_habitable": 250,
            "chambres": 4,
            "salle_de_bain": 4,
            "nb_etages": 2,
            "salons": 2,
            "condition": "Bon état",
            "standing": "Haut standing",
            "age_du_bien": "6-10 ans",
            "disponibilite": "Immédiate",
            "localisation": "Kasbah",
            "equipements": {"Piscine": 1, "Jardin": 1, "Climatisation": 1},
        },
    )
    print(f"  Prix estimé : {resultat['prix_estime_dh']:,} DH")
    print(f"  Fourchette  : {resultat['prix_estime_min_dh']:,} – {resultat['prix_estime_max_dh']:,} DH")
    print(f"  Modèle      : {resultat['modele_utilise']} (R²={resultat['r2_modele']})")
