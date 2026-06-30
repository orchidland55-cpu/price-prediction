# GateOne.immo — API d'estimation de prix immobiliers

API REST FastAPI permettant d'intégrer les modèles de prédiction de prix
immobiliers (Marrakech) dans une plateforme tierce : site web, application
mobile, CRM, etc.

## 1. Contenu du projet

```
backend/
├── app/
│   ├── main.py                 # Application FastAPI (point d'entrée)
│   ├── core/
│   │   ├── predictor.py        # Chargement des modèles + logique de prédiction
│   │   └── security.py         # Authentification par clé API
│   ├── routers/
│   │   └── estimation.py       # Endpoints /api/v1/*
│   └── schemas/
│       └── prediction.py       # Schémas Pydantic (requêtes/réponses)
├── models/                     # Modèles entraînés (.pkl) + registre (registry.json)
├── data/                       # Fichiers Excel sources (entraînement)
├── examples/
│   ├── client_example.py       # Exemple d'intégration backend (Python)
│   └── client_example.js       # Exemple d'intégration front (JavaScript)
├── train_models.py             # Script d'entraînement (à relancer si nouvelles données)
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## 2. Catégories de biens disponibles

| Catégorie         | Lignes entraînées | Modèle retenu | R² | Erreur médiane |
|--------------------|-------------------|---------------|-----|----------------|
| appartements       | 3075              | XGBoost       | 0.96 | 6.0%  |
| maisons            | 687               | XGBoost       | 0.99 | 3.1%  |
| villas             | 1265              | XGBoost       | 0.97 | 4.5%  |
| riads              | 286               | XGBoost       | 0.92 | 8.6%  |
| maisons_dhotes     | 151               | XGBoost       | 0.97 | 6.9%  |
| magasins           | 488               | LightGBM      | 0.87 | 18.3% |
| bureaux            | 87                | LightGBM      | 0.70 | 17.0% |
| terrains           | 1141              | LightGBM      | 0.55 | 29.2% |

Ces métriques sont aussi exposées dynamiquement via `GET /api/v1/categories`.
Les catégories avec peu de données (bureaux, terrains) ont une marge d'erreur
plus large : c'est attendu et reflété dans le champ `marge_erreur_pct` de
chaque réponse.

## 3. Démarrage rapide

### Avec Docker (recommandé)

```bash
cd backend
docker compose up --build
```

L'API est alors disponible sur `http://localhost:8000`, et la documentation
interactive sur `http://localhost:8000/docs`.

### Sans Docker

```bash
cd backend
pip install -r requirements.txt
export GATEONE_API_KEYS="demo-key-gateone-2026"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 4. Authentification

Tous les endpoints `/api/v1/*` nécessitent un en-tête HTTP `X-API-Key`.

Les clés valides sont définies via la variable d'environnement
`GATEONE_API_KEYS` (séparées par des virgules). Exemple :

```bash
export GATEONE_API_KEYS="cle-client-a,cle-client-b,cle-client-c"
```

> ⚠️ En production, ne stockez pas la clé API côté front-end (JavaScript
> exposé au navigateur). Faites transiter les appels par votre propre
> backend, qui détient la clé.

## 5. Endpoints

### `GET /api/v1/categories`
Liste les catégories disponibles avec leurs métriques de performance.

### `GET /api/v1/categories/{categorie}/champs`
Décrit les champs numériques, catégoriels et équipements pertinents pour
une catégorie donnée, ainsi que les valeurs/plages observées dans les
données d'entraînement. Utile pour générer dynamiquement un formulaire
côté plateforme.

### `POST /api/v1/predict/{categorie}`
Calcule une estimation de prix. Corps de la requête : objet JSON avec les
caractéristiques du bien (voir exemples ci-dessous). Tous les champs sont
optionnels (valeur par défaut neutre appliquée si absents), mais plus
d'informations = estimation plus fiable.

**Exemple de requête (villa) :**

```json
POST /api/v1/predict/villas
X-API-Key: demo-key-gateone-2026

{
  "surface_terrain": 800,
  "surface_habitable": 200,
  "chambres": 5,
  "salle_de_bain": 4,
  "nb_etages": 2,
  "salons": 2,
  "condition": "Bon état",
  "standing": "Haut standing",
  "age_du_bien": "1-5 ans",
  "disponibilite": "Immédiate",
  "localisation": "Palmeraie",
  "equipements": { "Piscine": 1, "Jardin": 1, "Parking": 1, "Climatisation": 1 }
}
```

**Exemple de réponse :**

```json
{
  "categorie": "villas",
  "prix_estime_dh": 5003224,
  "prix_estime_min_dh": 4777277,
  "prix_estime_max_dh": 5229172,
  "modele_utilise": "xgb",
  "marge_erreur_pct": 4.52,
  "r2_modele": 0.9689,
  "prix_median_dataset_dh": 3900000.0
}
```

Le champ `use_model` (`"xgb"` ou `"lgb"`) permet de forcer un modèle
spécifique au lieu du meilleur modèle par défaut.

## 6. Particularités par catégorie

- **terrains** : le champ distance à la route s'appelle `"distance_route m"`
  (avec un espace, hérité du fichier source). Utilisez l'alias Pydantic ou
  la clé brute identique dans le JSON.
- **maisons_dhotes** : utilise `"Classement touristique"` (avec espace et
  majuscule) au lieu de `standing`.
- **equipements** : peut être passé soit comme objet `{"Piscine": 1}`, soit
  directement comme clés au niveau racine du payload (les deux sont
  acceptés grâce à `extra = "allow"` dans le schéma).

Consultez toujours `GET /api/v1/categories/{categorie}/champs` avant
d'intégrer une catégorie : il donne la liste exacte des noms de champs
attendus pour cette catégorie spécifique.

## 7. Ré-entraînement des modèles

Si de nouvelles données sont disponibles (nouveaux fichiers Excel dans
`data/`, mêmes noms de colonnes), relancez :

```bash
python3 train_models.py
```

Cela régénère tous les fichiers `models/model_*.pkl` et `models/registry.json`.
Le script détecte automatiquement les colonnes numériques, catégorielles
et les équipements binaires à partir des données — il n'y a pas de noms
de colonnes codés en dur à maintenir manuellement.

## 8. Limitations connues

- Les catégories `bureaux` et `terrains` ont un R² de validation croisée
  modeste (respectivement 0.50–0.53 et 0.40–0.43), en raison d'un nombre de
  lignes d'entraînement limité ou d'une forte hétérogénéité des biens
  (terrains agricoles vs. urbains). Les estimations restent indicatives.
- Le modèle ne connaît que les localisations vues pendant l'entraînement ;
  une localisation inconnue est traitée par un encodage neutre (moyenne
  globale), ce qui peut réduire la précision locale.
- Aucune donnée temporelle (saisonnalité du marché) n'est prise en compte :
  les modèles reflètent un instantané du marché au moment de la collecte
  des données.
