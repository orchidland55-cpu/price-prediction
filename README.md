# GateOne.immo — Estimateur de prix (app Streamlit autonome)

Version **tout-en-un** : pas d'API séparée à héberger. Les modèles
XGBoost / LightGBM sont chargés et exécutés directement dans l'app
Streamlit, depuis le dossier `models/` commité dans ce repo.

## Fichiers à mettre à la racine de votre repo GitHub

- `app.py`
- `requirements.txt`
- `models/` (dossier complet avec les `.pkl` et `registry.json`)
- `.gitignore` (optionnel mais recommandé)

> Pas de `runtime.txt` dans cette version : au moment de la rédaction,
> Streamlit Community Cloud ignore ce fichier pour de nombreux comptes
> (bug connu, forçant Python 3.13/3.14 quoi qu'il arrive). Les versions
> dans `requirements.txt` ci-dessous sont donc choisies pour être
> compatibles avec ces versions récentes de Python.

## Déploiement sur Streamlit Community Cloud

1. Poussez ces fichiers sur GitHub (le dossier `models/` doit être inclus).
2. Sur [share.streamlit.io](https://share.streamlit.io) : "New app" →
   sélectionnez le repo, la branche, et `app.py` comme fichier principal.
3. Dans **"Advanced settings"** au moment du déploiement, vous pouvez
   tenter de sélectionner manuellement une version Python (3.11 ou 3.12
   si proposé) — c'est plus fiable que `runtime.txt` actuellement. Si
   l'option n'apparaît pas ou ne change rien, ce n'est pas grave : les
   versions figées dans `requirements.txt` fonctionnent aussi avec la
   version par défaut imposée par la plateforme.
4. Déployez.

## Si le build reste bloqué plus de 10 minutes

L'installation de pandas/numpy/scikit-learn/xgboost/lightgbm peut prendre
du temps (plusieurs centaines de Mo à télécharger). Patientez jusqu'à 10
minutes avant de considérer que c'est bloqué. Si ça échoue avec une erreur
explicite, copiez le texte du message d'erreur — la cause est presque
toujours une incompatibilité de version entre un paquet et la version de
Python utilisée par la plateforme à ce moment-là.

## Lancement en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Ré-entraînement des modèles

Si vous voulez ré-entraîner avec de nouvelles données, utilisez le script
`train_models.py` du projet complet (dossier `backend/` fourni séparément),
en l'exécutant avec les **mêmes versions** que `requirements.txt` ci-contre
(important : des versions différentes de scikit-learn / xgboost / pandas
entre l'entraînement et le déploiement peuvent rendre les fichiers `.pkl`
illisibles).
