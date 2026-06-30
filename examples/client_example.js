/**
 * Exemple d'intégration de l'API GateOne.immo côté front-end (JavaScript).
 * À adapter à votre framework (React, Vue, vanilla JS...).
 */

const BASE_URL = "http://localhost:8000"; // remplacer par l'URL de production
const API_KEY = "demo-key-gateone-2026"; // ne jamais exposer la vraie clé côté client en prod
// → en production, faites passer ces appels par votre propre backend qui détient la clé.

async function listerCategories() {
  const res = await fetch(`${BASE_URL}/api/v1/categories`, {
    headers: { "X-API-Key": API_KEY },
  });
  if (!res.ok) throw new Error(`Erreur ${res.status}`);
  return res.json();
}

async function obtenirChamps(categorie) {
  const res = await fetch(`${BASE_URL}/api/v1/categories/${categorie}/champs`, {
    headers: { "X-API-Key": API_KEY },
  });
  if (!res.ok) throw new Error(`Erreur ${res.status}`);
  return res.json();
}

async function estimerPrix(categorie, caracteristiques) {
  const res = await fetch(`${BASE_URL}/api/v1/predict/${categorie}`, {
    method: "POST",
    headers: {
      "X-API-Key": API_KEY,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(caracteristiques),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || `Erreur ${res.status}`);
  }
  return res.json();
}

// ── Exemple d'utilisation ──
(async () => {
  const categories = await listerCategories();
  console.log("Catégories disponibles :", categories);

  const estimation = await estimerPrix("appartements", {
    surface_m2_clean: 60,
    surface_habitable: 60,
    chambres: 3,
    salle_de_bain: 2,
    etage: 2,
    salons: 1,
    frais_syndic: 500,
    condition: "Neuf",
    standing: "Haut standing",
    age_du_bien: "1-5 ans",
    disponibilite: "Immédiate",
    type_appartement: "Appartement",
    localisation: "Guéliz",
    equipements: { equip_Ascenseur: 1, equip_Parking: 1, equip_Balcon: 1 },
  });

  console.log(`Prix estimé : ${estimation.prix_estime_dh.toLocaleString()} DH`);
  console.log(
    `Fourchette : ${estimation.prix_estime_min_dh.toLocaleString()} – ${estimation.prix_estime_max_dh.toLocaleString()} DH`
  );
})();
