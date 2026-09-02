from __future__ import annotations

import pandas as pd

from entrainement_xgboost import (
    entrainer_xgboost,
    FEATURES,
)


# ============================================================
# PARAMÈTRE DE PRÉVISION
# ============================================================

HORIZON_PREVISION = 12


# ============================================================
# PRÉVISION RÉCURSIVE XGBOOST
# ============================================================

def prevoir_futur(
    modele,
    df_ml: pd.DataFrame,
    horizon: int = HORIZON_PREVISION,
) -> pd.DataFrame:
    """
    Produit des prévisions futures de manière récursive.

    Le modèle utilise :
        - mois
        - trimestre
        - saison
        - lag_1
        - lag_2
        - moyenne_mobile_3

    À chaque nouvelle période :
        1. les variables calendaires sont créées ;
        2. les derniers volumes disponibles servent à construire
           lag_1 et lag_2 ;
        3. la moyenne mobile est calculée sur les trois dernières
           valeurs disponibles ;
        4. XGBoost produit la prévision ;
        5. cette prévision est ajoutée à l'historique ;
        6. elle peut donc servir à prévoir le mois suivant.

    Le système ne suppose aucune année future prédéfinie.
    La prévision commence toujours après la dernière date
    réellement présente dans le fichier chargé.
    """

    # --------------------------------------------------------
    # Vérifications
    # --------------------------------------------------------

    if df_ml.empty:
        raise ValueError(
            "Aucune donnée disponible pour effectuer la prévision."
        )

    if horizon < 1:
        raise ValueError(
            "L'horizon de prévision doit être supérieur ou égal à 1."
        )

    colonnes_necessaires = ["date", "volume"] + FEATURES

    colonnes_manquantes = [
        colonne
        for colonne in colonnes_necessaires
        if colonne not in df_ml.columns
    ]

    if colonnes_manquantes:
        raise ValueError(
            "Colonnes manquantes pour la prévision : "
            f"{colonnes_manquantes}"
        )

    # --------------------------------------------------------
    # Historique des volumes
    # --------------------------------------------------------

    historique = df_ml[
        ["date", "volume"]
    ].copy()

    historique = historique.sort_values("date").reset_index(drop=True)

    # Liste contenant les volumes réels puis, progressivement,
    # les volumes prédits.
    volumes = historique["volume"].astype(float).tolist()

    derniere_date = historique["date"].iloc[-1]

    # --------------------------------------------------------
    # Prévisions futures
    # --------------------------------------------------------

    resultats = []

    for horizon_actuel in range(1, horizon + 1):

        # ----------------------------------------------------
        # Date suivante
        # ----------------------------------------------------

        nouvelle_date = (
            derniere_date
            + pd.DateOffset(months=1)
        )

        # ----------------------------------------------------
        # Variables calendaires
        # ----------------------------------------------------

        mois = nouvelle_date.month

        trimestre = nouvelle_date.quarter

        saison = int(mois in [6, 7, 8])

        # ----------------------------------------------------
        # Variables retardées
        # ----------------------------------------------------

        if len(volumes) < 2:
            raise ValueError(
                "Historique insuffisant pour calculer les variables "
                "lag_1 et lag_2."
            )

        lag_1 = volumes[-1]

        lag_2 = volumes[-2]

        # ----------------------------------------------------
        # Moyenne mobile sur les trois dernières valeurs
        # ----------------------------------------------------

        if len(volumes) < 3:
            raise ValueError(
                "Historique insuffisant pour calculer "
                "la moyenne mobile sur trois périodes."
            )

        moyenne_mobile_3 = sum(
            volumes[-3:]
        ) / 3

        # ----------------------------------------------------
        # Construction de la ligne à prédire
        # ----------------------------------------------------

        ligne_prediction = pd.DataFrame(
            [{
                "mois": mois,
                "trimestre": trimestre,
                "saison": saison,
                "lag_1": lag_1,
                "lag_2": lag_2,
                "moyenne_mobile_3": moyenne_mobile_3,
            }]
        )

        # ----------------------------------------------------
        # Prédiction XGBoost
        # ----------------------------------------------------

        prediction = float(
            modele.predict(
                ligne_prediction[FEATURES]
            )[0]
        )

        # ----------------------------------------------------
        # Protection contre une prévision négative
        # ----------------------------------------------------

        prediction = max(0.0, prediction)

        # ----------------------------------------------------
        # Enregistrement
        # ----------------------------------------------------

        resultats.append(
            {
                "date": nouvelle_date,
                "volume_pred": prediction,
                "horizon": horizon_actuel,
            }
        )

        # ----------------------------------------------------
        # Mise à jour de l'historique
        #
        # La prévision devient une donnée disponible pour
        # calculer les variables du mois suivant.
        # ----------------------------------------------------

        volumes.append(prediction)

        derniere_date = nouvelle_date

    return pd.DataFrame(resultats)


# ============================================================
# FONCTION PRINCIPALE
# ============================================================

def executer_prevision(
    chemin_fichier,
    horizon: int = HORIZON_PREVISION,
):
    """
    Lance automatiquement :

        fichier
        ↓
        préparation des données
        ↓
        entraînement XGBoost
        ↓
        prévision récursive
    """


    # --------------------------------------------------------
    # 2. Entraînement du modèle final
    # --------------------------------------------------------

    modele, df_ml, df_resultats_historiques = (
        entrainer_xgboost(
            chemin_fichier
        )
    )

    # --------------------------------------------------------
    # 3. Prévision future
    # --------------------------------------------------------

    previsions = prevoir_futur(
        modele=modele,
        df_ml=df_ml,
        horizon=horizon,
    )

    # --------------------------------------------------------
    # 4. Affichage
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("AVI CENTER — PRÉVISION DYNAMIQUE XGBOOST")
    print("=" * 70)

    print(
        f"\nDernière observation réelle : "
        f"{df_ml['date'].max():%Y-%m}"
    )

    print(
        f"Nombre de mois prévus : "
        f"{len(previsions)}"
    )

    print(
        f"Première période prévue : "
        f"{previsions['date'].min():%Y-%m}"
    )

    print(
        f"Dernière période prévue : "
        f"{previsions['date'].max():%Y-%m}"
    )

    print("\nPrévisions futures :")

    tableau = previsions.copy()

    tableau["date"] = tableau["date"].dt.strftime(
        "%Y-%m"
    )

    tableau["volume_pred"] = tableau[
        "volume_pred"
    ].round(2)

    print(
        tableau.to_string(index=False)
    )

    print("\nPrévision dynamique terminée.")

    return modele, df_ml, previsions


# ============================================================
# TEST DIRECT
# ============================================================

if __name__ == "__main__":

    fichier = "Base.xlsx"

    try:

        modele, df_ml, previsions = executer_prevision(
            chemin_fichier=fichier,
            horizon=12,
        )

    except Exception as erreur:

        print("\n")
        print("=" * 70)
        print("ERREUR")
        print("=" * 70)
        print(erreur)
        raise