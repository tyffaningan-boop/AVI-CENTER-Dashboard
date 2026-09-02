from __future__ import annotations

import pandas as pd
from xgboost import XGBRegressor

from preparation_donnees import preparer_donnees_ml


# ============================================================
# PARAMÈTRES DU MODÈLE RETENU
# ============================================================

FEATURES = [
    "mois",
    "trimestre",
    "saison",
    "lag_1",
    "lag_2",
    "moyenne_mobile_3",
]

TARGET = "volume"


def entrainer_xgboost(chemin_fichier):
    """
    Prépare les données et entraîne directement XGBoost.

    Aucun autre modèle n'est entraîné ou comparé.
    """

    # --------------------------------------------------------
    # 1. Préparation des données
    # --------------------------------------------------------

    df_ml = preparer_donnees_ml(chemin_fichier)

    # --------------------------------------------------------
    # 2. Vérification des variables
    # --------------------------------------------------------

    colonnes_manquantes = [
        colonne
        for colonne in FEATURES + [TARGET]
        if colonne not in df_ml.columns
    ]

    if colonnes_manquantes:
        raise ValueError(
            "Variables nécessaires absentes : "
            f"{colonnes_manquantes}"
        )

    # --------------------------------------------------------
    # 3. Construction de X et y
    # --------------------------------------------------------

    X = df_ml[FEATURES].copy()
    y = df_ml[TARGET].copy()

    # --------------------------------------------------------
    # 4. Entraînement de XGBoost
    # --------------------------------------------------------

    modele = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        objective="reg:squarederror",
    )

    modele.fit(X, y)

    # --------------------------------------------------------
    # 5. Prédictions sur les données historiques
    # --------------------------------------------------------

    df_resultats = df_ml[
        ["date", "volume"]
    ].copy()

    df_resultats["volume_pred"] = modele.predict(X)

    # --------------------------------------------------------
    # 6. Résumé
    # --------------------------------------------------------

    print("=" * 70)
    print("AVI CENTER — ENTRAÎNEMENT XGBOOST")
    print("=" * 70)

    print(
        f"Période d'entraînement : "
        f"{df_ml['date'].min():%Y-%m} → "
        f"{df_ml['date'].max():%Y-%m}"
    )

    print(f"Observations utilisées : {len(df_ml)}")

    print("\nVariables explicatives :")
    for variable in FEATURES:
        print(f" - {variable}")

    print("\nVariable cible :")
    print(f" - {TARGET}")

    print("\nParamètres XGBoost :")
    print(" - n_estimators = 100")
    print(" - max_depth = 3")
    print(" - learning_rate = 0.1")
    print(" - random_state = 42")

    print("\nEntraînement terminé avec succès.")

    print("\nAperçu des résultats :")
    print(
        df_resultats
        .head(10)
        .to_string(index=False)
    )

    return modele, df_ml, df_resultats


# ============================================================
# TEST DIRECT
# ============================================================

if __name__ == "__main__":

    fichier = "Base.xlsx"

    try:

        modele, df_ml, df_resultats = entrainer_xgboost(
            fichier
        )

        print("\nModèle XGBoost prêt.")

    except Exception as erreur:

        print("\nERREUR :")
        print(erreur)
        raise