from __future__ import annotations

import pandas as pd

from lecteur_universel import normaliser_donnees


def preparer_donnees_ml(chemin_fichier):
    """
    Prépare les données pour le modèle XGBoost.

    Cette fonction reproduit le feature engineering
    utilisé dans le travail de modélisation.
    """

    # ---------------------------------------------------------
    # 1. Lecture et normalisation
    # ---------------------------------------------------------

    df = normaliser_donnees(chemin_fichier).copy()

    # ---------------------------------------------------------
    # 2. Variables calendaires
    # ---------------------------------------------------------

    df["mois"] = df["date"].dt.month

    df["trimestre"] = df["date"].dt.quarter

    # Saison définie dans le travail initial :
    # juin, juillet et août = période saisonnière
    df["saison"] = df["mois"].isin([6, 7, 8]).astype(int)

    # ---------------------------------------------------------
    # 3. Variables retardées
    # ---------------------------------------------------------

    df["lag_1"] = df["volume"].shift(1)

    df["lag_2"] = df["volume"].shift(2)

    # ---------------------------------------------------------
    # 4. Moyenne mobile
    # ---------------------------------------------------------

    df["moyenne_mobile_3"] = (
        df["volume"]
        .rolling(window=3)
        .mean()
    )

    # ---------------------------------------------------------
    # 5. Suppression des lignes impossibles à utiliser
    # ---------------------------------------------------------

    df_ml = df.dropna(
        subset=[
            "lag_1",
            "lag_2",
            "moyenne_mobile_3",
        ]
    ).copy()

    df_ml.reset_index(drop=True, inplace=True)

    # ---------------------------------------------------------
    # 6. Vérification
    # ---------------------------------------------------------

    colonnes_attendues = [
        "date",
        "Années",
        "Périodes",
        "volume",
        "mois",
        "trimestre",
        "saison",
        "lag_1",
        "lag_2",
        "moyenne_mobile_3",
    ]

    colonnes_manquantes = [
        colonne
        for colonne in colonnes_attendues
        if colonne not in df_ml.columns
    ]

    if colonnes_manquantes:
        raise ValueError(
            "Variables manquantes après préparation : "
            f"{colonnes_manquantes}"
        )

    return df_ml[colonnes_attendues]


# =============================================================
# TEST DIRECT
# =============================================================

if __name__ == "__main__":

    fichier = "Base.xlsx"

    try:

        df_ml = preparer_donnees_ml(fichier)

        print("=" * 70)
        print("AVI CENTER — PRÉPARATION DES DONNÉES ML")
        print("=" * 70)

        print(
            f"Première période ML : "
            f"{df_ml['date'].min():%Y-%m}"
        )

        print(
            f"Dernière période ML : "
            f"{df_ml['date'].max():%Y-%m}"
        )

        print(f"Observations ML : {len(df_ml)}")

        print("\nVariables utilisées :")
        print(df_ml.columns.tolist())

        print("\nAperçu des données préparées :")
        print(df_ml.head(10).to_string(index=False))

        print("\nValeurs manquantes :")
        print(df_ml.isna().sum())

        print("\nPréparation validée.")

    except Exception as erreur:

        print("\nERREUR :")
        print(erreur)