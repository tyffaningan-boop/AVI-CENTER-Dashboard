from prevision_xgboost import executer_prevision


def executer_pipeline(
    chemin_fichier,
    horizon=12
):
    """
    Pipeline principal AVI Center.

    Fichier Excel/CSV
        ↓
    Préparation des données
        ↓
    Entraînement XGBoost
        ↓
    Prévision future
    """

    print("\n")
    print("=" * 70)
    print("AVI CENTER — PIPELINE AUTOMATIQUE")
    print("=" * 70)

    print(f"\nFichier utilisé : {chemin_fichier}")
    print(f"Horizon de prévision : {horizon} mois")

    # --------------------------------------------------------
    # Exécution du processus complet
    # --------------------------------------------------------

    modele, df_ml, previsions = executer_prevision(
        chemin_fichier=chemin_fichier,
        horizon=horizon,
    )

    print("\n")
    print("=" * 70)
    print("PIPELINE TERMINÉ AVEC SUCCÈS")
    print("=" * 70)

    return {
        "modele": modele,
        "donnees_ml": df_ml,
        "previsions": previsions,
    }


# ============================================================
# TEST DIRECT
# ============================================================

if __name__ == "__main__":

    fichier = "Base.XLSX"

    try:

        resultats = executer_pipeline(
            chemin_fichier=fichier,
            horizon=12,
        )

        print("\nRésultats disponibles :")

        print(
            f"- Observations ML : "
            f"{len(resultats['donnees_ml'])}"
        )

        print(
            f"- Nombre de prévisions : "
            f"{len(resultats['previsions'])}"
        )

        print("\nPrévisions :")

        print(
            resultats["previsions"].to_string(
                index=False
            )
        )

    except Exception as erreur:

        print("\n")
        print("=" * 70)
        print("ERREUR DU PIPELINE")
        print("=" * 70)
        print(erreur)

        raise