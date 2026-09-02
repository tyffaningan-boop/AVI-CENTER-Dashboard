"""
AVI CENTER — Lecteur universel de données
Étape 1 du pipeline d'automatisation.

Objectif :
- accepter un fichier Excel (.xlsx/.xls) ou CSV (.csv), quel que soit son nom ;
- reconnaître les principales variantes de noms de colonnes ;
- construire une colonne temporelle mensuelle "date" ;
- identifier la variable cible "volume" ;
- retourner un DataFrame standardisé.

Format de sortie garanti :
    date | volume | Années | Périodes

Le module ne réalise PAS encore :
- de feature engineering ;
- d'entraînement XGBoost ;
- de prévision ;
- de calcul de chiffre d'affaires.

Ces étapes seront ajoutées après validation de ce lecteur.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# 1. Normalisation des noms de colonnes
# ---------------------------------------------------------------------

def normaliser_nom(nom) -> str:
    """Transforme un nom de colonne en forme comparable."""
    texte = str(nom).strip().lower()
    texte = unicodedata.normalize("NFKD", texte)
    texte = "".join(c for c in texte if not unicodedata.combining(c))
    texte = re.sub(r"[^a-z0-9]+", "_", texte)
    return texte.strip("_")


def construire_dictionnaire_colonnes(df: pd.DataFrame) -> dict:
    """Associe les noms originaux à leur forme normalisée."""
    return {col: normaliser_nom(col) for col in df.columns}


# ---------------------------------------------------------------------
# 2. Colonnes reconnues
# ---------------------------------------------------------------------

NOMS_VOLUME = {
    "volume",
    "volumes",
    "vente",
    "ventes",
    "dossiers",
    "dossier",
    "quantite",
    "quantite_vendue",
    "nombre_dossiers",
    "nombre_de_dossiers",
    "volume_dossiers",
    "xt",
}

NOMS_DATE = {
    "date",
    "dates",
    "date_vente",
    "date_ventes",
    "date_operation",
    "date_operations",
    "periode_date",
}

NOMS_ANNEE = {
    "annee",
    "annees",
    "annee_vente",
    "annee_des_ventes",
    "year",
    "years",
}

NOMS_MOIS = {
    "mois",
    "mois_vente",
    "month",
    "months",
    "periode",
    "periodes",
    "period",
    "periods",
}


# ---------------------------------------------------------------------
# 3. Recherche d'une colonne
# ---------------------------------------------------------------------

def trouver_colonne(df: pd.DataFrame, candidats: set[str], role: str):
    """
    Recherche une colonne par son nom normalisé.

    On privilégie d'abord une correspondance exacte.
    """
    mapping = construire_dictionnaire_colonnes(df)

    for colonne_originale, colonne_normalisee in mapping.items():
        if colonne_normalisee in candidats:
            return colonne_originale

    raise ValueError(
        f"Impossible d'identifier automatiquement la colonne correspondant "
        f"à '{role}'.\n"
        f"Colonnes disponibles : {list(df.columns)}\n"
        f"Veuillez utiliser un fichier contenant une colonne de {role} "
        f"reconnaissable."
    )


# ---------------------------------------------------------------------
# 4. Lecture des fichiers
# ---------------------------------------------------------------------

def lire_csv(chemin: Path) -> pd.DataFrame:
    """Lecture robuste d'un CSV avec détection du séparateur."""
    erreurs = []

    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(
                chemin,
                sep=None,
                engine="python",
                encoding=encoding,
            )
        except Exception as exc:
            erreurs.append(f"{encoding}: {exc}")

    raise ValueError(
        "Impossible de lire le fichier CSV.\n" + "\n".join(erreurs)
    )


def lire_excel(chemin: Path) -> pd.DataFrame:
    """
    Lit le premier onglet Excel contenant des colonnes compatibles.

    Cela évite de dépendre du nom de la feuille.
    """
    erreurs = []

    # Le contexte garantit la fermeture complète du classeur Excel.
    # C'est important sous Windows : un fichier encore ouvert peut provoquer
    # un PermissionError (WinError 32) lors de la suppression d'un fichier
    # temporaire utilisé par Streamlit.
    try:
        with pd.ExcelFile(chemin) as classeur:
            feuilles = list(classeur.sheet_names)

            for feuille in feuilles:
                try:
                    df = pd.read_excel(classeur, sheet_name=feuille)

                    # Une feuille est considérée comme exploitable si elle possède
                    # au moins une colonne de volume et une structure temporelle.
                    mapping = set(construire_dictionnaire_colonnes(df).values())

                    a_volume = bool(mapping & NOMS_VOLUME)
                    a_date = bool(mapping & NOMS_DATE)
                    a_annee = bool(mapping & NOMS_ANNEE)
                    a_mois = bool(mapping & NOMS_MOIS)

                    if a_volume and (a_date or (a_annee and a_mois)):
                        return df.copy()

                except Exception as exc:
                    erreurs.append(f"Feuille '{feuille}' : {exc}")
    except Exception as exc:
        raise ValueError(f"Impossible d'ouvrir le fichier Excel : {exc}") from exc

    raise ValueError(
        "Aucune feuille Excel ne contient une structure temporelle et "
        "une colonne de volume reconnaissables.\n"
        f"Feuilles examinées : {feuilles if 'feuilles' in locals() else 'inconnues'}\n"
        f"Détails : {erreurs}"
    )


def lire_fichier(chemin_fichier) -> pd.DataFrame:
    """
    Lit automatiquement un fichier Excel ou CSV.

    Le nom du fichier n'a aucune importance.
    """
    chemin = Path(chemin_fichier)

    if not chemin.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {chemin.resolve()}"
        )

    extension = chemin.suffix.lower()

    if extension == ".csv":
        return lire_csv(chemin)

    if extension in {".xlsx", ".xls"}:
        return lire_excel(chemin)

    raise ValueError(
        f"Format non pris en charge : '{extension}'. "
        "Formats acceptés : .xlsx, .xls et .csv."
    )


# ---------------------------------------------------------------------
# 5. Conversion des périodes
# ---------------------------------------------------------------------

MOIS_FR = {
    "janvier": 1,
    "fevrier": 2,
    "mars": 3,
    "avril": 4,
    "mai": 5,
    "juin": 6,
    "juillet": 7,
    "aout": 8,
    "septembre": 9,
    "octobre": 10,
    "novembre": 11,
    "decembre": 12,
}

MOIS_EN = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def convertir_mois(valeur):
    """
    Convertit une période en numéro de mois.

    Accepte :
    - 1 à 12 ;
    - "01", "02", ... ;
    - janvier, février, ... ;
    - January, February, ...
    """
    if pd.isna(valeur):
        return pd.NA

    if isinstance(valeur, (int, float)) and not isinstance(valeur, bool):
        mois = int(valeur)
        return mois if 1 <= mois <= 12 else pd.NA

    texte = str(valeur).strip().lower()
    texte = unicodedata.normalize("NFKD", texte)
    texte = "".join(c for c in texte if not unicodedata.combining(c))

    if texte.isdigit():
        mois = int(texte)
        return mois if 1 <= mois <= 12 else pd.NA

    if texte in MOIS_FR:
        return MOIS_FR[texte]

    if texte in MOIS_EN:
        return MOIS_EN[texte]

    # Accepte aussi les formes abrégées : jan, fev, sept, etc.
    correspondances = {
        "jan": 1, "janv": 1,
        "fev": 2, "fevr": 2,
        "mar": 3,
        "avr": 4,
        "mai": 5,
        "jun": 6,
        "jui": 7, "juil": 7,
        "aou": 8,
        "sep": 9, "sept": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }

    for debut, mois in correspondances.items():
        if texte.startswith(debut):
            return mois

    return pd.NA


# ---------------------------------------------------------------------
# 6. Construction de la date
# ---------------------------------------------------------------------

def construire_date(df: pd.DataFrame) -> pd.Series:
    """
    Construit la date selon deux structures possibles :

    A. une colonne date ;
    B. deux colonnes année + mois/période.
    """
    mapping = construire_dictionnaire_colonnes(df)

    # --- Cas A : une colonne date existe ------------------------------
    colonne_date = None
    for colonne_originale, colonne_normalisee in mapping.items():
        if colonne_normalisee in NOMS_DATE:
            colonne_date = colonne_originale
            break

    if colonne_date is not None:
        dates = pd.to_datetime(
            df[colonne_date],
            errors="coerce",
            dayfirst=True,
        )

        # Si une colonne date existe mais qu'elle est totalement illisible,
        # on ne bascule pas silencieusement vers une autre interprétation.
        if dates.notna().sum() == 0:
            raise ValueError(
                f"La colonne '{colonne_date}' a été identifiée comme date, "
                "mais aucune date n'a pu être interprétée."
            )

        return dates.dt.to_period("M").dt.to_timestamp()

    # --- Cas B : année + mois/période -------------------------------
    colonne_annee = None
    colonne_mois = None

    for colonne_originale, colonne_normalisee in mapping.items():
        if colonne_normalisee in NOMS_ANNEE and colonne_annee is None:
            colonne_annee = colonne_originale

        if colonne_normalisee in NOMS_MOIS and colonne_mois is None:
            colonne_mois = colonne_originale

    if colonne_annee is None or colonne_mois is None:
        raise ValueError(
            "Impossible de construire la date.\n"
            "Le fichier doit contenir soit une colonne 'date', "
            "soit deux colonnes correspondant à l'année et au mois/période."
        )

    annees = pd.to_numeric(df[colonne_annee], errors="coerce")
    mois = df[colonne_mois].apply(convertir_mois)

    dates = pd.to_datetime(
        {
            "year": annees,
            "month": pd.to_numeric(mois, errors="coerce"),
            "day": 1,
        },
        errors="coerce",
    )

    return dates


# ---------------------------------------------------------------------
# 7. Normalisation complète
# ---------------------------------------------------------------------

def normaliser_donnees(chemin_fichier) -> pd.DataFrame:
    """
    Fonction principale du lecteur universel.

    Retourne toujours :
        date | volume | Années | Périodes

    avec :
        - dates mensuelles ;
        - ordre chronologique ;
        - volume numérique ;
        - absence de doublons temporels ;
        - absence de valeurs manquantes.
    """
    df_brut = lire_fichier(chemin_fichier)

    if df_brut.empty:
        raise ValueError("Le fichier importé est vide.")

    # Identifier la colonne volume
    colonne_volume = trouver_colonne(
        df_brut,
        NOMS_VOLUME,
        "volume des ventes/dossiers",
    )

    # Construire la date
    dates = construire_date(df_brut)

    # Construire le tableau standard
    df = pd.DataFrame({
        "date": dates,
        "volume": pd.to_numeric(
            df_brut[colonne_volume],
            errors="coerce",
        ),
    })

    # Contrôle des données non interprétables
    lignes_invalides = df["date"].isna() | df["volume"].isna()

    if lignes_invalides.any():
        nombre = int(lignes_invalides.sum())
        raise ValueError(
            f"{nombre} ligne(s) ne peuvent pas être interprétées "
            "correctement comme date et volume. "
            "Le fichier doit contenir des données temporelles et "
            "des volumes numériques valides."
        )

    # Normalisation mensuelle
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()

    # Vérification des doublons
    doublons = df["date"].duplicated(keep=False)

    if doublons.any():
        dates_dupliquees = (
            df.loc[doublons, "date"]
            .dt.strftime("%Y-%m")
            .unique()
            .tolist()
        )

        raise ValueError(
            "Plusieurs lignes correspondent au même mois : "
            f"{dates_dupliquees}. "
            "Le lecteur ne les additionne pas automatiquement afin "
            "d'éviter de modifier les données originales."
        )

    # Tri chronologique
    df = df.sort_values("date").reset_index(drop=True)

    # Variables calendaires standardisées
    df["Années"] = df["date"].dt.year
    df["Périodes"] = df["date"].dt.month

    # Contrôle de la chronologie
    if not df["date"].is_monotonic_increasing:
        raise ValueError("La chronologie n'a pas pu être ordonnée correctement.")

    return df[["date", "Années", "Périodes", "volume"]]


# ---------------------------------------------------------------------
# 8. Contrôle pratique
# ---------------------------------------------------------------------

def afficher_resume(df: pd.DataFrame) -> None:
    """Affiche un résumé simple pour vérifier le chargement."""
    print("=" * 65)
    print("AVI CENTER — LECTEUR UNIVERSEL")
    print("=" * 65)
    print(f"Première période : {df['date'].min():%Y-%m}")
    print(f"Dernière période  : {df['date'].max():%Y-%m}")
    print(f"Observations      : {len(df)}")
    print(f"Volume total      : {df['volume'].sum():,.0f}")
    print()
    print("Colonnes standardisées :", list(df.columns))
    print()
    print(df.head())
    print()
    print("Lecture validée.")


if __name__ == "__main__":
    # Pour un test manuel :
    # python lecteur_universel.py "Base.xlsx"
    import sys

    if len(sys.argv) != 2:
        print(
            "Utilisation : python lecteur_universel.py "
            '"chemin_vers_fichier.xlsx_ou_csv"'
        )
        raise SystemExit(1)

    fichier = sys.argv[1]

    try:
        donnees = normaliser_donnees(fichier)
        afficher_resume(donnees)
    except Exception as exc:
        print("\nERREUR DE LECTURE :")
        print(exc)
        raise SystemExit(1)
