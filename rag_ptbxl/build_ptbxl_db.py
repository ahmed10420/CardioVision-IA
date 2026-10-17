import ast
from pathlib import Path

import pandas as pd

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

PTBXL_DIR = PROJECT_DIR / "data" / "ptb-xl"

DATABASE_CSV = PTBXL_DIR / "ptbxl_database.csv"
SCP_CSV = PTBXL_DIR / "scp_statements.csv"

CHROMA_DIR = PROJECT_DIR / "chroma_ptbxl"

COLLECTION_NAME = "ptbxl_cases"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# EMBEDDINGS
# ============================================================

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={
            "normalize_embeddings": True
        },
    )


# ============================================================
# CHARGEMENT SCP STATEMENTS
# ============================================================

def load_scp_statements():
    if not SCP_CSV.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {SCP_CSV}"
        )

    scp = pd.read_csv(
        SCP_CSV,
        index_col=0,
    )

    return scp


# ============================================================
# CONVERSION DES CODES SCP EN SUPERCLASSES
# ============================================================

def aggregate_diagnostic_codes(
    scp_codes,
    scp_df,
):
    """
    Convertit les codes SCP diagnostiques en superclasses PTB-XL.

    Exemples possibles :
    NORM, MI, STTC, CD, HYP
    """

    superclasses = set()

    for code in scp_codes.keys():

        if code not in scp_df.index:
            continue

        row = scp_df.loc[code]

        diagnostic = row.get(
            "diagnostic",
            0,
        )

        if diagnostic != 1:
            continue

        diagnostic_class = row.get(
            "diagnostic_class"
        )

        if pd.notna(diagnostic_class):
            superclasses.add(
                str(diagnostic_class)
            )

    return sorted(superclasses)


# ============================================================
# CONSTRUCTION DU TEXTE D'UN CAS
# ============================================================

def build_case_text(
    row,
    scp_codes,
    superclasses,
):
    ecg_id = row.get(
        "ecg_id",
        "inconnu",
    )

    age = row.get(
        "age",
        None,
    )

    sex = row.get(
        "sex",
        None,
    )

    report = row.get(
        "report",
        "",
    )

    heart_axis = row.get(
        "heart_axis",
        "",
    )

    infarction_stadium1 = row.get(
        "infarction_stadium1",
        "",
    )

    rhythm = row.get(
        "rhythm",
        "",
    )

    filename_lr = row.get(
        "filename_lr",
        "",
    )

    filename_hr = row.get(
        "filename_hr",
        "",
    )

    sex_text = "inconnu"

    if pd.notna(sex):
        try:
            sex_value = int(sex)

            if sex_value == 0:
                sex_text = "homme"
            elif sex_value == 1:
                sex_text = "femme"
            else:
                sex_text = str(sex_value)

        except Exception:
            sex_text = str(sex)

    age_text = (
        str(age)
        if pd.notna(age)
        else "inconnu"
    )

    report_text = (
        str(report).strip()
        if pd.notna(report)
        and str(report).strip()
        else "non disponible"
    )

    scp_text = ", ".join(
        sorted(scp_codes.keys())
    )

    superclass_text = (
        ", ".join(superclasses)
        if superclasses
        else "non disponible"
    )

    parts = [
        f"ECG ID : {ecg_id}",
        f"Âge : {age_text}",
        f"Sexe : {sex_text}",
        f"Rapport clinique : {report_text}",
        f"Codes SCP : {scp_text}",
        f"Superclasses diagnostiques : {superclass_text}",
    ]

    if pd.notna(rhythm) and str(rhythm).strip():
        parts.append(
            f"Rythme : {rhythm}"
        )

    if (
        pd.notna(heart_axis)
        and str(heart_axis).strip()
    ):
        parts.append(
            f"Axe cardiaque : {heart_axis}"
        )

    if (
        pd.notna(infarction_stadium1)
        and str(infarction_stadium1).strip()
    ):
        parts.append(
            "Stade d'infarctus : "
            f"{infarction_stadium1}"
        )

    if (
        pd.notna(filename_lr)
        and str(filename_lr).strip()
    ):
        parts.append(
            f"Signal 100 Hz : {filename_lr}"
        )

    if (
        pd.notna(filename_hr)
        and str(filename_hr).strip()
    ):
        parts.append(
            f"Signal 500 Hz : {filename_hr}"
        )

    return "\n".join(parts)


# ============================================================
# CONSTRUCTION DES DOCUMENTS LANGCHAIN
# ============================================================

def build_documents(
    limit=None,
):
    if not DATABASE_CSV.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {DATABASE_CSV}"
        )

    database = pd.read_csv(
        DATABASE_CSV
    )

    scp_df = load_scp_statements()

    documents = []

    for index, row in database.iterrows():

        raw_codes = row.get(
            "scp_codes",
            "{}",
        )

        try:
            scp_codes = ast.literal_eval(
                raw_codes
            )

        except Exception:
            scp_codes = {}

        if not isinstance(
            scp_codes,
            dict,
        ):
            scp_codes = {}

        superclasses = (
            aggregate_diagnostic_codes(
                scp_codes,
                scp_df,
            )
        )

        text = build_case_text(
            row=row,
            scp_codes=scp_codes,
            superclasses=superclasses,
        )

        metadata = {
            "source": "PTB-XL",
            "ecg_id": int(
                row.get(
                    "ecg_id",
                    index,
                )
            ),
            "age": (
                float(row["age"])
                if pd.notna(
                    row.get("age")
                )
                else -1
            ),
            "sex": (
                int(row["sex"])
                if pd.notna(
                    row.get("sex")
                )
                else -1
            ),
            "superclasses": (
                "|".join(superclasses)
            ),
            "filename_lr": str(
                row.get(
                    "filename_lr",
                    "",
                )
            ),
            "filename_hr": str(
                row.get(
                    "filename_hr",
                    "",
                )
            ),
        }

        documents.append(
            Document(
                page_content=text,
                metadata=metadata,
            )
        )

        if (
            limit is not None
            and len(documents) >= limit
        ):
            break

    return documents


# ============================================================
# CRÉATION CHROMA
# ============================================================

def build_ptbxl_chroma(
    limit=None,
):
    print(
        "\nConstruction de la base PTB-XL..."
    )

    print(
        "Dataset :",
        DATABASE_CSV,
    )

    print(
        "Sortie Chroma :",
        CHROMA_DIR,
    )

    documents = build_documents(
        limit=limit
    )

    print(
        "Nombre de cas préparés :",
        len(documents),
    )

    embeddings = get_embeddings()

    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(
            CHROMA_DIR
        ),
    )

    print(
        "Base PTB-XL créée avec succès."
    )

    return db


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

   

    build_ptbxl_chroma(
        limit=None 
    )