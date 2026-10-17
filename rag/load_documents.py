import torch

import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

from rag.pdf_metadata import use_one_based_page_number


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_FOLDER = PROJECT_ROOT / "data" / "guidelines"


def _extract_year(filename: str):
    match = re.search(r"(?<!\d)(20\d{2})(?!\d)", filename)
    return int(match.group(1)) if match else -1


def _extract_organization(filename: str):
    name = filename.lower()

    if "esc" in name:
        return "ESC"
    if "aha" in name:
        return "AHA"
    if "acc" in name:
        return "ACC"

    return "UNKNOWN"


def load_pdfs():
    """
    Charge tous les PDF présents dans data/guidelines.

    Chaque page devient un Document LangChain avec des métadonnées
    permettant de tracer précisément sa provenance.
    """

    if not PDF_FOLDER.exists():
        raise FileNotFoundError(
            f"Dossier guidelines introuvable : {PDF_FOLDER}"
        )

    documents = []
    pdf_files = sorted(PDF_FOLDER.glob("*.pdf"))

    if not pdf_files:
        raise RuntimeError(
            f"Aucun PDF trouvé dans : {PDF_FOLDER}"
        )

    print(f"Dossier guidelines : {PDF_FOLDER}")
    print(f"Nombre de PDF trouvés : {len(pdf_files)}")

    for file_path in pdf_files:
        print(f"Chargement du PDF : {file_path.name}")

        loader = PyPDFLoader(str(file_path))
        pages = loader.load()

        organization = _extract_organization(file_path.name)
        year = _extract_year(file_path.name)

        for page in pages:
            use_one_based_page_number(page.metadata)
            page.metadata["source"] = file_path.name
            page.metadata["source_type"] = "guideline"
            page.metadata["organization"] = organization
            page.metadata["year"] = year

        documents.extend(pages)

    return documents


if __name__ == "__main__":
    docs = load_pdfs()

    print(f"\nNombre total de pages chargées : {len(docs)}")

    if docs:
        print("\nExemple de métadonnées :")
        print(docs[0].metadata)
