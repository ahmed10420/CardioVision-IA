import torch

import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

from rag.pdf_metadata import use_one_based_page_number


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_FOLDER = PROJECT_ROOT / "data" / "research_papers"


def _extract_year(filename: str):
    match = re.search(
        r"(?<!\d)(20\d{2})(?!\d)",
        filename,
    )

    return int(match.group(1)) if match else -1


def _detect_topic(filename: str):
    name = filename.lower()

    if "ptb-xl" in name or "ptbxl" in name:
        return "ptbxl"

    if "digitiz" in name or "digitis" in name or "image" in name:
        return "ecg_digitization"

    if (
        "deep" in name
        or "cnn" in name
        or "transformer" in name
        or "classification" in name
    ):
        return "ecg_ai"

    return "ecg_research"


def load_research_pdfs():
    if not PDF_FOLDER.exists():
        raise FileNotFoundError(
            f"Dossier introuvable : {PDF_FOLDER}"
        )

    pdf_files = sorted(PDF_FOLDER.glob("*.pdf"))

    if not pdf_files:
        raise RuntimeError(
            f"Aucun PDF trouvé dans : {PDF_FOLDER}"
        )

    print(f"Dossier research papers : {PDF_FOLDER}")
    print(f"Nombre de PDF trouvés : {len(pdf_files)}")

    documents = []

    for file_path in pdf_files:
        print(f"Chargement : {file_path.name}")

        loader = PyPDFLoader(str(file_path))
        pages = loader.load()

        year = _extract_year(file_path.name)
        topic = _detect_topic(file_path.name)

        for page in pages:
            use_one_based_page_number(page.metadata)
            page.metadata["source"] = file_path.name
            page.metadata["source_type"] = "research"
            page.metadata["topic"] = topic
            page.metadata["year"] = year

        documents.extend(pages)

    return documents


if __name__ == "__main__":
    docs = load_research_pdfs()

    print("\nNombre total de pages :", len(docs))

    if docs:
        print("\nMétadonnées exemple :")
        print(docs[0].metadata)
