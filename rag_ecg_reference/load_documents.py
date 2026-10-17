import torch

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

from rag.pdf_metadata import use_one_based_page_number


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PDF_FOLDER = PROJECT_ROOT / "data" / "ecg_reference"


def load_ecg_reference_pdfs():
    if not PDF_FOLDER.exists():
        raise FileNotFoundError(
            f"Dossier introuvable : {PDF_FOLDER}"
        )

    pdf_files = sorted(PDF_FOLDER.glob("*.pdf"))

    if not pdf_files:
        raise RuntimeError(
            f"Aucun PDF trouvé dans : {PDF_FOLDER}"
        )

    print(f"Dossier ECG reference : {PDF_FOLDER}")
    print(f"Nombre de PDF trouvés : {len(pdf_files)}")

    documents = []

    for file_path in pdf_files:
        print(f"Chargement : {file_path.name}")

        loader = PyPDFLoader(str(file_path))
        pages = loader.load()

        for page in pages:
            use_one_based_page_number(page.metadata)
            page.metadata["source"] = file_path.name
            page.metadata["source_type"] = "ecg_reference"
            page.metadata["category"] = "ECG technical reference"

        documents.extend(pages)

    return documents
