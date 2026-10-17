from rag.load_documents import load_pdfs
from rag.chunking import split_documents
from rag.embeddings import create_vector_database
from rag.retriever import (
    retrieve_documents,
    display_results,
)


def build_index():
    """
    Reconstruit complètement l'index guidelines.

    À exécuter après chaque ajout, suppression ou remplacement
    d'un PDF dans data/guidelines.
    """

    documents = load_pdfs()

    print(
        f"\nNombre total de pages chargées : "
        f"{len(documents)}"
    )

    chunks = split_documents(
        documents
    )

    create_vector_database(
        chunks,
        reset=True,
    )


def test_question():
    """
    Test simple après reconstruction.
    """

    question = (
        "Quels sont les critères ECG de diagnostic "
        "de la fibrillation atriale ?"
    )

    results = retrieve_documents(
        question,
        k=5,
    )

    display_results(
        question,
        results,
    )


if __name__ == "__main__":
    build_index()
    test_question()
