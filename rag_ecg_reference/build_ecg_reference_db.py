import torch

from rag.chunking import split_documents
from rag_ecg_reference.load_documents import load_ecg_reference_pdfs
from rag_ecg_reference.embeddings import create_database
from rag_ecg_reference.retriever import retrieve_ecg_reference, build_context


def build():
    documents = load_ecg_reference_pdfs()

    print("\nNombre total de pages :", len(documents))

    chunks = split_documents(documents)

    print("Nombre de chunks utiles :", len(chunks))

    create_database(chunks, reset=True)


def test():
    question = (
        "Quelles sont les caractéristiques normales de l'onde P, "
        "du complexe QRS et de l'onde T ?"
    )

    documents = retrieve_ecg_reference(question, k=5)

    print("\nQUESTION :")
    print(question)
    print("\nRÉSULTATS :")
    print(build_context(documents))


if __name__ == "__main__":
    build()
    test()
