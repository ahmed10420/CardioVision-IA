import torch

import sys

from rag.chunking import split_documents
from rag_research.load_documents import load_research_pdfs
from rag_research.embeddings import create_database
from rag_research.retriever import retrieve_research, build_context


def build():
    documents = load_research_pdfs()

    print("\nNombre total de pages :", len(documents))

    chunks = split_documents(documents)

    print("Nombre de chunks utiles :", len(chunks))

    create_database(chunks, reset=True)


def test():
    question = (
        "Quelles méthodes de deep learning sont utilisées avec "
        "PTB-XL pour la classification ECG ?"
    )

    documents = retrieve_research(question, k=5)

    print("\nQUESTION :")
    print(question)
    print("\nRÉSULTATS :")
    print(build_context(documents))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    build()
    test()
