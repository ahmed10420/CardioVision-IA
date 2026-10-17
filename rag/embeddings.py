import torch

import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DB_FOLDER = PROJECT_ROOT / "chroma_guidelines"
COLLECTION_NAME = "guidelines"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model():
    """
    Charge le modèle d'embedding partagé par l'indexation et le retrieval.
    """

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )


def reset_vector_database():
    """
    Supprime complètement l'ancien index guidelines.

    C'est important quand on ajoute/supprime des PDF afin d'éviter :
    - les doublons ;
    - les anciens chunks encore présents ;
    - les métadonnées obsolètes.
    """

    if DB_FOLDER.exists():
        print(f"Suppression de l'ancienne base : {DB_FOLDER}")
        shutil.rmtree(DB_FOLDER)


def create_vector_database(chunks, reset=True):
    """
    Reconstruit ChromaDB à partir de tous les chunks guidelines.
    """

    if not chunks:
        raise ValueError("Aucun chunk à indexer.")

    if reset:
        reset_vector_database()

    embeddings = get_embedding_model()

    print(f"Création de Chroma : {DB_FOLDER}")
    print(f"Collection : {COLLECTION_NAME}")
    print(f"Nombre de chunks à indexer : {len(chunks)}")

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(DB_FOLDER),
    )

    print("Base Chroma guidelines créée avec succès.")

    return db


def load_vector_database():
    """
    Recharge la base guidelines déjà construite.
    """

    if not DB_FOLDER.exists():
        raise FileNotFoundError(
            f"La base guidelines n'existe pas : {DB_FOLDER}\n"
            "Exécute d'abord : python -m rag.test_rag"
        )

    embeddings = get_embedding_model()

    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(DB_FOLDER),
        embedding_function=embeddings,
    )


if __name__ == "__main__":
    print("DB_FOLDER =", DB_FOLDER)
    print("EMBEDDING_MODEL =", EMBEDDING_MODEL)
