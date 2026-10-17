import torch

import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_FOLDER = PROJECT_ROOT / "chroma_research"
COLLECTION_NAME = "research"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )


def reset_database():
    if DB_FOLDER.exists():
        print(f"Suppression ancienne base : {DB_FOLDER}")
        shutil.rmtree(DB_FOLDER)


def create_database(chunks, reset=True):
    if not chunks:
        raise ValueError("Aucun chunk à indexer.")

    if reset:
        reset_database()

    embeddings = get_embedding_model()

    print(f"Création Chroma : {DB_FOLDER}")
    print(f"Collection : {COLLECTION_NAME}")
    print(f"Nombre de chunks : {len(chunks)}")

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(DB_FOLDER),
    )

    print("Base research créée avec succès.")

    return db


def load_database():
    if not DB_FOLDER.exists():
        raise FileNotFoundError(
            f"Base introuvable : {DB_FOLDER}"
        )

    embeddings = get_embedding_model()

    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(DB_FOLDER),
        embedding_function=embeddings,
    )
