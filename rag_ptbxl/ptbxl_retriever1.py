import torch

from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from rag_ptbxl.build_ptbxl_db import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)


_CROSS_ENCODER = None


def get_cross_encoder():
    """Charge une seule fois le modèle de reranking sur CPU."""
    global _CROSS_ENCODER

    if _CROSS_ENCODER is None:
        _CROSS_ENCODER = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            device="cpu",
        )

    return _CROSS_ENCODER


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
# CHARGEMENT DE LA BASE
# ============================================================

def load_ptbxl_db():

    if not Path(
        CHROMA_DIR
    ).exists():
        raise FileNotFoundError(
            "La base Chroma PTB-XL n'existe pas.\n"
            "Exécute d'abord :\n"
            "python -m rag_ptbxl.build_ptbxl_db"
        )

    embeddings = get_embeddings()

    db = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(
            CHROMA_DIR
        ),
        embedding_function=embeddings,
    )

    return db


# ============================================================
# RECHERCHE DE CAS SIMILAIRES
# ============================================================

def retrieve_ptbxl_cases(
    query: str,
    k: int = 5,
) -> List:
    """Recherche vectorielle simple, conservée pour compatibilité."""
    db = load_ptbxl_db()
    return db.similarity_search(query, k=k)


# ============================================================
# FORMATAGE POUR LES AGENTS
# ============================================================

def format_ptbxl_cases(
    documents,
):
    if not documents:
        return (
            "Aucun cas PTB-XL pertinent "
            "n'a été retrouvé."
        )

    formatted = []

    for index, doc in enumerate(
        documents,
        start=1,
    ):

        metadata = doc.metadata

        ecg_id = metadata.get(
            "ecg_id",
            "inconnu",
        )

        superclasses = metadata.get(
            "superclasses",
            "",
        )

        reranker_score = metadata.get("reranker_score")
        reranker_line = (
            f"\nScore CrossEncoder : {reranker_score:.4f}"
            if isinstance(reranker_score, (int, float))
            else ""
        )

        formatted.append(
            f"""
[CAS PTB-XL {index}]
ECG ID : {ecg_id}
Superclasses : {superclasses}
{reranker_line}

{doc.page_content}
""".strip()
        )

    return "\n\n".join(
        formatted
    )


# ============================================================
# FONCTION PRINCIPALE
# ============================================================

def search_ptbxl(
    query,
    k=3,
    candidate_k=15,
    use_reranker=True,
):
    """
    Recherche PTB-XL en deux étapes : Chroma récupère les candidats,
    puis un CrossEncoder conserve les cas les plus pertinents.

    Le score CrossEncoder mesure la pertinence documentaire. Il ne constitue
    jamais une probabilité diagnostique.
    """
    db = load_ptbxl_db()
    candidate_k = max(k, candidate_k)

    candidate_docs = db.similarity_search(
        query,
        k=candidate_k,
    )

    if not candidate_docs:
        return {
            "query": query,
            "documents": [],
            "context": "",
            "reranking_used": False,
            "candidate_count": 0,
        }

    if use_reranker:
        reranker = get_cross_encoder()
        pairs = [
            [query, doc.page_content]
            for doc in candidate_docs
        ]
        scores = reranker.predict(pairs)

        ranked = sorted(
            zip(candidate_docs, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        documents = []
        for rank, (doc, score) in enumerate(ranked[:k], start=1):
            doc.metadata["reranker_score"] = float(score)
            doc.metadata["reranker_rank"] = rank
            documents.append(doc)
    else:
        documents = candidate_docs[:k]

    context_parts = []
    for index, doc in enumerate(documents, start=1):
        metadata = doc.metadata or {}
        ecg_id = metadata.get("ecg_id", "N/A")
        reranker_score = metadata.get("reranker_score")

        score_text = ""
        if reranker_score is not None:
            score_text = (
                "\nScore de pertinence CrossEncoder : "
                f"{reranker_score:.4f}"
            )

        context_parts.append(
            f"""
CAS PTB-XL {index}

ECG ID : {ecg_id}
{score_text}

{doc.page_content}
""".strip()
        )

    context = "\n\n".join(context_parts)

    return {
        "query": query,
        "documents": documents,
        "context": context,
        "reranking_used": use_reranker,
        "candidate_count": len(candidate_docs),
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    query = (
        "fibrillation atriale avec "
        "rythme irrégulier"
    )

    result = search_ptbxl(
        query=query,
        k=5,
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "CAS PTB-XL RETROUVÉS"
    )

    print(
        "=" * 80
    )

    print(
        result["context"]
    )
