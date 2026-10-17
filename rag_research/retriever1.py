import sys

from rag_research.embeddings import load_database


def retrieve_research(question, k=5, topic=None):
    db = load_database()

    search_filter = {
        "source_type": "research",
    }

    if topic:
        search_filter["topic"] = topic

    return db.similarity_search(
        question,
        k=k,
        filter=search_filter,
    )


def build_context(documents):
    passages = []

    for index, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "inconnue")
        page = doc.metadata.get("page", "N/A")
        year = doc.metadata.get("year", -1)
        topic = doc.metadata.get("topic", "ecg_research")

        passage = f"""
PASSAGE RESEARCH {index}

Source : {source}
Page : {page}
Année : {year}
Topic : {topic}

{doc.page_content}
"""

        passages.append(passage.strip())

    return "\n\n".join(passages)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    question = (
        "Quelles méthodes sont utilisées pour la classification "
        "automatique des ECG avec PTB-XL ?"
    )

    docs = retrieve_research(question, k=5)
    print(build_context(docs))
