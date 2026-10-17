import re
import sys
import unicodedata

from rag_research.embeddings import load_database


EXPANSIONS = (
    {
        "aliases": (
            "fibrillation atriale",
            "atrial fibrillation",
            "afib",
        ),
        "keywords": (
            "atrial fibrillation detection",
            "ecg classification",
            "deep learning",
            "machine learning",
        ),
    },
    {
        "aliases": (
            "stemi",
            "sus decalage st",
            "sus-decalage st",
            "st elevation",
            "infarctus",
        ),
        "keywords": (
            "stemi detection",
            "st segment elevation",
            "myocardial infarction",
            "ecg classification",
        ),
    },
    {
        "aliases": (
            "ptb xl",
            "ptb-xl",
        ),
        "keywords": (
            "ptb-xl",
            "ecg benchmark",
            "ecg classification",
        ),
    },
)


def normalize_text(text):
    text = unicodedata.normalize("NFKD", str(text).lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def contains(text, term):
    return normalize_text(term) in normalize_text(text)


def build_query(question):
    extra = set()
    for group in EXPANSIONS:
        if any(contains(question, alias) for alias in group["aliases"]):
            extra.update(group["keywords"])
    return " ".join([question, *sorted(extra)]).strip()


def _bonus(doc, question):
    text = normalize_text(doc.page_content)
    score = 0

    for group in EXPANSIONS:
        if any(contains(question, alias) for alias in group["aliases"]):
            for keyword in group["keywords"]:
                if normalize_text(keyword) in text:
                    score += 3 if " " in keyword else 1

    # Small recency bonus, used only as a tiebreaker.
    try:
        year = int(doc.metadata.get("year", -1))
        if year >= 2024:
            score += 2
        elif year >= 2020:
            score += 1
    except (TypeError, ValueError):
        pass

    return score


def retrieve_research(question, k=5, topic=None, fetch_k=20):
    db = load_database()

    search_filter = {"source_type": "research"}
    if topic:
        search_filter["topic"] = topic

    query = build_query(question)

    docs = db.similarity_search(
        query,
        k=max(fetch_k, k),
        filter=search_filter,
    )

    ranked = sorted(
        enumerate(docs),
        key=lambda item: (
            _bonus(item[1], question),
            -item[0],
        ),
        reverse=True,
    )

    return [doc for _, doc in ranked[:k]]


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
