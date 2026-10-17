import re
import unicodedata

from rag_ecg_reference.embeddings import load_database


EXPANSIONS = (
    {
        "aliases": (
            "fibrillation atriale",
            "atrial fibrillation",
            "afib",
            "rythme irregulier",
            "rr irreguliers",
            "absence d ondes p",
        ),
        "keywords": (
            "atrial fibrillation",
            "irregular rr intervals",
            "absence of p waves",
            "irregular rhythm",
        ),
    },
    {
        "aliases": (
            "sus decalage st",
            "sus-decalage st",
            "st elevation",
            "v2 v3 v4",
        ),
        "keywords": (
            "st segment elevation",
            "st elevation",
            "anterior leads",
            "v2 v3 v4",
            "myocardial infarction ecg",
        ),
    },
    {
        "aliases": (
            "qualite ecg",
            "qualité ecg",
            "artefact",
            "calibration",
            "electrode",
            "électrode",
        ),
        "keywords": (
            "ecg quality",
            "electrode placement",
            "calibration",
            "paper speed",
            "artifact",
            "lead consistency",
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


def _bonus(doc_text, question):
    score = 0
    normalized_doc = normalize_text(doc_text)

    for group in EXPANSIONS:
        if any(contains(question, alias) for alias in group["aliases"]):
            for keyword in group["keywords"]:
                if normalize_text(keyword) in normalized_doc:
                    score += 3 if " " in keyword else 1

    return score


def retrieve_ecg_reference(question, k=5, fetch_k=20):
    db = load_database()
    query = build_query(question)

    docs = db.similarity_search(
        query,
        k=max(fetch_k, k),
        filter={"source_type": "ecg_reference"},
    )

    ranked = sorted(
        enumerate(docs),
        key=lambda item: (
            _bonus(item[1].page_content, question),
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

        passage = f"""
PASSAGE ECG REFERENCE {index}

Source : {source}
Page : {page}

{doc.page_content}
"""
        passages.append(passage.strip())

    return "\n\n".join(passages)


if __name__ == "__main__":
    question = (
        "Femme de 68 ans : sus-décalage ST de 3 mm en V2-V3-V4. "
        "Quels éléments ECG permettent l'interprétation ?"
    )
    docs = retrieve_ecg_reference(question, k=5)
    print(build_context(docs))
