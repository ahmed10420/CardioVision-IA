from rag_ecg_reference.embeddings import load_database


def retrieve_ecg_reference(question, k=5):
    db = load_database()

    return db.similarity_search(
        question,
        k=k,
        filter={
            "source_type": "ecg_reference",
        },
    )


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
        "Quelles sont les caractéristiques ECG "
        "d'une hypertrophie ventriculaire ?"
    )

    docs = retrieve_ecg_reference(question, k=5)
    print(build_context(docs))
