import re
import unicodedata

from rag.embeddings import load_vector_database


TOPIC_EXPANSIONS = (
    {
        "aliases": (
            "atrial fibrillation",
            "fibrillation atriale",
            "afib",
            "clinical af",
        ),
        "keywords": (
            "atrial fibrillation",
            "ecg documentation",
            "p waves",
            "rr intervals",
        ),
    },
    {
        "aliases": (
            "acute coronary syndrome",
            "syndrome coronarien aigu",
            "stemi",
            "nstemi",
        ),
        "keywords": (
            "acute coronary syndrome",
            "suspected acs",
            "st-segment elevation",
            "first medical contact",
        ),
    },
    {
        "aliases": (
            "chronic coronary syndrome",
            "syndrome coronarien chronique",
            "chronic coronary disease",
        ),
        "keywords": (
            "chronic coronary syndrome",
            "chest pain",
            "initial evaluation",
            "resting ecg",
        ),
    },
    {
        "aliases": (
            "ventricular arrhythmia",
            "arythmie ventriculaire",
            "sudden cardiac death",
            "mort subite",
        ),
        "keywords": (
            "ventricular arrhythmia",
            "ventricular tachycardia",
            "ventricular fibrillation",
            "sudden cardiac death",
        ),
    },
)

STOPWORDS = {
    "a", "an", "and", "according", "au", "aux", "avec", "dans", "de",
    "des", "du", "en", "est", "et", "for", "guideline", "guidelines",
    "in", "is", "la", "le", "les", "of", "on", "pour", "quel", "quelle",
    "quels", "quelles", "role", "the", "to", "un", "une", "what",
}

BAD_KEYWORDS = (
    "doi.org",
    "references",
    "downloaded from",
    "national cardiac society",
    "national cardiac societies",
    "society of cardiology",
    "author",
    "reviewer",
)

CONCEPT_EXPANSIONS = (
    {
        "aliases": (
            "initial assessment",
            "initial evaluation",
            "evaluation initiale",
            "prise en charge initiale",
        ),
        "keywords": (
            "first-line diagnostic tool",
            "first medical contact",
            "obtained immediately",
            "within 10 min",
        ),
    },
    {
        "aliases": (
            "12-lead ecg",
            "ecg 12 derivations",
        ),
        "keywords": (
            "12-lead ecg",
            "resting 12-lead ecg",
        ),
    },
)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text).lower())
    text = "".join(
        char
        for char in text
        if not unicodedata.combining(char)
    )
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def contains_term(text: str, term: str) -> bool:
    normalized_term = normalize_text(term)

    if not normalized_term:
        return False

    return (
        re.search(
            rf"\b{re.escape(normalized_term)}\b",
            text,
        )
        is not None
    )


def question_keywords(question: str) -> set[str]:
    normalized_question = normalize_text(question)

    tokens = {
        token
        for token in normalized_question.split()
        if (
            len(token) >= 3
            and token not in STOPWORDS
            and not token.isdigit()
        )
    }

    keywords = set(tokens)

    words = normalized_question.split()

    keywords.update(
        " ".join(words[index:index + 2])
        for index in range(len(words) - 1)
        if (
            words[index] not in STOPWORDS
            and words[index + 1] not in STOPWORDS
        )
    )

    for topic in TOPIC_EXPANSIONS:
        if any(
            contains_term(
                normalized_question,
                alias,
            )
            for alias in topic["aliases"]
        ):
            keywords.update(
                normalize_text(keyword)
                for keyword in topic["keywords"]
            )

    for concept in CONCEPT_EXPANSIONS:
        if any(
            contains_term(
                normalized_question,
                alias,
            )
            for alias in concept["aliases"]
        ):
            keywords.update(
                normalize_text(keyword)
                for keyword in concept["keywords"]
            )

    return keywords


def build_search_query(question: str) -> str:
    normalized_question = normalize_text(question)
    expansions = set()

    for topic in TOPIC_EXPANSIONS:
        if any(
            contains_term(
                normalized_question,
                alias,
            )
            for alias in topic["aliases"]
        ):
            expansions.update(topic["keywords"])

    for concept in CONCEPT_EXPANSIONS:
        if any(
            contains_term(
                normalized_question,
                alias,
            )
            for alias in concept["aliases"]
        ):
            expansions.update(concept["keywords"])

    return " ".join(
        [
            question,
            *sorted(expansions),
        ]
    )


def keyword_bonus(text: str, question: str) -> int:
    normalized_text = normalize_text(text)

    bonus = sum(
        3 if " " in keyword else 1
        for keyword in question_keywords(question)
        if contains_term(
            normalized_text,
            keyword,
        )
    )

    for keyword in BAD_KEYWORDS:
        if contains_term(
            normalized_text,
            keyword,
        ):
            bonus -= 4

    return bonus


def retrieve_documents(
    question,
    k=5,
    fetch_k=20,
):
    """
    Recherche dans TOUTES les guidelines indexées.

    Important :
    on ne filtre plus sur trois noms de fichiers codés en dur.
    Ainsi, toute nouvelle guideline ajoutée à data/guidelines puis
    réindexée peut être récupérée automatiquement.
    """

    db = load_vector_database()

    search_query = build_search_query(
        question
    )

    results = db.similarity_search(
        search_query,
        k=fetch_k,
        filter={
            "source_type": "guideline"
        },
    )

    scored_results = [
        (
            index,
            doc,
            keyword_bonus(
                doc.page_content,
                question,
            ),
        )
        for index, doc in enumerate(results)
    ]

    ranked_results = sorted(
        scored_results,
        key=lambda item: (
            item[2],
            -item[0],
        ),
        reverse=True,
    )

    relevant_results = [
        item
        for item in ranked_results
        if item[2] > 0
    ]

    selected_results = (
        relevant_results
        or ranked_results
    )

    return [
        doc
        for _, doc, _ in selected_results[:k]
    ]


def display_results(
    question,
    results,
):
    print("\nQUESTION :")
    print(question)

    print("\nPASSAGES TROUVÉS :")

    for i, doc in enumerate(
        results,
        start=1,
    ):
        print("\n" + "=" * 70)
        print(f"PASSAGE {i}")
        print(
            "Source :",
            doc.metadata.get("source"),
        )
        print(
            "Organisation :",
            doc.metadata.get("organization"),
        )
        print(
            "Année :",
            doc.metadata.get("year"),
        )
        print(
            "Page :",
            doc.metadata.get("page"),
        )
        print(
            "Bonus mots-clés :",
            keyword_bonus(
                doc.page_content,
                question,
            ),
        )
        print("-" * 70)
        print(doc.page_content[:1200])


if __name__ == "__main__":
    question = (
        "atrial fibrillation diagnosis ECG "
        "irregular RR intervals absence of P waves"
    )

    results = retrieve_documents(
        question,
        k=5,
    )

    display_results(
        question,
        results,
    )
