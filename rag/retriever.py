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
            "rythme irregulier",
            "rr irreguliers",
            "intervalles rr irreguliers",
            "absence d ondes p",
            "absence onde p",
            "ondes p non discernables",
        ),
        "keywords": (
            "atrial fibrillation",
            "ecg diagnosis",
            "ecg documentation",
            "absence of p waves",
            "p waves",
            "irregular rr intervals",
            "irregular rhythm",
        ),
    },
    {
        "aliases": (
            "acute coronary syndrome",
            "syndrome coronarien aigu",
            "stemi",
            "nstemi",
            "sus decalage st",
            "sus-decalage st",
            "st elevation",
            "st segment elevation",
            "douleur thoracique",
            "chest pain",
            "v2 v3 v4",
        ),
        "keywords": (
            "acute coronary syndrome",
            "suspected acs",
            "st-segment elevation",
            "stemi diagnostic criteria",
            "st elevation criteria",
            "anterior leads",
            "v2 v3 v4",
            "first medical contact",
            "emergency management",
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

CONCEPT_EXPANSIONS = (
    {
        "aliases": (
            "initial assessment",
            "initial evaluation",
            "evaluation initiale",
            "prise en charge initiale",
            "urgence",
            "urgent",
            "niveau d urgence",
        ),
        "keywords": (
            "first-line diagnostic tool",
            "first medical contact",
            "obtained immediately",
            "within 10 min",
            "emergency management",
        ),
    },
    {
        "aliases": (
            "12-lead ecg",
            "ecg 12 derivations",
            "ecg 12 dérivations",
        ),
        "keywords": (
            "12-lead ecg",
            "resting 12-lead ecg",
        ),
    },
    {
        "aliases": (
            "qrs",
            "qrs duration",
            "duree qrs",
            "durée qrs",
        ),
        "keywords": (
            "qrs duration",
            "qrs prolongation",
            "bundle branch block",
            "intraventricular conduction",
        ),
    },
    {
        "aliases": (
            "qtc",
            "qt interval",
            "intervalle qt",
            "qt corrige",
            "qt corrigé",
        ),
        "keywords": (
            "qt interval",
            "corrected qt",
            "qtc",
        ),
    },
)

STOPWORDS = {
    "a", "an", "and", "according", "au", "aux", "avec", "dans", "de",
    "des", "du", "en", "est", "et", "for", "guideline", "guidelines",
    "in", "is", "la", "le", "les", "of", "on", "pour", "quel", "quelle",
    "quels", "quelles", "role", "the", "to", "un", "une", "what",
    "patient", "femme", "homme", "ans", "depuis", "rapport", "clinique",
    "complet", "generer", "générer",
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


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text).lower())
    text = "".join(
        char for char in text if not unicodedata.combining(char)
    )
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def contains_term(text: str, term: str) -> bool:
    normalized_term = normalize_text(term)
    if not normalized_term:
        return False
    return re.search(rf"\b{re.escape(normalized_term)}\b", text) is not None


def _matched_expansions(question: str) -> set[str]:
    normalized_question = normalize_text(question)
    expansions = set()

    for group in (*TOPIC_EXPANSIONS, *CONCEPT_EXPANSIONS):
        if any(
            contains_term(normalized_question, alias)
            for alias in group["aliases"]
        ):
            expansions.update(group["keywords"])

    return {normalize_text(x) for x in expansions if x}


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

    keywords.update(_matched_expansions(question))
    return keywords


def build_search_query(question: str) -> str:
    expansions = sorted(_matched_expansions(question))
    return " ".join([question, *expansions]).strip()


def keyword_bonus(text: str, question: str) -> int:
    normalized_text = normalize_text(text)

    bonus = sum(
        4 if " " in keyword else 1
        for keyword in question_keywords(question)
        if contains_term(normalized_text, keyword)
    )

    # Give extra weight to high-value diagnostic phrases when the question
    # contains a matching clinical observation.
    high_value_terms = (
        "st segment elevation",
        "stemi diagnostic criteria",
        "atrial fibrillation",
        "irregular rr intervals",
        "absence of p waves",
        "first medical contact",
        "emergency management",
    )

    for term in high_value_terms:
        if contains_term(normalized_text, term):
            bonus += 3

    for keyword in BAD_KEYWORDS:
        if contains_term(normalized_text, keyword):
            bonus -= 4

    return bonus


def retrieve_documents(
    question,
    k=5,
    fetch_k=40,
):
    """
    Recherche hybride dans toutes les guidelines indexées.

    - La question clinique est enrichie avec des synonymes anglais/français.
    - Les observations ECG du cas peuvent déclencher une recherche ciblée
      vers les critères correspondants sans que le LLM n'invente le diagnostic.
    - On récupère davantage de candidats puis on les rerank par recouvrement
      lexical/clinique.
    """

    db = load_vector_database()
    search_query = build_search_query(question)

    results = db.similarity_search(
        search_query,
        k=max(fetch_k, k),
        filter={"source_type": "guideline"},
    )

    scored_results = [
        (
            index,
            doc,
            keyword_bonus(doc.page_content, question),
        )
        for index, doc in enumerate(results)
    ]

    ranked_results = sorted(
        scored_results,
        key=lambda item: (
            item[2],      # keyword/clinical bonus
            -item[0],     # preserve semantic rank as tiebreaker
        ),
        reverse=True,
    )

    relevant_results = [
        item for item in ranked_results if item[2] > 0
    ]

    # Never return an empty context solely because lexical overlap is weak.
    selected_results = relevant_results or ranked_results

    return [
        doc for _, doc, _ in selected_results[:k]
    ]


def display_results(question, results):
    print("\nQUESTION :")
    print(question)
    print("\nREQUÊTE ENRICHIE :")
    print(build_search_query(question))
    print("\nPASSAGES TROUVÉS :")

    for i, doc in enumerate(results, start=1):
        print("\n" + "=" * 70)
        print(f"PASSAGE {i}")
        print("Source :", doc.metadata.get("source"))
        print("Organisation :", doc.metadata.get("organization"))
        print("Année :", doc.metadata.get("year"))
        print("Page :", doc.metadata.get("page"))
        print("Bonus mots-clés :", keyword_bonus(doc.page_content, question))
        print("-" * 70)
        print(doc.page_content[:1200])


if __name__ == "__main__":
    question = """
    Femme de 68 ans avec douleur thoracique.
    ECG : sus-décalage ST de 3 mm en V2-V3-V4.
    Déterminer les critères ESC applicables et le niveau d'urgence.
    """

    results = retrieve_documents(question, k=5)
    display_results(question, results)
