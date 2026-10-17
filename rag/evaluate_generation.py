import re
import unicodedata
from functools import lru_cache

from bert_score import BERTScorer
from nltk.tokenize import wordpunct_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer

from rag.rag_with_llm import MEDICAL_WARNING, generate_medical_answer


STOPWORDS = {
    "a", "afin", "ainsi", "au", "aux", "avec", "ce", "ces", "cette",
    "chez", "dans", "de", "des", "du", "elle", "en", "est", "et", "il",
    "la", "le", "les", "leur", "lui", "mais", "ne", "ou", "par", "pas",
    "plus", "pour", "qu", "que", "qui", "se", "selon", "son", "sont",
    "sur", "un", "une",
}

FORBIDDEN_TERMS = {
    "fmc": "Utiliser « premier contact médical ».",
    "pattern": "Employer un terme français à la place de « pattern ».",
    "non st elevation": "Employer « sans sus-décalage du segment ST ».",
}

INFERENCE_PATTERNS = {
    r"\bcela implique\b": "Déduction introduite par « cela implique ».",
    r"\bon peut en deduire\b": "Déduction introduite par « on peut en déduire ».",
    r"\bil est probable\b": "Affirmation probabiliste non autorisée.",
    r"\bon suppose\b": "Supposition non autorisée.",
}

ENGLISH_QUOTE_MARKERS = {
    "as", "at", "especially", "first", "if", "necessary", "symptoms",
    "medical", "contact", "have", "waned", "abnormal", "pattern",
}

PASS_THRESHOLDS = {
    # ROUGE reste un garde-fou faible : un score très bas peut signaler une
    # réponse hors-sujet, mais un score élevé n'est pas exigé pour du texte
    # médical légitimement reformulé (voir note ci-dessus).
    "rouge1_f1": 0.20,
    # BERTScore F1 est le critère sémantique principal : c'est lui qui doit
    # réellement décider si le sens de la réponse correspond à la référence.
    "bertscore_f1": 0.75,
    "keyword_score": 0.80,
    "source_precision": 1.00,
    "page_hit": 1.00,
    "citation_accuracy": 0.80,
    "structure_score": 1.00,
    "warning_score": 1.00,
    "policy_score": 1.00,
}


TEST_CASES = [
    {
        "question": "Quels sont les critères ECG diagnostiques de la fibrillation atriale selon les recommandations ESC 2024 ?",
        "reference_answer": f"""
1. Réponse directe
La documentation ECG est nécessaire pour établir le diagnostic de fibrillation atriale clinique.
Elle peut reposer sur un ECG 12 dérivations ou sur un tracé ECG monodérivation d’au moins 30 secondes.

2. Critères cliniques et ECG
- Absence d’ondes P répétitives discernables
- Intervalles RR irréguliers
- Conduction auriculo-ventriculaire non altérée

3. Justification documentaire
Les recommandations ESC 2024 sur la fibrillation atriale (pages 11 et 12) précisent les critères
ECG permettant d’établir le diagnostic de fibrillation atriale clinique.

4. Limites du contexte
Les passages fournis ne précisent pas de seuil de fréquence cardiaque ni de durée minimale
de l’épisode au-delà du tracé de 30 secondes.

5. Avertissement médical
{MEDICAL_WARNING}
""",
        "expected_keywords": [
            "fibrillation atriale",
            "ECG 12 dérivations",
            "ECG monodérivation",
            "30 secondes",
            "ondes P",
            "intervalles RR irréguliers",
            "conduction auriculo-ventriculaire",
        ],
        "expected_source": "esc_afib_guidelines_2024.pdf",
        "expected_pages": {11, 12},
    },
    {
        "question": "Quel est le rôle de l’ECG 12 dérivations dans l’évaluation initiale du syndrome coronarien aigu selon les recommandations ESC 2023 ?",
        "reference_answer": f"""
1. Réponse directe
L’ECG 12 dérivations au repos est l’outil diagnostique de première intention chez les patients
présentant une suspicion de syndrome coronarien aigu.

2. Critères cliniques et ECG
- Obtention immédiate au premier contact médical
- Interprétation par un professionnel qualifié dans les 10 minutes

3. Justification documentaire
Les recommandations ESC 2023 sur le syndrome coronarien aigu (pages 17, 23 et 76) précisent
le rôle et les délais de réalisation de l’ECG 12 dérivations au repos lors de l’évaluation initiale.

4. Limites du contexte
Les passages ne précisent pas les critères diagnostiques additionnels au-delà du délai
d’obtention et d’interprétation de l’ECG.

5. Avertissement médical
{MEDICAL_WARNING}
""",
        "expected_keywords": [
            "syndrome coronarien aigu",
            "ECG 12 dérivations",
            ("première intention", "première ligne", "outil diagnostique de première ligne"),
            "premier contact médical",
            "10 minutes",
            ("professionnel qualifié", "médecin qualifié", "technicien qualifié"),
        ],
        "expected_source": "esc_acute_coronary_syndromes_2023.pdf",
        "expected_pages": {17, 23, 76},
    },
    {
        "question": "Quand faut-il répéter l’ECG 12 dérivations chez un patient suspecté de syndrome coronarien aigu selon les recommandations ESC 2023 ?",
        "reference_answer": f"""
1. Réponse directe
Chez un patient présentant une suspicion de syndrome coronarien aigu, l’ECG 12 dérivations doit être
répété lorsque cela est nécessaire, particulièrement si les symptômes se sont atténués au moment du
premier contact médical.

2. Critères cliniques et ECG
- Répétition de l’ECG 12 dérivations lorsque nécessaire
- Situation mentionnée : symptômes atténués au premier contact médical

3. Justification documentaire
Les recommandations ESC 2023 sur le syndrome coronarien aigu (pages 17 et 76) mentionnent
la nécessité de répéter l’ECG 12 dérivations dans cette situation.

4. Limites du contexte
Les passages ne précisent ni délai ni fréquence de répétition de l’ECG.

5. Avertissement médical
{MEDICAL_WARNING}
""",
        "expected_keywords": [
            "syndrome coronarien aigu",
            "ECG 12 dérivations",
            ("répété", "répéter"),
            "symptômes",
            "premier contact médical",
        ],
        "expected_source": "esc_acute_coronary_syndromes_2023.pdf",
        "expected_pages": {17, 76},
    },
    {
        "question": "Quel est le rôle de l’ECG 12 dérivations au repos dans le syndrome coronarien chronique selon les recommandations ESC 2024 ?",
        "reference_answer": f"""
1. Réponse directe
L’ECG 12 dérivations au repos fait partie de l’évaluation initiale des patients présentant une suspicion
de syndrome coronarien chronique.

2. Critères cliniques et ECG
- Contribution à l’évaluation clinique
- Contribution à l’identification d’anomalies

3. Justification documentaire
Les recommandations ESC 2024 sur le syndrome coronarien chronique (pages 18, 21 et 84) précisent
la place de l’ECG 12 dérivations au repos dans l’évaluation initiale.

4. Limites du contexte
L’absence d’anomalie ECG ne suffit pas à confirmer ou exclure un syndrome coronarien chronique.

5. Avertissement médical
{MEDICAL_WARNING}
""",
        "expected_keywords": [
            "syndrome coronarien chronique",
            "ECG 12 dérivations",
            "repos",
            "évaluation initiale",
            ("absence d’anomalie", "ECG normal"),
        ],
        "expected_source": "esc_chronic_coronary_syndromes_2024.pdf",
        "expected_pages": {18, 21, 84},
    },
]


def normalize_text(text):
    """Normalise accents, Markdown et ponctuation pour les comparaisons."""

    text = text.replace(MEDICAL_WARNING, " ")
    text = re.sub(r"[*_`#]", " ", text)
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def content_tokens(text):
    """Retourne les mots porteurs de contenu, sans avertissement ni mots-outils."""

    return [
        token
        for token in normalize_text(text).split()
        if token not in STOPWORDS and len(token) > 1
    ]


def prepare_metric_text(text):
    """Retire l'avertissement fixe, qui ne fait pas partie du contenu à comparer."""

    return text.replace(MEDICAL_WARNING, " ").strip()


def metric_tokens(text):
    """Tokenise un texte français sans exiger le téléchargement de données NLTK."""

    return wordpunct_tokenize(prepare_metric_text(text).lower())


def calculate_bleu(reference, prediction):
    """Calcule le BLEU-4 NLTK officiel avec lissage pour une phrase."""

    reference_tokens = metric_tokens(reference)
    prediction_tokens = metric_tokens(prediction)
    if not reference_tokens or not prediction_tokens:
        return 0.0

    return sentence_bleu(
        [reference_tokens],
        prediction_tokens,
        smoothing_function=SmoothingFunction().method1,
    )


def calculate_rouge(reference, prediction):
    """Calcule ROUGE-1, ROUGE-2 et ROUGE-L avec la bibliothèque rouge-score."""

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=False,
    )
    return scorer.score(
        prepare_metric_text(reference),
        prepare_metric_text(prediction),
    )


@lru_cache(maxsize=1)
def get_bert_scorer():
    """Charge une seule fois le modèle multilingue utilisé par BERTScore."""

    return BERTScorer(lang="fr")


def calculate_bertscore(reference, prediction):
    """Calcule précision, rappel et F1 BERTScore pour une paire de textes."""

    reference = prepare_metric_text(reference)
    prediction = prepare_metric_text(prediction)
    if not reference or not prediction:
        return 0.0, 0.0, 0.0

    precision, recall, f1 = get_bert_scorer().score(
        [prediction],
        [reference],
    )
    return precision.item(), recall.item(), f1.item()


def contains_phrase(text, phrase):
    normalized_text = f" {normalize_text(text)} "
    normalized_phrase = normalize_text(phrase)
    return re.search(rf"\b{re.escape(normalized_phrase)}\b", normalized_text) is not None


def calculate_keyword_score(expected_keywords, prediction):
    """Mesure la couverture des concepts médicaux attendus, accents neutralisés."""

    if not expected_keywords:
        return 1.0, []

    missing_keywords = []
    for keyword in expected_keywords:
        alternatives = keyword if isinstance(keyword, tuple) else (keyword,)
        if not any(contains_phrase(prediction, alternative) for alternative in alternatives):
            missing_keywords.append(" / ".join(alternatives))
    score = (len(expected_keywords) - len(missing_keywords)) / len(expected_keywords)
    return score, missing_keywords


def validate_structure(answer):
    """Vérifie la présence et l’ordre des cinq sections obligatoires."""

    matches = re.findall(r"(?m)^\s*(?:\*\*)?([1-5])\.\s*", answer)
    expected = ["1", "2", "3", "4", "5"]
    return sum(number in matches for number in expected) / 5, matches == expected


def validate_citations(answer, docs):
    """Vérifie que chaque couple passage/page cité correspond aux métadonnées RAG."""

    citations = []
    for line in answer.splitlines():
        passage_match = re.search(r"passage\s*[\s\u202f]*(\d+)", line, re.IGNORECASE)
        if not passage_match:
            continue

        page_match = re.search(r"(?:page|p\.)\s*[\s\u202f]*(\d+)", line, re.IGNORECASE)
        passage_number = int(passage_match.group(1))
        cited_page = int(page_match.group(1)) if page_match else None

        valid = 1 <= passage_number <= len(docs)
        if valid and cited_page is not None:
            actual_page = docs[passage_number - 1].metadata.get("page")
            valid = str(actual_page) == str(cited_page)
        else:
            valid = False

        citations.append(valid)

    accuracy = sum(citations) / len(citations) if citations else 0.0
    return accuracy, len(citations)


def calculate_source_precision(docs, expected_source):
    if not docs:
        return 0.0

    correct = sum(
        expected_source in str(doc.metadata.get("source", ""))
        for doc in docs
    )
    return correct / len(docs)


def calculate_page_hit(docs, expected_pages):
    retrieved_pages = {doc.metadata.get("page") for doc in docs}
    return float(bool(retrieved_pages.intersection(expected_pages)))


def extract_section(answer, section_number):
    pattern = re.compile(
        rf"(?ms)^\s*(?:\*\*)?{section_number}\.\s*.*?(?=^\s*(?:\*\*)?[1-5]\.\s|\Z)"
    )
    match = pattern.search(answer)
    return match.group(0) if match else ""


def detect_policy_violations(answer):
    """Détecte les violations observables sans prétendre juger toute la vérité médicale."""

    violations = []
    normalized_answer = normalize_text(answer)

    for term, message in FORBIDDEN_TERMS.items():
        if contains_phrase(answer, term):
            violations.append(message)

    for pattern, message in INFERENCE_PATTERNS.items():
        if re.search(pattern, normalized_answer):
            violations.append(message)

    quoted_segments = re.findall(r"[«\"]([^»\"]+)[»\"]", answer)
    for segment in quoted_segments:
        segment_tokens = set(normalize_text(segment).split())
        if segment_tokens.intersection(ENGLISH_QUOTE_MARKERS):
            violations.append("Citation ou extrait anglais détecté entre guillemets.")
            break

    limits_section = extract_section(answer, 4)
    if re.search(r"\b(par exemple|ex\.)\b", limits_section, re.IGNORECASE):
        violations.append("Exemple potentiellement non documenté dans les limites du contexte.")

    risky_patterns = {
        r"\bconfirme (?:un |le )?nstemi\b": "Un NSTEMI est présenté comme confirmé.",
        r"\babsence de sus decalage.*confirme\b": "Conclusion tirée de l’absence de sus-décalage ST.",
    }
    for pattern, message in risky_patterns.items():
        if re.search(pattern, normalized_answer):
            violations.append(message)

    return sorted(set(violations))


def evaluate_one_case(case):
    question = case["question"]

    print("\n" + "=" * 100)
    print("QUESTION")
    print("=" * 100)
    print(question)

    result = generate_medical_answer(question)
    generated_answer = result["answer"]
    docs = result["retrieved_docs"]

    print("\n" + "=" * 100)
    print("RÉPONSE GÉNÉRÉE")
    print("=" * 100)
    print(generated_answer)

    bleu = calculate_bleu(case["reference_answer"], generated_answer)
    rouge = calculate_rouge(case["reference_answer"], generated_answer)
    bertscore_precision, bertscore_recall, bertscore_f1 = calculate_bertscore(
        case["reference_answer"], generated_answer
    )
    keyword_score, missing_keywords = calculate_keyword_score(
        case["expected_keywords"], generated_answer
    )
    source_precision = calculate_source_precision(docs, case["expected_source"])
    page_hit = calculate_page_hit(docs, case["expected_pages"])
    citation_accuracy, citation_count = validate_citations(generated_answer, docs)
    structure_score, structure_in_order = validate_structure(generated_answer)
    warning_score = float(MEDICAL_WARNING in generated_answer)
    violations = detect_policy_violations(generated_answer)
    policy_score = float(not violations)

    metrics = {
        "bleu": bleu,
        "rouge1_precision": rouge["rouge1"].precision,
        "rouge1_recall": rouge["rouge1"].recall,
        "rouge1_f1": rouge["rouge1"].fmeasure,
        "rouge2_precision": rouge["rouge2"].precision,
        "rouge2_recall": rouge["rouge2"].recall,
        "rouge2_f1": rouge["rouge2"].fmeasure,
        "rougeL_precision": rouge["rougeL"].precision,
        "rougeL_recall": rouge["rougeL"].recall,
        "rougeL_f1": rouge["rougeL"].fmeasure,
        "bertscore_precision": bertscore_precision,
        "bertscore_recall": bertscore_recall,
        "bertscore_f1": bertscore_f1,
        "keyword_score": keyword_score,
        "source_precision": source_precision,
        "page_hit": page_hit,
        "citation_accuracy": citation_accuracy,
        "structure_score": structure_score,
        "warning_score": warning_score,
        "policy_score": policy_score,
    }

    failed_thresholds = [
        name
        for name, threshold in PASS_THRESHOLDS.items()
        if metrics[name] < threshold
    ]
    verdict = "PASS" if not failed_thresholds else "À REVOIR"

    print("\n" + "-" * 100)
    print("MÉTRIQUES DE GÉNÉRATION (indicateurs secondaires)")
    print("-" * 100)
    print(f"BLEU-4 (NLTK)          : {bleu:.3f}")
    for rouge_name in ("rouge1", "rouge2", "rougeL"):
        score = rouge[rouge_name]
        print(f"{rouge_name.upper():<8} précision    : {score.precision:.3f}")
        print(f"{rouge_name.upper():<8} rappel       : {score.recall:.3f}")
        print(f"{rouge_name.upper()} F1           : {score.fmeasure:.3f}")
    print(f"BERTScore précision    : {bertscore_precision:.3f}")
    print(f"BERTScore rappel       : {bertscore_recall:.3f}")
    print(f"BERTScore F1           : {bertscore_f1:.3f}")
    print(f"Couverture mots-clés   : {keyword_score:.3f}")
    print("Mots-clés manquants    :", missing_keywords or "aucun")

    print("\n" + "-" * 100)
    print("FIDÉLITÉ DOCUMENTAIRE")
    print("-" * 100)
    print(f"Précision des sources  : {source_precision:.3f}")
    print(f"Page attendue trouvée  : {page_hit:.3f}")
    print(f"Exactitude citations   : {citation_accuracy:.3f} ({citation_count} citation(s))")

    print("\n" + "-" * 100)
    print("CONFORMITÉ ET SÉCURITÉ")
    print("-" * 100)
    print(f"Structure 5 sections   : {structure_score:.3f} (ordre correct : {structure_in_order})")
    print(f"Avertissement exact    : {warning_score:.3f}")
    print(f"Règles automatiques    : {policy_score:.3f}")
    print("Violations détectées   :", violations or "aucune")
    print(f"VERDICT                 : {verdict}")
    if failed_thresholds:
        print("Seuils non atteints     :", ", ".join(failed_thresholds))

    return {
        "question": question,
        "generated_answer": generated_answer,
        "reference_answer": case["reference_answer"],
        "verdict": verdict,
        "violations": violations,
        "missing_keywords": missing_keywords,
        **metrics,
    }


def evaluate_generation():
    results = [evaluate_one_case(case) for case in TEST_CASES]

    metric_names = [
        "bleu",
        "rouge1_precision",
        "rouge1_recall",
        "rouge1_f1",
        "rouge2_precision",
        "rouge2_recall",
        "rouge2_f1",
        "rougeL_precision",
        "rougeL_recall",
        "rougeL_f1",
        "bertscore_precision",
        "bertscore_recall",
        "bertscore_f1",
        "keyword_score",
        "source_precision",
        "page_hit",
        "citation_accuracy",
        "structure_score",
        "warning_score",
        "policy_score",
    ]

    print("\n" + "=" * 100)
    print("RÉSULTATS GLOBAUX")
    print("=" * 100)
    for name in metric_names:
        average = sum(result[name] for result in results) / len(results)
        print(f"{name:<24}: {average:.3f}")

    passed = sum(result["verdict"] == "PASS" for result in results)
    print(f"Cas validés             : {passed}/{len(results)}")
    print(
        "Conclusion              :",
        "PASS GLOBAL" if passed == len(results) else "RÉVISION NÉCESSAIRE",
    )
    print(
        "Note                    : la fidélité sémantique complète nécessite toujours une revue humaine."
    )

    return results


if __name__ == "__main__":
    evaluate_generation()