

from __future__ import annotations

import csv
import json
import math
import os
import re
import statistics
import sys
import time
import types
from pathlib import Path

# ---------------------------------------------------------------------------
# Contournement de compatibilité : DOIT précéder tout import de ragas.
# ---------------------------------------------------------------------------
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _stub = types.ModuleType("langchain_community.chat_models.vertexai")

    class _StubChatVertexAI:  # jamais instancié, sert seulement à satisfaire l'import
        pass

    _stub.ChatVertexAI = _StubChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _stub


sys.path.insert(0, str(Path(__file__).resolve().parent))        # evaluation/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))    # racine du projet

from orchestrator.langgraph_orchestrator import run_agentic_rag  # noqa: E402

from evaluate_full_pipeline import (  # noqa: E402  (réutilise l'existant, ne le duplique pas)
    check_expected_sources,
    check_structure,
    check_warning,
    get_groq_wait_seconds,
    is_groq_rate_limit_error,
    normalize_critic_score,
)
from recalculate_metrics import (  # noqa: E402  (métriques texte déjà corrigées)
    calculate_bertscore_batch,
    calculate_bleu,
    calculate_meteor,
    calculate_rouge,
    extract_direct_answer,
)
from evaluation.reference_lock import verify_frozen_references  # noqa: E402


# ============================================================
# CHEMINS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_FILE = PROJECT_ROOT / "evaluation" / "questions.json"
RESULTS_FOLDER = PROJECT_ROOT / "evaluation" / "results"
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

JSON_FILE = RESULTS_FOLDER / "evaluation_results_full.json"
CSV_FILE = RESULTS_FOLDER / "evaluation_results_full.csv"
SUMMARY_FILE = RESULTS_FOLDER / "evaluation_summary_full.json"
EVALUATION_PROTOCOL_VERSION = 3


# ============================================================
# CONFIGURATION RAGAS
# ============================================================

RAGAS_LLM_MODEL = os.getenv(
    "RAGAS_LLM_MODEL",
    "gemini-3.1-flash-lite"
)
RAGAS_EMBEDDING_MODEL = os.getenv(
    "RAGAS_EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
# Endpoint OpenAI-compatible du juge Gemini, modifiable pour les tests locaux.
RAGAS_BASE_URL = os.getenv(
    "RAGAS_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai/"
)
RAGAS_MAX_TOKENS = int(os.getenv("RAGAS_MAX_TOKENS", "4096"))
RAGAS_FAITHFULNESS_FULL_MAX_TOKENS = int(
    os.getenv("RAGAS_FAITHFULNESS_FULL_MAX_TOKENS", "8192")
)
RAGAS_MAX_CONTEXTS = int(os.getenv("RAGAS_MAX_CONTEXTS", "20"))
RAGAS_MAX_CHARS_PER_CONTEXT = int(os.getenv("RAGAS_MAX_CHARS_PER_CONTEXT", "1500"))
RAGAS_ENABLED = os.getenv("RAGAS_ENABLED", "1").strip().lower() not in {"0", "false", "no"}

RAGAS_METRIC_NAMES = (
    "ragas_faithfulness",
    "ragas_answer_relevancy",
    "ragas_context_precision",
    "ragas_context_recall",
    "ragas_factual_correctness",
)
_RAGAS_NEEDS_CONTEXTS = {"ragas_faithfulness", "ragas_context_precision", "ragas_context_recall"}


# ============================================================
# DÉCOUPAGE DES CONTEXTES RÉCUPÉRÉS EN PASSAGES INDIVIDUELS
# ============================================================

# Chaque agent formate ses passages avec un en-tête numéroté propre à sa
# source (voir agents/literature_agent.py, rag/rag_with_llm.py,
# rag_ecg_reference/retriever.py, rag_research/retriever.py,
# rag_ptbxl/ptbxl_retriever.py, agents/websearch_agent.py). On s'appuie
# dessus pour re-séparer les blocs en passages individuels pour RAGAS.
CONTEXT_FIELDS: dict[str, str] = {
    "rag_context": r"\[PASSAGE GUIDELINE \d+\]",
    "ecg_reference_context": r"PASSAGE ECG REFERENCE \d+",
    "research_context": r"PASSAGE RESEARCH \d+",
    "ptbxl_context": r"CAS PTB-XL \d+",
    "web_context": r"PASSAGE WEB \d+",
}

_TRUNCATION_MARKER = "\n...[contexte tronqué]..."


def split_context_blob(text: str, header_pattern: str) -> list[str]:
    """Redécoupe un bloc de contexte concaténé en passages individuels.

    Repli : si aucun en-tête n'est trouvé (champ vide, message d'erreur du
    retriever, « aucune source Web trouvée »...), le bloc entier est renvoyé
    comme un seul passage plutôt que d'être perdu — c'est un signal honnête
    (RAGAS verra alors ce message d'échec comme le seul contexte disponible).
    """
    text = (text or "").strip()
    if not text:
        return []
    text = text.replace(_TRUNCATION_MARKER, "")

    pattern = re.compile(rf"(?=^{header_pattern}\s*$)", re.MULTILINE)
    parts = [p.strip() for p in pattern.split(text) if p.strip()]
    return parts if parts else [text]


def extract_sections_1_to_4(answer: str) -> str | None:
    """Extrait les sections cliniques 1 à 4, sans l'avertissement médical."""

    text = str(answer or "").strip()
    if not text:
        return None

    start_match = re.search(
        r"(?im)^[ \t]*(?:\*\*)?1\.[ \t]*Réponse directe(?:\*\*)?[ \t]*$",
        text,
    )
    if start_match is None:
        return None

    end_match = re.search(
        r"(?im)^[ \t]*(?:\*\*)?5\.[ \t]*Avertissement médical(?:\*\*)?[ \t]*$",
        text,
    )
    end = end_match.start() if end_match else len(text)
    result = text[start_match.start():end].strip()
    return result or None


def collect_contexts(
    result: dict,
    max_contexts: int = RAGAS_MAX_CONTEXTS,
    max_chars: int = RAGAS_MAX_CHARS_PER_CONTEXT,
) -> list[str]:
    """Rassemble tous les passages récupérés (4 corpus + web) pour RAGAS."""
    contexts: list[str] = []
    for field, header_pattern in CONTEXT_FIELDS.items():
        for passage in split_context_blob(result.get(field, ""), header_pattern):
            contexts.append(passage[:max_chars])
    return contexts[:max_contexts]


# ============================================================
# MÉTRIQUES RAGAS
# ============================================================

def _faithfulness_kwargs(question, answer, contexts, reference):
    return {"user_input": question, "response": answer, "retrieved_contexts": contexts}


def _answer_relevancy_kwargs(question, answer, contexts, reference):
    return {"user_input": question, "response": answer}


def _context_precision_kwargs(question, answer, contexts, reference):
    return {"user_input": question, "reference": reference, "retrieved_contexts": contexts}


def _context_recall_kwargs(question, answer, contexts, reference):
    return {"user_input": question, "retrieved_contexts": contexts, "reference": reference}


def _factual_correctness_kwargs(question, answer, contexts, reference):
    return {"response": answer, "reference": reference}


_KWARGS_BUILDERS = {
    "ragas_faithfulness": _faithfulness_kwargs,
    "ragas_answer_relevancy": _answer_relevancy_kwargs,
    "ragas_context_precision": _context_precision_kwargs,
    "ragas_context_recall": _context_recall_kwargs,
    "ragas_factual_correctness": _factual_correctness_kwargs,
}


def build_ragas_metrics():
    """Construit les métriques RAGAS avec un juge OpenAI-compatible.

    Renvoie None si RAGAS_ENABLED=0, si GROQ_API_KEY est absent, ou si
    l'initialisation échoue (paquets manquants, réseau...) — dans ce cas
    toutes les colonnes ragas_* resteront 'null' pour l'exécution plutôt que
    de faire planter le script.
    """
    if not RAGAS_ENABLED:
        print("RAGAS désactivé (RAGAS_ENABLED=0) : seules les métriques texte seront calculées.")
        return None

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("GEMINI_API_KEY absent : RAGAS désactivé pour cette exécution.")
        return None

    try:
        from openai import AsyncOpenAI
        from ragas.embeddings.base import embedding_factory
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecisionWithReference,
            ContextRecall,
            FactualCorrectness,
            Faithfulness,
        )

        # Gemini et Groq exposent une API compatible OpenAI. Le client reste
        # donc identique ; RAGAS_BASE_URL choisit effectivement le fournisseur.
        # Client ASYNCHRONE obligatoire : les métriques RAGAS appellent le LLM via
        # agenerate() et lèvent « Cannot use agenerate() with a synchronous client »
        # avec le client OpenAI synchrone. Cette TypeError était attrapée
        # silencieusement, d'où des colonnes ragas_* entièrement vides.
        client = AsyncOpenAI(api_key=api_key, base_url=RAGAS_BASE_URL)
        llm = llm_factory(
            RAGAS_LLM_MODEL,
            client=client,
            max_tokens=RAGAS_MAX_TOKENS,
        )
        faithfulness_full_llm = llm_factory(
            RAGAS_LLM_MODEL,
            client=client,
            max_tokens=RAGAS_FAITHFULNESS_FULL_MAX_TOKENS,
        )
        embeddings = embedding_factory("huggingface", RAGAS_EMBEDDING_MODEL)

        return {
            "ragas_faithfulness": Faithfulness(llm=llm),
            "ragas_answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
            "ragas_context_precision": ContextPrecisionWithReference(llm=llm),
            "ragas_context_recall": ContextRecall(llm=llm),
            "ragas_factual_correctness": FactualCorrectness(llm=llm),
            "ragas_faithfulness_full": Faithfulness(
                llm=faithfulness_full_llm
            ),
        }
    except Exception as exc:
        print(f"Impossible d'initialiser RAGAS ({exc!r}).")
        print("Les colonnes ragas_* resteront 'null' pour cette exécution.")
        print("Vérifiez : pip install ragas sentence-transformers openai")
        return None


def _score_one(
    name: str,
    metric,
    kwargs: dict,
    label
) -> tuple[float | None, str | None]:
    """
    Appelle une métrique RAGAS.

    Renvoie :
        (valeur, None)
    ou :
        (None, message_d_erreur)

    Réessaie automatiquement dans les cas suivants :
    - rate limit fournisseur (429)
    - IncompleteOutputException
    - json_validate_failed
    - Failed to validate JSON
    - erreur de connexion ou timeout transitoire

    Le nombre total de tentatives est limité afin qu'une métrique
    défaillante ne bloque pas toute l'évaluation.
    """

    max_attempts = 3

    for attempt in range(1, max_attempts + 1):

        try:
            result = metric.score(**kwargs)

        except Exception as exc:

            error = (
                f"{type(exc).__name__}: "
                f"{str(exc)[:300]}"
            )

            is_last_attempt = (
                attempt == max_attempts
            )

            # =====================================================
            # 1. RATE LIMIT FOURNISSEUR
            # =====================================================

            if is_groq_rate_limit_error(exc):

                if is_last_attempt:
                    return None, error

                wait = get_groq_wait_seconds(
                    str(exc)
                )

                print(
                    f"  [RAGAS:{name}] limite API ({label}), "
                    f"tentative {attempt}/{max_attempts}, "
                    f"pause {wait / 60:.1f} min..."
                )

                time.sleep(wait)

                continue

            # =====================================================
            # 2. SORTIE INCOMPLÈTE RAGAS
            # =====================================================

            if (
                "IncompleteOutputException"
                in type(exc).__name__
            ):

                if is_last_attempt:
                    return None, error

                print(
                    f"  [RAGAS:{name}] sortie incomplète "
                    f"({label}), "
                    f"tentative {attempt}/{max_attempts}"
                )

                time.sleep(2)

                continue

            # =====================================================
            # 3. JSON INVALIDE / STRUCTURED OUTPUT GROQ
            # =====================================================

            error_text = str(exc)

            if (
                "json_validate_failed"
                in error_text
                or
                "Failed to validate JSON"
                in error_text
            ):

                if is_last_attempt:
                    return None, error

                print(
                    f"  [RAGAS:{name}] JSON invalide "
                    f"({label}), "
                    f"tentative {attempt}/{max_attempts}"
                )

                time.sleep(2)

                continue

            # =====================================================
            # 4. CONNEXION / TIMEOUT TRANSITOIRE
            # =====================================================

            exception_name = type(exc).__name__.lower()
            normalized_error_text = error_text.lower()
            is_connection_error = (
                any(
                    marker in exception_name
                    for marker in (
                        "connectionerror",
                        "connecterror",
                        "timeout",
                    )
                )
                or "connection error" in normalized_error_text
                or "connection reset" in normalized_error_text
                or "temporarily unavailable" in normalized_error_text
            )

            if is_connection_error:
                if is_last_attempt:
                    return None, error

                wait = 2 ** attempt
                print(
                    f"  [RAGAS:{name}] erreur de connexion "
                    f"({label}), tentative {attempt}/{max_attempts}, "
                    f"pause {wait}s..."
                )
                time.sleep(wait)
                continue

            # =====================================================
            # 5. AUTRE ERREUR
            # =====================================================

            return None, error

        # =========================================================
        # VÉRIFICATION DE LA VALEUR RETOURNÉE
        # =========================================================

        value = result.value

        if (
            value is None
            or (
                isinstance(value, float)
                and math.isnan(value)
            )
        ):
            return None, "valeur vide (None/NaN)"

        return (
            round(float(value), 4),
            None
        )

    return (
        None,
        "nombre maximal de tentatives RAGAS atteint"
    )


def score_ragas_metrics(
    metrics: dict | None,
    question_id,
    question: str,
    answer: str | None,
    contexts: list[str],
    reference: str,
    errors: dict | None = None,
) -> dict:
    """Calcule les scores RAGAS d'une question.

    Une métrique qui a besoin de contexte mais n'en a reçu aucun (retrieval
    vide) reste à None. Une métrique absente de ``metrics`` (désactivée par le
    test de démarrage) est ignorée. Si ``errors`` est fourni, il reçoit
    {nom_métrique: message} pour chaque échec : rien n'est plus perdu en
    silence.
    """
    scores: dict = {name: None for name in RAGAS_METRIC_NAMES}
    if metrics is None or not answer:
        return scores

    for name in RAGAS_METRIC_NAMES:
        if name not in metrics:
            continue
        if name in _RAGAS_NEEDS_CONTEXTS and not contexts:
            continue

        kwargs = _KWARGS_BUILDERS[name](question, answer, contexts, reference)
        value, error = _score_one(name, metrics[name], kwargs, f"question {question_id}")
        scores[name] = value
        if error:
            print(f"  [RAGAS:{name}] échec (question {question_id}) : {error}")
            if errors is not None:
                errors[name] = error

    return scores


def smoke_test_ragas(metrics: dict | None) -> dict | None:
    """Teste chaque métrique sur un exemple minimal AVANT les 30 questions.

    Un échec systématique se voit en quelques secondes au lieu de ressortir
    en fin de run sous forme de colonnes vides. Renvoie les seules métriques
    qui fonctionnent. Si aucune ne fonctionne, arrête le script (sauf si
    RAGAS_ALLOW_FAILURE=1, auquel cas la suite se fait sans RAGAS).
    """
    if metrics is None:
        return None

    question = "Quelle est la capitale de la France ?"
    answer = "La capitale de la France est Paris."
    contexts = ["Paris est la capitale de la France."]
    reference = "Paris est la capitale de la France."

    print("Vérification de RAGAS sur un exemple minimal...")
    working: dict = {}
    failed: dict = {}
    for name in RAGAS_METRIC_NAMES:
        if name not in metrics:
            continue
        kwargs = _KWARGS_BUILDERS[name](question, answer, contexts, reference)
        value, error = _score_one(name, metrics[name], kwargs, "test de démarrage")
        if error is None:
            working[name] = metrics[name]
            print(f"  OK     {name} = {value}")
        else:
            failed[name] = error
            print(f"  ECHEC  {name} : {error}")

    # Le test court valide la logique Faithfulness commune. Le juge dédié à
    # la version sections 1-4 est conservé avec son budget de sortie supérieur.
    if (
        "ragas_faithfulness" in working
        and "ragas_faithfulness_full" in metrics
    ):
        working["ragas_faithfulness_full"] = metrics[
            "ragas_faithfulness_full"
        ]

    if not working:
        message = ("RAGAS ne fonctionne avec aucune métrique (voir les erreurs ci-dessus). "
                   "Corrigez-les, ou lancez avec RAGAS_ENABLED=0 pour ne calculer que "
                   "les métriques texte.")
        if os.getenv("RAGAS_ALLOW_FAILURE", "0").strip() == "1":
            print(message + "\nRAGAS_ALLOW_FAILURE=1 : on continue sans RAGAS.")
            return None
        raise SystemExit(message)

    if failed:
        print("Métriques RAGAS désactivées pour cette exécution :", ", ".join(failed))
    return working


# ============================================================
# ÉVALUATION D'UNE QUESTION
# ============================================================

def evaluate_question_full(item: dict, ragas_metrics: dict | None) -> dict:
    question_id = item["id"]
    question = item["question"]

    # Référence complète conservée pour la traçabilité.
    reference_full = item.get("reference_answer", "")
    # Les métriques standard comparent uniquement les réponses directes.
    reference = (
        extract_direct_answer(reference_full)
        or reference_full
    )

    print("\n" + "=" * 80)
    print(f"QUESTION {question_id}")
    print("=" * 80)
    print(question)

    start = time.perf_counter()
    result: dict = {}
    error = None

    while True:
        try:
            result = run_agentic_rag(
                question=question,
                ecg_input_path=item.get("ecg_input_path", ""),
                ecg_required=item.get("ecg_required", False),
                test_force_improve=False,
            )
            break
        except Exception as exc:
            if is_groq_rate_limit_error(exc):
                wait = get_groq_wait_seconds(str(exc))
                print(f"[GROQ RATE LIMIT] question {question_id}, "
                      f"pause {wait / 60:.1f} min...")
                time.sleep(wait)
                continue
            print(f"[ERREUR] question {question_id} : {type(exc).__name__}: {exc}")
            result, error = {}, str(exc)
            break

    latency = time.perf_counter() - start
    final_answer = result.get("final_answer", "")
    critic = result.get("final_critic_result", {})
    confidence = result.get("confidence_result", {})
    publication_status = result.get("publication_status", "ERROR")

    contexts = collect_contexts(result)
    full_clinical_answer = extract_sections_1_to_4(final_answer)

    # --- Métriques texte : section « Réponse directe » uniquement, ---
    # --- REJECTED exclu (voir recalculate_metrics.py) -----------------
    excerpt = None
    rouge = {"rouge1": None, "rouge2": None, "rougeL": None}
    bleu = meteor = None
    metrics_skipped_reason = None

    if publication_status == "REJECTED":
        metrics_skipped_reason = "rejected_publication"
    elif not reference:
        metrics_skipped_reason = "missing_reference"
    else:
        excerpt = extract_direct_answer(final_answer)
        if excerpt is None:
            metrics_skipped_reason = "no_extractable_section"
        else:
            try:
                rouge = calculate_rouge(reference, excerpt)
            except Exception as exc:
                print(f"  [ROUGE] échec (id {question_id}) : {exc!r}")
            try:
                bleu = calculate_bleu(reference, excerpt)
            except Exception as exc:
                print(f"  [BLEU] échec (id {question_id}) : {exc!r}")
            try:
                meteor = calculate_meteor(reference, excerpt)
            except Exception as exc:
                print(f"  [METEOR] échec (id {question_id}) : {exc!r}")

    ragas_errors: dict = {}
    ragas_scores = (
        score_ragas_metrics(ragas_metrics, question_id, question, excerpt, contexts,
                            reference, errors=ragas_errors)
        if metrics_skipped_reason is None
        else {name: None for name in RAGAS_METRIC_NAMES}
    )

    # Faithfulness complémentaire sur tout le contenu clinique, sections 1 à 4.
    ragas_faithfulness_full = None
    if (
        ragas_metrics is not None
        and "ragas_faithfulness_full" in ragas_metrics
        and publication_status != "REJECTED"
        and full_clinical_answer
        and contexts
    ):
        full_kwargs = _faithfulness_kwargs(
            question,
            full_clinical_answer,
            contexts,
            reference,
        )
        ragas_faithfulness_full, full_error = _score_one(
            "ragas_faithfulness_full",
            ragas_metrics["ragas_faithfulness_full"],
            full_kwargs,
            f"question {question_id} - sections 1 à 4",
        )
        if full_error:
            print(
                "  [RAGAS:ragas_faithfulness_full] "
                f"échec (question {question_id}) : {full_error}"
            )
            ragas_errors["ragas_faithfulness_full"] = full_error

    structure = check_structure(final_answer)
    source_eval = check_expected_sources(final_answer, item.get("expected_sources", []))

    row = {
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "id": question_id,
        "category": item.get("category", ""),
        "question": question,
        "reference_answer": reference_full,
        "metrics_reference_direct": reference,
        "final_answer": final_answer,
        "rejected_answer": result.get("rejected_answer", ""),
        "clinical_answer_sections_1_4": full_clinical_answer,
        "metrics_reference_excerpt": excerpt,
        "metrics_skipped_reason": metrics_skipped_reason,
        "retrieved_contexts": contexts,
        "n_contexts": len(contexts),

        "latency_seconds": round(latency, 3),
        "publication_status": publication_status,
        "error": error,

        "critic_verdict": critic.get("verdict"),
        "critic_score": critic.get("score"),
        "critic_max_score": critic.get("max_score", 7),
        "critic_normalized": normalize_critic_score(critic),
        "critic_applicable_score": critic.get("applicable_score"),
        "critic_applicable_max_score": critic.get("applicable_max_score"),
        "critic_applicable_normalized": critic.get("applicable_normalized"),
        "critic_feedback": critic.get("feedback", []),

        "confidence": confidence.get("confidence"),
        "entropy": confidence.get("entropy"),
        "confidence_status": confidence.get("status"),
        "confidence_error": confidence.get("error"),

        "ptbxl_ids": result.get("ptbxl_case_ids", [])[:3],
        "web_used": result.get("web_search_used", False),
        "web_source": result.get("web_source", ""),
        "iterations": result.get("iterations", 0),

        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],
        "rougeL": rouge["rougeL"],
        "bleu": bleu,
        "meteor": meteor,
        "bertscore": None,   # rempli après coup, en lot (voir add_bertscore)

        "structure_score": structure["structure_score"],
        "structure_complete": structure["structure_complete"],
        "medical_warning": check_warning(final_answer),

        "source_hits": source_eval["source_hits"],
        "source_total": source_eval["source_total"],
        "source_recall": source_eval["source_recall"],
        "missing_sources": source_eval["missing_sources"],

        **ragas_scores,
        "ragas_faithfulness_full": ragas_faithfulness_full,
        "ragas_errors": ragas_errors,
    }

    print("Publication :", row["publication_status"], "| Critic :", row["critic_verdict"])
    print("Contextes récupérés :", row["n_contexts"])
    print(
        "RAGAS -> faithfulness:", row["ragas_faithfulness"],
        "| faithfulness_full:", row["ragas_faithfulness_full"],
        "| answer_relevancy:", row["ragas_answer_relevancy"],
        "| context_precision:", row["ragas_context_precision"],
        "| context_recall:", row["ragas_context_recall"],
        "| factual_correctness:", row["ragas_factual_correctness"],
    )
    print("Latence :", row["latency_seconds"], "s")

    return row


# ============================================================
# BERTSCORE (en lot, après la boucle principale)
# ============================================================

def add_bertscore(rows: list[dict]) -> None:
    """Calcule BERTScore en un seul appel groupé pour toutes les lignes
    éligibles. Idempotent : peut être rappelé sans risque sur une reprise.
    """
    eligible = [
        row
        for row in rows
        if row.get("metrics_skipped_reason") is None
        and row.get("metrics_reference_excerpt")
    ]
    if not eligible:
        return

    references = [
        row.get("metrics_reference_direct")
        or row["reference_answer"]
        for row in eligible
    ]
    predictions = [row["metrics_reference_excerpt"] for row in eligible]
    scores = calculate_bertscore_batch(references, predictions)
    if scores is None:
        return

    for row, score in zip(eligible, scores):
        row["bertscore"] = round(score, 4) if score is not None else None


# ============================================================
# RÉSUMÉ
# ============================================================

def mean_valid(rows: list[dict], key: str) -> float | None:
    values = [
        row[key] for row in rows
        if isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)
    ]
    return round(statistics.mean(values), 4) if values else None


def build_summary(rows: list[dict]) -> dict:
    total = len(rows)
    confidence_success_rows = [
        row
        for row in rows
        if row.get("confidence_status") == "success"
    ]
    pass_count = sum(
        row.get("publication_status") == "PASS"
        for row in rows
    )
    critic_pass_count = sum(
        row.get("critic_verdict") == "PASS"
        for row in rows
    )
    structure_complete_count = sum(
        row.get("structure_complete") is True
        for row in rows
    )
    medical_warning_count = sum(
        row.get("medical_warning") is True
        for row in rows
    )
    total_iterations = sum(
        value
        for row in rows
        if isinstance((value := row.get("iterations")), int)
        and not isinstance(value, bool)
    )

    skipped = {}
    for row in rows:
        reason = row.get("metrics_skipped_reason")
        if reason:
            skipped[reason] = skipped.get(reason, 0) + 1

    return {
        "questions": total,
        "publication_pass_count": pass_count,
        "publication_pass_rate": round(pass_count / total, 4) if total else 0,
        "critic_pass_count": critic_pass_count,
        "critic_pass_rate": round(critic_pass_count / total, 4) if total else 0,
        "mean_critic_score": mean_valid(rows, "critic_score"),
        "mean_critic_normalized": mean_valid(rows, "critic_normalized"),
        "mean_critic_applicable_normalized": mean_valid(
            rows,
            "critic_applicable_normalized",
        ),

        "structure_complete_count": structure_complete_count,
        "structure_complete_rate": (
            round(structure_complete_count / total, 4) if total else 0
        ),
        "medical_warning_count": medical_warning_count,
        "medical_warning_rate": (
            round(medical_warning_count / total, 4) if total else 0
        ),
        "total_iterations": total_iterations,
        "mean_iterations": mean_valid(rows, "iterations"),

        "n_scored_for_text_metrics": total - sum(skipped.values()),
        "n_skipped_for_text_metrics": skipped,
        "mean_n_contexts": mean_valid(rows, "n_contexts"),

        "mean_rouge1": mean_valid(rows, "rouge1"),
        "mean_rouge2": mean_valid(rows, "rouge2"),
        "mean_rougeL": mean_valid(rows, "rougeL"),
        "mean_bleu": mean_valid(rows, "bleu"),
        "mean_meteor": mean_valid(rows, "meteor"),
        "mean_bertscore": mean_valid(rows, "bertscore"),

        "mean_ragas_faithfulness": mean_valid(rows, "ragas_faithfulness"),
        "mean_ragas_faithfulness_full": mean_valid(
            rows,
            "ragas_faithfulness_full",
        ),
        "mean_ragas_answer_relevancy": mean_valid(rows, "ragas_answer_relevancy"),
        "mean_ragas_context_precision": mean_valid(rows, "ragas_context_precision"),
        "mean_ragas_context_recall": mean_valid(rows, "ragas_context_recall"),
        "mean_ragas_factual_correctness": mean_valid(rows, "ragas_factual_correctness"),

        "n_ragas_errors": sum(len(row.get("ragas_errors") or {}) for row in rows),
        "n_confidence_success": len(confidence_success_rows),
        "mean_confidence": mean_valid(
            confidence_success_rows,
            "confidence",
        ),
        "mean_entropy": mean_valid(rows, "entropy"),
        "mean_latency_seconds": mean_valid(rows, "latency_seconds"),
        "mean_source_recall": mean_valid(rows, "source_recall"),
    }


# ============================================================
# SAUVEGARDE
# ============================================================

def save_results(rows: list[dict], summary: dict) -> None:
    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if not rows:
        return

    fieldnames = list(
        dict.fromkeys(
            key
            for row in rows
            for key in row
        )
    )
    csv_rows = []
    for row in rows:
        clean = {}
        for key in fieldnames:
            value = row.get(key)
            clean[key] = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
        csv_rows.append(clean)

    with open(CSV_FILE, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 80)
    print("ÉVALUATION COMPLÈTE : métriques texte + RAGAS")
    print("=" * 80)
    if RAGAS_ENABLED:
        print(f"RAGAS activé : juge={RAGAS_LLM_MODEL} (via Gemini), "
              f"embeddings={RAGAS_EMBEDDING_MODEL}")
        print("Coût attendu : plusieurs centaines d'appels Gemini supplémentaires "
              "pour les 5 métriques RAGAS. Voir le docstring en tête de fichier.")
    else:
        print("RAGAS désactivé (RAGAS_ENABLED=0) : métriques texte uniquement.")

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)
    verify_frozen_references(questions)
    print("Nombre de questions :", len(questions))
    

    if JSON_FILE.exists():
        try:
            with open(JSON_FILE, "r", encoding="utf-8") as f:
                results = json.load(f)
            if not isinstance(results, list):
                results = []
        except Exception as exc:
            print("[AVERTISSEMENT] Impossible de relire les résultats précédents :", exc)
            results = []
    else:
        results = []

    completed_ids = {
        row.get("id") for row in results
        if row.get("publication_status") in {"PASS", "REJECTED"}
        and row.get("evaluation_protocol_version") == EVALUATION_PROTOCOL_VERSION
        and "metrics_reference_direct" in row
        and "ragas_faithfulness_full" in row
    }
    if completed_ids:
        print("[REPRISE] Questions déjà terminées :", sorted(completed_ids))

    ragas_metrics = smoke_test_ragas(build_ragas_metrics())

    for item in questions:
        question_id = item["id"]
        if question_id in completed_ids:
            print(f"[SKIP] Question {question_id} déjà terminée.")
            continue

        row = evaluate_question_full(item, ragas_metrics)

        results = [old for old in results if old.get("id") != question_id]
        results.append(row)
        results.sort(key=lambda x: x.get("id", 999999))

        add_bertscore(results)
        save_results(results, build_summary(results))

        if row.get("publication_status") in {"PASS", "REJECTED"}:
            completed_ids.add(question_id)

    add_bertscore(results)
    final_summary = build_summary(results)
    save_results(results, final_summary)

    print("\n" + "=" * 80)
    print("RÉSUMÉ FINAL")
    print("=" * 80)
    for key, value in final_summary.items():
        print(f"{key}: {value}")

    print("\nRésultats JSON :", JSON_FILE)
    print("Résultats CSV  :", CSV_FILE)
    print("Résumé JSON    :", SUMMARY_FILE)


if __name__ == "__main__":
    main()
