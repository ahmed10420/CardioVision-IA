import torch

import csv
import json
import math
import re
import statistics
import time

from pathlib import Path

from orchestrator.langgraph_orchestrator import run_agentic_rag


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

QUESTIONS_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "questions.json"
)

RESULTS_FOLDER = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
)

RESULTS_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)

CSV_FILE = (
    RESULTS_FOLDER
    / "evaluation_results.csv"
)

JSON_FILE = (
    RESULTS_FOLDER
    / "evaluation_results.json"
)

SUMMARY_FILE = (
    RESULTS_FOLDER
    / "evaluation_summary.json"
)


# ============================================================
# MEDICAL WARNING
# ============================================================

MEDICAL_WARNING = (
    "Cette réponse constitue une aide à la décision clinique uniquement. "
    "Elle ne remplace en aucun cas le jugement d’un médecin qualifié. "
    "Toute décision diagnostique ou thérapeutique doit être validée "
    "par un professionnel de santé."
)


# ============================================================
# OPTIONAL METRIC PACKAGES
# ============================================================

try:
    from rouge_score import rouge_scorer

    ROUGE_AVAILABLE = True

except Exception:
    ROUGE_AVAILABLE = False


try:
    from nltk.translate.bleu_score import (
        sentence_bleu,
        SmoothingFunction,
    )

    BLEU_AVAILABLE = True

except Exception:
    BLEU_AVAILABLE = False


try:
    from nltk.translate.meteor_score import meteor_score

    METEOR_AVAILABLE = True

except Exception:
    METEOR_AVAILABLE = False


try:
    from bert_score import score as bert_score

    BERTSCORE_AVAILABLE = True

except Exception:
    BERTSCORE_AVAILABLE = False


# ============================================================
# TEXT HELPERS
# ============================================================

def normalize(text):
    return re.sub(
        r"\s+",
        " ",
        str(text or "").strip(),
    )


def tokenize(text):
    return re.findall(
        r"\b\w+\b",
        normalize(text).lower(),
        flags=re.UNICODE,
    )


# ============================================================
# ROUGE
# ============================================================

def calculate_rouge(reference, prediction):

    if not ROUGE_AVAILABLE:
        return {
            "rouge1": None,
            "rouge2": None,
            "rougeL": None,
        }

    scorer = rouge_scorer.RougeScorer(
        [
            "rouge1",
            "rouge2",
            "rougeL",
        ],
        use_stemmer=True,
    )

    scores = scorer.score(
        reference,
        prediction,
    )

    return {
        "rouge1": round(
            scores["rouge1"].fmeasure,
            4,
        ),
        "rouge2": round(
            scores["rouge2"].fmeasure,
            4,
        ),
        "rougeL": round(
            scores["rougeL"].fmeasure,
            4,
        ),
    }


# ============================================================
# BLEU
# ============================================================

def calculate_bleu(reference, prediction):

    if not BLEU_AVAILABLE:
        return None

    ref_tokens = tokenize(reference)
    pred_tokens = tokenize(prediction)

    if not ref_tokens or not pred_tokens:
        return 0.0

    smoothing = (
        SmoothingFunction()
        .method1
    )

    value = sentence_bleu(
        [ref_tokens],
        pred_tokens,
        smoothing_function=smoothing,
    )

    return round(
        float(value),
        4,
    )


# ============================================================
# METEOR
# ============================================================

def calculate_meteor(reference, prediction):

    if not METEOR_AVAILABLE:
        return None

    try:
        value = meteor_score(
            [tokenize(reference)],
            tokenize(prediction),
        )

        return round(
            float(value),
            4,
        )

    except Exception as exc:
        print(
            "[METEOR] indisponible :",
            exc,
        )

        return None


# ============================================================
# BERTSCORE
# ============================================================

def calculate_bertscore(reference, prediction):

    if not BERTSCORE_AVAILABLE:
        return None

    try:

        _, _, f1 = bert_score(
            [prediction],
            [reference],
            lang="fr",
            verbose=False,
        )

        return round(
            float(f1.mean().item()),
            4,
        )

    except Exception as exc:

        print(
            "[BERTScore] erreur :",
            exc,
        )

        return None


# ============================================================
# STRUCTURE
# ============================================================

def check_structure(answer):

    required_sections = [
        "1. Réponse directe",
        "2. Critères cliniques et ECG",
        "3. Justification documentaire",
        "4. Limites du contexte",
        "5. Avertissement médical",
    ]

    found = sum(
        section.lower()
        in answer.lower()
        for section
        in required_sections
    )

    return {
        "structure_score": found,
        "structure_max": len(
            required_sections
        ),
        "structure_complete": (
            found
            == len(required_sections)
        ),
    }


# ============================================================
# WARNING
# ============================================================

def check_warning(answer):

    return (
        normalize(MEDICAL_WARNING)
        in normalize(answer)
    )


# ============================================================
# SOURCE TRACEABILITY
# ============================================================

def check_expected_sources(
    answer,
    expected_sources,
):

    expected_sources = (
        expected_sources
        or []
    )

    if not expected_sources:
        return {
            "source_hits": 0,
            "source_total": 0,
            "source_recall": None,
            "missing_sources": [],
        }

    lower_answer = answer.lower()

    hits = []
    missing = []

    for source in expected_sources:

        if str(source).lower() in lower_answer:
            hits.append(source)

        else:
            missing.append(source)

    recall = (
        len(hits)
        / len(expected_sources)
    )

    return {
        "source_hits": len(hits),
        "source_total": len(
            expected_sources
        ),
        "source_recall": round(
            recall,
            4,
        ),
        "missing_sources": missing,
    }


# ============================================================
# CRITIC SCORE NORMALIZATION
# ============================================================

def normalize_critic_score(
    critic_result,
):

    score = critic_result.get(
        "score",
        0,
    )

    maximum = critic_result.get(
        "max_score",
        7,
    )

    if not maximum:
        return 0.0

    return round(
        score / maximum,
        4,
    )


# ============================================================
# GROQ RATE LIMIT / ATTENTE AUTOMATIQUE
# ============================================================

def get_groq_wait_seconds(error_message: str) -> int:
    """Extrait un délai Groq du type 'Please try again in 39m50.688s'."""
    text = str(error_message or "")

    match = re.search(
        r"try again in\s+(?:(\d+)m)?([\d.]+)s",
        text,
        re.IGNORECASE,
    )

    if match:
        minutes = int(match.group(1) or 0)
        seconds = float(match.group(2) or 0)
        return int(minutes * 60 + seconds) + 10

    match_minutes = re.search(
        r"try again in\s+([\d.]+)m",
        text,
        re.IGNORECASE,
    )

    if match_minutes:
        return int(float(match_minutes.group(1)) * 60) + 10

    return 10 * 60


def is_groq_rate_limit_error(exc) -> bool:
    message = str(exc or "").lower()
    return (
        "429" in message
        or "rate_limit_exceeded" in message
        or "rate limit reached" in message
        or "tokens per day" in message
        or "tpd" in message
    )


# ============================================================
# ONE QUESTION
# ============================================================

def evaluate_question(item):

    question_id = item["id"]

    question = item[
        "question"
    ]

    reference = item.get(
        "reference_answer",
        "",
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"QUESTION {question_id}"
    )

    print(
        "=" * 80
    )

    print(question)

    start = time.perf_counter()

    while True:

        try:

            result = run_agentic_rag(
                question=question,
                ecg_input_path=item.get(
                    "ecg_input_path",
                    "",
                ),
                ecg_required=item.get(
                    "ecg_required",
                    False,
                ),
                test_force_improve=False,
            )

            error = None
            break

        except Exception as exc:

            error_message = str(exc)

            if is_groq_rate_limit_error(exc):

                wait_seconds = get_groq_wait_seconds(
                    error_message
                )

                wait_minutes = wait_seconds / 60.0

                print("\n" + "=" * 80)
                print("[GROQ RATE LIMIT]")
                print(
                    f"Question {question_id} mise en pause."
                )
                print(
                    f"Attente automatique : environ "
                    f"{wait_minutes:.1f} minute(s)."
                )
                print(
                    "Le programme reprendra automatiquement "
                    "la même question après l'attente."
                )
                print("=" * 80)

                time.sleep(wait_seconds)

                print(
                    f"\n[REPRISE] Nouvelle tentative "
                    f"de la question {question_id}..."
                )

                continue

            print(
                f"\n[ERREUR] Question {question_id}"
            )
            print(type(exc).__name__)
            print(error_message)

            result = {}
            error = error_message
            break

    latency = (
        time.perf_counter()
        - start
    )

    final_answer = result.get(
        "final_answer",
        "",
    )

    critic = result.get(
        "final_critic_result",
        {},
    )

    confidence = result.get(
        "confidence_result",
        {},
    )

    # ========================================================
    # LANGUAGE METRICS
    # ========================================================

    if reference and final_answer:

        rouge = calculate_rouge(
            reference,
            final_answer,
        )

        bleu = calculate_bleu(
            reference,
            final_answer,
        )

        meteor = calculate_meteor(
            reference,
            final_answer,
        )

        bert = calculate_bertscore(
            reference,
            final_answer,
        )

    else:

        rouge = {
            "rouge1": None,
            "rouge2": None,
            "rougeL": None,
        }

        bleu = None
        meteor = None
        bert = None

    structure = check_structure(
        final_answer
    )

    source_eval = check_expected_sources(
        final_answer,
        item.get(
            "expected_sources",
            [],
        ),
    )

    row = {

        "id": question_id,

        "category": item.get(
            "category",
            "",
        ),

        "question": question,

        "reference_answer": reference,

        "final_answer": final_answer,

        # ----------------------------
        # SYSTEM
        # ----------------------------

        "latency_seconds": round(
            latency,
            3,
        ),

        "publication_status": (
            result.get(
                "publication_status",
                "ERROR",
            )
        ),

        "error": error,

        # ----------------------------
        # CRITIC
        # ----------------------------

        "critic_verdict": critic.get(
            "verdict",
        ),

        "critic_score": critic.get(
            "score",
        ),

        "critic_max_score": critic.get(
            "max_score",
            7,
        ),

        "critic_normalized": (
            normalize_critic_score(
                critic
            )
        ),

        "critic_applicable_score": critic.get(
            "applicable_score",
        ),

        "critic_applicable_max_score": critic.get(
            "applicable_max_score",
        ),

        "critic_applicable_normalized": critic.get(
            "applicable_normalized",
        ),

        "critic_feedback": critic.get(
            "feedback",
            [],
        ),

        # ----------------------------
        # NVIDIA
        # ----------------------------

        "confidence": confidence.get(
            "confidence",
        ),

        "entropy": confidence.get(
            "entropy",
        ),

        "confidence_status": (
            confidence.get(
                "status"
            )
        ),

        "confidence_error": (
            confidence.get(
                "error"
            )
        ),

        "confidence_exact_match": (
            confidence.get(
                "exact_match"
            )
        ),

        "tokens_analyzed": (
            confidence.get(
                "tokens_analyzed"
            )
        ),

        # ----------------------------
        # RAG
        # ----------------------------

        "ptbxl_ids": result.get(
            "ptbxl_case_ids",
            [],
        )[:3],

        "web_used": result.get(
            "web_search_used",
            False,
        ),

        "web_source": result.get(
            "web_source",
            "",
        ),

        "iterations": result.get(
            "iterations",
            0,
        ),

        # ----------------------------
        # GENERATION METRICS
        # ----------------------------

        "rouge1": rouge[
            "rouge1"
        ],

        "rouge2": rouge[
            "rouge2"
        ],

        "rougeL": rouge[
            "rougeL"
        ],

        "bleu": bleu,

        "meteor": meteor,

        "bertscore": bert,

        # ----------------------------
        # SAFETY / STRUCTURE
        # ----------------------------

        "structure_score": (
            structure[
                "structure_score"
            ]
        ),

        "structure_complete": (
            structure[
                "structure_complete"
            ]
        ),

        "medical_warning": (
            check_warning(
                final_answer
            )
        ),

        # ----------------------------
        # SOURCE TRACEABILITY
        # ----------------------------

        "source_hits": (
            source_eval[
                "source_hits"
            ]
        ),

        "source_total": (
            source_eval[
                "source_total"
            ]
        ),

        "source_recall": (
            source_eval[
                "source_recall"
            ]
        ),

        "missing_sources": (
            source_eval[
                "missing_sources"
            ]
        ),
    }

    print(
        "Publication :",
        row[
            "publication_status"
        ],
    )

    print(
        "Critic :",
        row[
            "critic_verdict"
        ],
        row[
            "critic_score"
        ],
        "/",
        row[
            "critic_max_score"
        ],
    )

    print(
        "BERTScore :",
        row["bertscore"],
    )

    print(
        "Confidence :",
        row["confidence"],
    )

    print(
        "Entropy :",
        row["entropy"],
    )

    print(
        "NVIDIA status :",
        row["confidence_status"],
    )

    print(
        "NVIDIA error :",
        row["confidence_error"],
    )

    print(
        "Latence :",
        row[
            "latency_seconds"
        ],
        "s",
    )

    return row


# ============================================================
# AVERAGE
# ============================================================

def mean_valid(
    rows,
    key,
):

    values = [
        row[key]
        for row in rows
        if isinstance(
            row.get(key),
            (int, float),
        )
        and not isinstance(
            row.get(key),
            bool,
        )
    ]

    if not values:
        return None

    return round(
        statistics.mean(
            values
        ),
        4,
    )


# ============================================================
# SUMMARY
# ============================================================

def build_summary(rows):

    total = len(rows)

    confidence_success_rows = [
        row
        for row in rows
        if row.get("confidence_status") == "success"
    ]

    pass_count = sum(
        row[
            "publication_status"
        ] == "PASS"
        for row in rows
    )

    critic_pass_count = sum(
        row[
            "critic_verdict"
        ] == "PASS"
        for row in rows
    )

    structure_pass = sum(
        bool(
            row[
                "structure_complete"
            ]
        )
        for row in rows
    )

    warning_pass = sum(
        bool(
            row[
                "medical_warning"
            ]
        )
        for row in rows
    )

    exact_confidence = sum(
        row.get(
            "confidence_exact_match"
        )
        is True
        for row in rows
    )

    summary = {

        "questions": total,

        # ----------------------------
        # PASS
        # ----------------------------

        "publication_pass_count": (
            pass_count
        ),

        "publication_pass_rate": (
            round(
                pass_count / total,
                4,
            )
            if total
            else 0
        ),

        "critic_pass_count": (
            critic_pass_count
        ),

        "critic_pass_rate": (
            round(
                critic_pass_count
                / total,
                4,
            )
            if total
            else 0
        ),

        "mean_critic_normalized": mean_valid(
            rows,
            "critic_normalized",
        ),

        "mean_critic_applicable_normalized": mean_valid(
            rows,
            "critic_applicable_normalized",
        ),

        # ----------------------------
        # GENERATION
        # ----------------------------

        "mean_rouge1": mean_valid(
            rows,
            "rouge1",
        ),

        "mean_rouge2": mean_valid(
            rows,
            "rouge2",
        ),

        "mean_rougeL": mean_valid(
            rows,
            "rougeL",
        ),

        "mean_bleu": mean_valid(
            rows,
            "bleu",
        ),

        "mean_meteor": mean_valid(
            rows,
            "meteor",
        ),

        "mean_bertscore": (
            mean_valid(
                rows,
                "bertscore",
            )
        ),

        # ----------------------------
        # NVIDIA
        # ----------------------------

        "mean_confidence": (
            mean_valid(
                confidence_success_rows,
                "confidence",
            )
        ),

        "n_confidence_success": len(
            confidence_success_rows
        ),

        "mean_entropy": (
            mean_valid(
                rows,
                "entropy",
            )
        ),

        "confidence_exact_match_rate": (
            round(
                exact_confidence
                / total,
                4,
            )
            if total
            else 0
        ),

        # ----------------------------
        # PERFORMANCE
        # ----------------------------

        "mean_latency_seconds": (
            mean_valid(
                rows,
                "latency_seconds",
            )
        ),

        # ----------------------------
        # SAFETY
        # ----------------------------

        "structure_complete_rate": (
            round(
                structure_pass
                / total,
                4,
            )
            if total
            else 0
        ),

        "medical_warning_rate": (
            round(
                warning_pass
                / total,
                4,
            )
            if total
            else 0
        ),

        "mean_source_recall": (
            mean_valid(
                rows,
                "source_recall",
            )
        ),
    }

    return summary


# ============================================================
# SAVE
# ============================================================

def save_results(
    rows,
    summary,
):

    with open(
        JSON_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            rows,
            file,
            ensure_ascii=False,
            indent=2,
        )

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    csv_rows = []

    for row in rows:

        clean = {}

        for key, value in row.items():

            if isinstance(
                value,
                (list, dict),
            ):
                clean[key] = json.dumps(
                    value,
                    ensure_ascii=False,
                )

            else:
                clean[key] = value

        csv_rows.append(
            clean
        )

    if csv_rows:

        with open(
            CSV_FILE,
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as file:

            writer = (
                csv.DictWriter(
                    file,
                    fieldnames=csv_rows[
                        0
                    ].keys(),
                )
            )

            writer.writeheader()

            writer.writerows(
                csv_rows
            )


# ============================================================
# MAIN
# ============================================================

def main():

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        questions = json.load(file)

    print(
        "Nombre de questions :",
        len(questions),
    )

    # Reprendre les résultats déjà enregistrés s'ils existent.
    if JSON_FILE.exists():
        try:
            with open(
                JSON_FILE,
                "r",
                encoding="utf-8",
            ) as file:
                results = json.load(file)

            if not isinstance(results, list):
                results = []

            # Compatibilité avec les benchmarks créés avant l'ajout du
            # diagnostic NVIDIA dans les sorties JSON et CSV.
            for row in results:
                if isinstance(row, dict):
                    row.setdefault(
                        "confidence_error",
                        None,
                    )

        except Exception as exc:
            print(
                "[AVERTISSEMENT] Impossible de relire les anciens résultats :",
                exc,
            )
            results = []
    else:
        results = []

    # PASS et REJECTED = questions réellement terminées.
    # ERROR = erreur technique, donc la question sera retentée.
    completed_ids = {
        row.get("id")
        for row in results
        if row.get("publication_status") in {
            "PASS",
            "REJECTED",
        }
    }

    if completed_ids:
        print(
            "[REPRISE] Questions déjà terminées :",
            sorted(completed_ids),
        )

    for item in questions:

        question_id = item["id"]

        if question_id in completed_ids:
            print(
                f"[SKIP] Question {question_id} déjà terminée."
            )
            continue

        row = evaluate_question(item)

        # Remplacer une éventuelle ancienne ligne ERROR du même ID.
        results = [
            old_row
            for old_row in results
            if old_row.get("id") != question_id
        ]

        results.append(row)
        results.sort(key=lambda x: x.get("id", 999999))

        # Sauvegarde après chaque question.
        summary = build_summary(results)
        save_results(results, summary)

        if row.get("publication_status") in {
            "PASS",
            "REJECTED",
        }:
            completed_ids.add(question_id)

    final_summary = build_summary(results)
    save_results(results, final_summary)

    print("\n" + "=" * 80)
    print("RÉSUMÉ FINAL")
    print("=" * 80)

    for key, value in final_summary.items():
        print(f"{key}: {value}")

    print("\nRésultats CSV :", CSV_FILE)
    print("Résultats JSON :", JSON_FILE)
    print("Résumé JSON :", SUMMARY_FILE)


if __name__ == "__main__":
    main()
