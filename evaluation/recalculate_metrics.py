

from __future__ import annotations
import json
import re
import statistics
import unicodedata
from pathlib import Path

# Important sous Windows : charger PyTorch avant transformers / bert_score.
# Cela évite certains échecs d'initialisation de DLL (WinError 1114).
try:
    import torch  # noqa: F401
except Exception as _torch_import_error:
    print(f"Avertissement : import torch impossible ({_torch_import_error!r}).")

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score

try:
    from evaluation.reference_lock import verify_frozen_references
except ModuleNotFoundError:  # exécution directe depuis le dossier evaluation/
    from reference_lock import verify_frozen_references

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_FILE = PROJECT_ROOT / "evaluation" / "questions.json"
RESULTS_FILE = PROJECT_ROOT / "evaluation" / "results" / "evaluation_results.json"
OUTPUT_FILE = PROJECT_ROOT / "evaluation" / "results" / "evaluation_results_with_metrics.json"

BERTSCORE_LANG = "fr"


# ---------------------------------------------------------------------------
# Ressources NLTK (téléchargées une fois si absentes)
# ---------------------------------------------------------------------------

def ensure_nltk_resources() -> bool:
    """Télécharge wordnet/omw-1.4 si besoin. Renvoie False si indisponibles
    (le script continue alors, mais meteor_score sera désactivé)."""
    import nltk

    for resource, path in (("wordnet", "corpora/wordnet"), ("omw-1.4", "corpora/omw-1.4")):
        found = False
        for candidate in (path, path + ".zip"):
            try:
                nltk.data.find(candidate)
                found = True
                break
            except LookupError:
                pass

        if not found:
            print(f"Ressource NLTK '{resource}' absente, téléchargement...")
            try:
                nltk.download(resource, quiet=True)
            except Exception as exc:
                print(f"  -> échec du téléchargement de '{resource}' ({exc}). "
                      "METEOR sera désactivé pour cette exécution.")
                return False
    return True


METEOR_AVAILABLE = ensure_nltk_resources()


# ---------------------------------------------------------------------------
# Extraction de la « Réponse directe » (texte réellement comparable)
# ---------------------------------------------------------------------------

_DIRECT_ANSWER_HEADING = re.compile(
    r"\*{0,2}\s*1\.\s*Réponse directe\s*\*{0,2}\s*:?\s*\n+", re.IGNORECASE
)
_NEXT_SECTION = re.compile(
    r"\n\s*\*{0,2}\d+\.\s|\n\s*Traçabilité", re.IGNORECASE
)


def extract_direct_answer(final_answer: str) -> str | None:
    """Isole la section « 1. Réponse directe » d'un rapport structuré.

    Renvoie None si la section n'est pas trouvée (format inattendu) : dans
    ce cas la ligne est exclue des métriques plutôt que comparée en entier.
    """
    match = _DIRECT_ANSWER_HEADING.search(final_answer or "")
    if not match:
        return None

    rest = final_answer[match.end():]
    next_section = _NEXT_SECTION.search(rest)
    excerpt = rest[: next_section.start()] if next_section else rest
    excerpt = excerpt.strip()
    return excerpt or None


# ---------------------------------------------------------------------------
# Tokenisation (BLEU / METEOR uniquement ; ROUGE tokenise en interne)
# ---------------------------------------------------------------------------

_WORD = re.compile(r"[^\W\d_](?:[^\W_]|[-'])*", re.UNICODE)


def normalize_unicode(text: str) -> str:
    """Uniformise les variantes typographiques que le LLM introduit.

    Sans cela, « auriculo‑ventriculaire » (tiret insécable U+2011, sortie du
    LLM) et « auriculo-ventriculaire » (tiret ASCII, reference_answer) sont
    deux mots différents pour BLEU/METEOR, qui ne matchent jamais.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.translate({
        # NFKC ramène U+2011 (tiret insécable) à U+2010, jamais à '-' ASCII :
        # les deux doivent donc être mappés explicitement, après NFKC.
        0x2010: "-", 0x2011: "-", 0x2013: "-", 0x2014: "-",
        0x00A0: " ", 0x202F: " ",                  # espaces insécable / fine insécable
        0x2018: "'", 0x2019: "'",                  # apostrophes typographiques
    })
    return text


def tokenize(text) -> list[str]:
    return _WORD.findall(normalize_unicode(str(text)).lower())


# ---------------------------------------------------------------------------
# Métriques individuelles
# ---------------------------------------------------------------------------

_rouge_scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=False,   # le stemmer par défaut est anglais (Porter) : sans effet
)                        # utile sur du français, on ne feint pas de le faire.


def calculate_rouge(reference: str, prediction: str) -> dict[str, float]:
    scores = _rouge_scorer.score(reference, prediction)
    return {name: scores[name].fmeasure for name in ("rouge1", "rouge2", "rougeL")}


def calculate_bleu(reference: str, prediction: str) -> float | None:
    ref, pred = tokenize(reference), tokenize(prediction)
    if not ref or not pred:
        return None
    return sentence_bleu([ref], pred, smoothing_function=SmoothingFunction().method1)


def calculate_meteor(reference: str, prediction: str) -> float | None:
    if not METEOR_AVAILABLE:
        return None
    ref, pred = tokenize(reference), tokenize(prediction)
    if not ref or not pred:
        return None
    return meteor_score([ref], pred)


def calculate_bertscore_batch(references: list[str], predictions: list[str]) -> list[float] | None:
    """Un seul appel pour toutes les lignes. None si le modèle est inaccessible."""
    if not references:
        return []
    try:
        from bert_score import score as bert_score
        _, _, f1 = bert_score(predictions, references, lang=BERTSCORE_LANG, verbose=False)
        return [float(v) for v in f1.tolist()]
    except Exception as exc:
        print(f"\nBERTScore indisponible ({exc!r}).")
        print("mean_bertscore sera 'null' pour cette exécution ; les autres "
              "métriques (ROUGE/BLEU/METEOR) restent valides.\n")
        return None


# ---------------------------------------------------------------------------
# Calcul sur un jeu de résultats déjà chargé (pur, testable sans fichiers)
# ---------------------------------------------------------------------------

def score_results(results: list[dict], questions: list[dict]) -> dict:
    """Calcule les métriques sur ``results`` et renvoie le résumé.

    Modifie ``results`` en place (ajoute rouge1/rouge2/rougeL/bleu/meteor/
    bertscore/metrics_reference_excerpt/metrics_skipped_reason à chaque ligne),
    comme le fait le script complet.
    """
    questions_by_id = {q["id"]: q for q in questions}

    skipped_counts: dict[str, int] = {}
    failed_metrics: dict[str, int] = {}
    pending_bertscore: list[tuple[dict, str, str]] = []   # (row, reference, excerpt)

    for row in results:
        question = questions_by_id.get(row["id"], {})
        reference_full = (question.get("reference_answer") or "").strip()
        reference = extract_direct_answer(reference_full) or reference_full

        row["metrics_reference_excerpt"] = None
        row["metrics_skipped_reason"] = None

        if row.get("publication_status") == "REJECTED":
            row["metrics_skipped_reason"] = "rejected_publication"
            skipped_counts["rejected_publication"] = skipped_counts.get("rejected_publication", 0) + 1
            continue

        if not reference:
            row["metrics_skipped_reason"] = "missing_reference"
            skipped_counts["missing_reference"] = skipped_counts.get("missing_reference", 0) + 1
            continue

        excerpt = extract_direct_answer(row.get("final_answer", ""))
        if excerpt is None:
            row["metrics_skipped_reason"] = "no_extractable_section"
            skipped_counts["no_extractable_section"] = skipped_counts.get("no_extractable_section", 0) + 1
            continue

        row["metrics_reference_excerpt"] = excerpt

        try:
            rouge = calculate_rouge(reference, excerpt)
            row["rouge1"] = round(rouge["rouge1"], 4)
            row["rouge2"] = round(rouge["rouge2"], 4)
            row["rougeL"] = round(rouge["rougeL"], 4)
        except Exception as exc:
            failed_metrics["rouge"] = failed_metrics.get("rouge", 0) + 1
            print(f"  [id {row['id']}] ROUGE a échoué : {exc!r}")

        try:
            bleu = calculate_bleu(reference, excerpt)
            row["bleu"] = round(bleu, 4) if bleu is not None else None
        except Exception as exc:
            failed_metrics["bleu"] = failed_metrics.get("bleu", 0) + 1
            print(f"  [id {row['id']}] BLEU a échoué : {exc!r}")

        try:
            meteor = calculate_meteor(reference, excerpt)
            row["meteor"] = round(meteor, 4) if meteor is not None else None
        except Exception as exc:
            failed_metrics["meteor"] = failed_metrics.get("meteor", 0) + 1
            print(f"  [id {row['id']}] METEOR a échoué : {exc!r}")

        pending_bertscore.append((row, reference, excerpt))

    # -----------------------------------------------------------------------
    # BERTScore : un seul appel groupé pour toutes les lignes éligibles
    # -----------------------------------------------------------------------

    if pending_bertscore:
        references = [reference for _, reference, _ in pending_bertscore]
        excerpts = [excerpt for _, _, excerpt in pending_bertscore]
        scores = calculate_bertscore_batch(references, excerpts)

        for index, (row, _, _) in enumerate(pending_bertscore):
            row["bertscore"] = round(scores[index], 4) if scores is not None else None

    def mean_valid(key: str) -> float | None:
        values = [row[key] for row in results if isinstance(row.get(key), (int, float))]
        return round(statistics.mean(values), 4) if values else None

    return {
        "n_total": len(results),
        "n_scored": len(pending_bertscore),
        "n_skipped": dict(skipped_counts),
        "n_failed_metrics": dict(failed_metrics),
        "mean_rouge1": mean_valid("rouge1"),
        "mean_rouge2": mean_valid("rouge2"),
        "mean_rougeL": mean_valid("rougeL"),
        "mean_bleu": mean_valid("bleu"),
        "mean_meteor": mean_valid("meteor"),
        "mean_bertscore": mean_valid("bertscore"),
    }


# ---------------------------------------------------------------------------
# Point d'entrée : lecture des fichiers, calcul, écriture, affichage
# ---------------------------------------------------------------------------

def main() -> None:
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)
    verify_frozen_references(questions)
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)

    summary = score_results(results, questions)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)

    print("\nRÉSUMÉ DES MÉTRIQUES")
    print("=" * 60)
    print(f"Lignes évaluées      : {summary['n_scored']} / {summary['n_total']}")
    for reason, count in summary["n_skipped"].items():
        print(f"  exclues ({reason:<22s}): {count}")
    if summary["n_failed_metrics"]:
        print("Échecs ponctuels par métrique :", summary["n_failed_metrics"])
    print("-" * 60)
    for key in ("mean_rouge1", "mean_rouge2", "mean_rougeL", "mean_bleu", "mean_meteor", "mean_bertscore"):
        print(f"{key}: {summary[key]}")

    print("\nFichier créé :", OUTPUT_FILE)


if __name__ == "__main__":
    main()
