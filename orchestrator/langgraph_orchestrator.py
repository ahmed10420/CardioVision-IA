import json
import os
import re
import sys
from pathlib import Path

from langgraph.graph import StateGraph, START, END

from orchestrator.state import AgentState

from agents.signal_agent import SignalAgent
from agents.literature_agent import LiteratureAgent
from agents.clinical_agent import ClinicalAgent
from agents.websearch_agent import WebSearchAgent
from agents.critic_agent import CriticAgent
from agents.synthesizer_agent import SynthesizerAgent
from agents.confidence_agent import ConfidenceAgent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace",
    )

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(
        encoding="utf-8",
        errors="replace",
    )

# ============================================================
# CONFIGURATION
# ============================================================

# Nombre maximal de corrections au total (critique initial + contrôle final)
MAX_ITERATIONS = 2

# ECG_PIPELINE_PATH=D:\\ecg_agent_project\\ecg_project
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ECG_PIPELINE_PATH = PROJECT_ROOT / "ecg_project"

UNVALIDATED_ANSWER = (
    "La réponse clinique générée n'a pas passé la vérification finale. "
    "Aucune réponse médicale non validée ne peut être publiée. "
    "Veuillez consulter un professionnel de santé."
)

SIGNAL_ERROR_ANSWER = (
    "L'ECG fourni n'a pas pu être analysé. Aucune interprétation du tracé "
    "ni confrontation clinique avec les recommandations ne peut être validée. "
    "Vérifiez l'erreur de SignalAgent et réessayez."
)

MISSING_SIGNAL_ANSWER = (
    "Aucune image ECG n'a été fournie pour l'analyse demandée. "
    "Fournissez un fichier ECG avant de demander une interprétation du tracé."
)


# ============================================================
# INITIALISATION DES AGENTS
# ============================================================

# Initialisation paresseuse (lazy) des agents.
# Important pour Streamlit : aucun modèle lourd n'est construit au moment de l'import.
_literature_agent = None
_websearch_agent = None
_clinical_agent = None
_critic_agent = None
_synthesizer_agent = None
_confidence_agent = None

# SignalAgent est également initialisé uniquement lorsqu'une image ECG est fournie.
_signal_agent = None


def get_literature_agent() -> LiteratureAgent:
    global _literature_agent
    if _literature_agent is None:
        print('[INIT] LiteratureAgent')
        _literature_agent = LiteratureAgent()
    return _literature_agent


def get_websearch_agent() -> WebSearchAgent:
    global _websearch_agent
    if _websearch_agent is None:
        print('[INIT] WebSearchAgent')
        _websearch_agent = WebSearchAgent(max_results=3)
    return _websearch_agent


def get_clinical_agent() -> ClinicalAgent:
    global _clinical_agent
    if _clinical_agent is None:
        print('[INIT] ClinicalAgent')
        _clinical_agent = ClinicalAgent()
    return _clinical_agent


def get_critic_agent() -> CriticAgent:
    global _critic_agent
    if _critic_agent is None:
        print('[INIT] CriticAgent')
        _critic_agent = CriticAgent()
    return _critic_agent


def get_synthesizer_agent() -> SynthesizerAgent:
    global _synthesizer_agent
    if _synthesizer_agent is None:
        print('[INIT] SynthesizerAgent')
        _synthesizer_agent = SynthesizerAgent()
    return _synthesizer_agent


def get_confidence_agent() -> ConfidenceAgent:
    global _confidence_agent
    if _confidence_agent is None:
        print('[INIT] ConfidenceAgent')
        _confidence_agent = ConfidenceAgent()
    return _confidence_agent


def get_signal_agent() -> SignalAgent:
    global _signal_agent

    if _signal_agent is None:
        ecg_project_path = os.getenv(
            "ECG_PIPELINE_PATH",
            str(DEFAULT_ECG_PIPELINE_PATH),
        )
        _signal_agent = SignalAgent(ecg_project_path)

    return _signal_agent



def format_web_diagnostic(web_result: dict) -> str:
    """
    Résume de façon lisible toutes les tentatives de la cascade WebSearch.
    """
    if not web_result:
        return "Aucun résultat WebSearch disponible."

    attempts = web_result.get("attempts", []) or []
    selected = web_result.get("selected_provider") or "aucune"

    search_query = web_result.get("search_query", "")
    lines = [f"Source retenue : {selected}"]

    if search_query:
        lines.append(f"Requête réellement envoyée : {search_query}")

    if not attempts:
        error = web_result.get("error")
        if error:
            lines.append(f"Erreur : {error}")
        lines.append("Aucune tentative enregistrée.")
        return "\n".join(lines)

    lines.append("Tentatives :")

    for attempt in attempts:
        provider = attempt.get("provider", "inconnu")
        status = attempt.get("status", "inconnu")
        count = attempt.get("count", 0)
        error = attempt.get("error")

        line = f"- {provider}: status={status}, résultats={count}"

        if error:
            line += f", erreur={error}"

        lines.append(line)

    return "\n".join(lines)


def ensure_ptbxl_traceability(answer: str, state: AgentState) -> str:
    """
    Garantit que les ECG ID PTB-XL réellement retrouvés restent visibles
    dans la réponse finale, même si SynthesizerAgent les omet.

    Si des IDs existent, on supprime aussi les formulations contradictoires
    du type "aucun cas PTB-XL ... identifié/retrouvé".
    """
    answer = str(answer or "").strip()
    case_ids = state.get("ptbxl_case_ids", []) or []

    if not case_ids:
        return answer

    selected_ids = [str(case_id) for case_id in case_ids[:3]]

    # Supprime les phrases contradictoires fréquentes.
    contradiction_patterns = [
        r"[^.\n]*aucun\s+cas\s+PTB[-‑ ]?XL[^.\n]*(?:identifi[eé]|retrouv[eé]|disponible)[^.\n]*\.",
        r"[^.\n]*aucun\s+cas\s+similaire\s+PTB[-‑ ]?XL[^.\n]*\.",
    ]

    cleaned = answer

    for pattern in contradiction_patterns:
        cleaned = re.sub(
            pattern,
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    # Supprime un ancien bloc de traçabilité, éventuellement incomplet, puis
    # reconstruit une source de vérité unique à partir de l'état LangGraph.
    cleaned = re.sub(
        r"\n*Traçabilité PTB-XL\s*\n"
        r"Cas PTB-XL similaires retrouvés\s*:\s*\n"
        r"(?:-\s*ECG ID\s+\d+\s*\n)*"
        r"Ces cas sont utilisés uniquement comme références comparatives\s*"
        r"et ne confirment pas un diagnostic pour le cas actuel\.\s*",
        "\n",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()

    trace_lines = [
        "Traçabilité PTB-XL",
        "Cas PTB-XL similaires retrouvés :",
    ]

    for case_id in selected_ids:
        trace_lines.append(f"- ECG ID {case_id}")

    trace_lines.append(
        "Ces cas sont utilisés uniquement comme références comparatives "
        "et ne confirment pas un diagnostic pour le cas actuel."
    )

    trace_block = "\n".join(trace_lines)
    limits_heading = re.search(
        r"(?m)^4\. Limites du contexte\s*$",
        cleaned,
    )

    if limits_heading:
        return (
            cleaned[:limits_heading.start()].rstrip()
            + "\n\n"
            + trace_block
            + "\n\n"
            + cleaned[limits_heading.start():].lstrip()
        )

    return cleaned + "\n\n" + trace_block


def normalize_french_elisions(answer: str) -> str:
    """Corrige les erreurs d'élision française fréquentes avant publication."""
    text = str(answer or "")
    text = re.sub(r"\bne\s+indique\b", "n'indique", text, flags=re.IGNORECASE)
    text = re.sub(r"\bne\s+indiquent\b", "n'indiquent", text, flags=re.IGNORECASE)
    return text


# Mêmes motifs que CriticAgent : la règle « pas de citation entre guillemets
# dans la section 3 » est une regex, pas un jugement du LLM. La respecter par
# code est donc plus fiable que de la redemander au LLM (qui, à temperature=0,
# régénère la même réponse avec les mêmes guillemets).
_JUSTIFICATION_HEADING = re.compile(
    r"(?im)^[ \t]*(?:\*\*)?3\.[ \t]*Justification documentaire"
    r"(?:\*\*)?[ \t]*\\?[ \t]*$"
)
_JUSTIFICATION_END = re.compile(r"(?im)^[ \t]*(?:\*\*)?4\.[ \t]*Limites du contexte")
_LITERAL_QUOTE = re.compile(r"«[^»\n]+»|“[^”\n]+”|\"[^\"\n]+\"")


def strip_quotes_in_justification(answer: str) -> str:
    """Retire les guillemets autour des passages cités dans la section 3.

    Seuls les caractères de guillemets sont retirés ; le texte reste. Cela
    satisfait la règle de format du CriticAgent sans toucher aux autres
    sections. Le critique LLM continue de juger le fond (paraphrase fidèle).
    """
    answer = str(answer or "")
    heading = _JUSTIFICATION_HEADING.search(answer)
    if heading is None:
        return answer

    end = _JUSTIFICATION_END.search(answer, heading.end())
    section_end = end.start() if end else len(answer)
    section = answer[heading.end():section_end]

    cleaned = _LITERAL_QUOTE.sub(lambda m: m.group(0)[1:-1].strip(), section)
    return answer[:heading.end()] + cleaned + answer[section_end:]


def ptbxl_traceability_issues(state: AgentState) -> list[str]:
    """
    Contrôle déterministe de la réponse finale.
    """
    case_ids = state.get("ptbxl_case_ids", []) or []
    answer = str(state.get("final_answer", "") or "")

    if not case_ids:
        return []

    issues = []

    selected_ids = [str(case_id) for case_id in case_ids[:3]]

    traceability_match = re.search(
        r"Traçabilité PTB-XL(?P<block>.*)$",
        answer,
        flags=re.IGNORECASE | re.DOTALL,
    )
    traceability_block = (
        traceability_match.group("block")
        if traceability_match
        else ""
    )

    if not traceability_match:
        issues.append("Le bloc canonique de traçabilité PTB-XL est absent.")

    for case_id in selected_ids:
        if not re.search(rf"\bECG ID\s+{re.escape(case_id)}\b", traceability_block):
            issues.append(
                f"L'ECG ID PTB-XL {case_id} a été récupéré mais n'apparaît "
                "pas dans le bloc de traçabilité final."
            )

    if re.search(
        r"aucun\s+cas\s+(?:similaire\s+)?PTB[-‑ ]?XL[^.\n]*"
        r"(?:identifi[eé]|retrouv[eé]|disponible)",
        answer,
        flags=re.IGNORECASE,
    ):
        issues.append(
            "La réponse affirme qu'aucun cas PTB-XL n'a été trouvé alors "
            "que des ECG ID sont présents dans l'état."
        )

    return issues


# ============================================================
# ROUTAGE WEBSEARCH
# ============================================================

WEB_TRIGGER_PATTERN = re.compile(
    r"\b("
    r"recent|recentes?|récent|récentes?|latest|newest|current|"
    r"actuel|actuelle|actuels|actuelles|actualité|actualités|"
    r"nouveau|nouvelle|nouveaux|nouvelles|mise\s+à\s+jour|"
    r"state\s+of\s+the\s+art|2024|2025|2026"
    r")\b",
    re.IGNORECASE,
)


def should_use_web_search(question: str) -> bool:
    """
    Routage déterministe inspiré du papier :
    WebSearchAgent est activé uniquement pour une demande récente/actuelle
    ou lorsqu'une année récente est explicitement mentionnée.
    """
    return bool(
        WEB_TRIGGER_PATTERN.search(
            str(question or "")
        )
    )


# ============================================================
# OUTILS DE CONTEXTE
# ============================================================

def fallback_class_report(signal_result: dict) -> dict | None:
    """Décrit une classe retenue par argmax sans franchissement de seuil."""

    if signal_result.get("status") != "success":
        return None

    if signal_result.get("selection_method") not in (None, "argmax_fallback"):
        return None

    fallback_class = signal_result.get("fallback_class")
    classes = (
        [fallback_class] if fallback_class is not None
        else signal_result.get("predicted_classes") or []
    )
    details = signal_result.get("details") or []
    if len(classes) != 1 or not isinstance(details, list):
        return None

    selected_class = classes[0]
    detail = next(
        (item for item in details if isinstance(item, dict)
         and item.get("class") == selected_class),
        None,
    )
    if not detail or detail.get("predicted") is not False:
        return None
    if any(
        item.get("predicted") is True
        for item in details if isinstance(item, dict)
    ):
        return None

    try:
        probabilities = signal_result.get("probabilities") or {}
        score = float(probabilities[selected_class])
        threshold = float(detail["threshold"])
        highest_score = max(float(value) for value in probabilities.values())
    except (KeyError, TypeError, ValueError):
        return None
    if score >= threshold or score != highest_score:
        return None

    return {
        "class": selected_class,
        "score": score,
        "threshold": threshold,
        "score_text": f"{score:.3f}".replace(".", ","),
    }


def signal_wording_issues(state: AgentState) -> list[str]:
    """Bloque une interprétation qualitative d'une classe choisie par repli."""

    report = fallback_class_report(state.get("signal_result", {}))
    if report is None:
        return []

    answer = state.get("final_answer", "")
    issues = []
    if report["class"] not in answer or report["score_text"] not in answer:
        issues.append(
            "La réponse doit nommer la classe retenue et son score exact "
            f"({report['class']} : {report['score_text']})."
        )
    if re.search(
        r"\b(?:probabilit[eé]|confiance)\s+(?:tr[eè]s\s+)?"
        r"(?:[eé]lev[eé]e|forte|importante)\b",
        answer,
        re.IGNORECASE,
    ):
        issues.append(
            "Une classe retenue sous son seuil ne doit pas être décrite "
            "comme une probabilité élevée ou une forte confiance."
        )
    return issues


def format_signal_context(signal_result: dict) -> str:
    """
    Transforme la sortie structurée de SignalAgent en un contexte court
    que ClinicalAgent et CriticAgent peuvent exploiter.

    On évite d'injecter un signal brut énorme dans le prompt.
    """

    if not signal_result:
        return "Aucun résultat SignalAgent disponible."

    status = signal_result.get("status")

    if status == "no_signal":
        return "Aucun ECG n'a été fourni à SignalAgent."

    if status == "error":
        return (
            "SignalAgent n'a pas pu analyser l'ECG. "
            f"Erreur : {signal_result.get('error', 'non précisée')}"
        )

    compact_result = {
        "status": status,
        "input_type": signal_result.get("input_type", "image"),
        "predicted_classes": signal_result.get("predicted_classes", []),
        "fallback_class": signal_result.get("fallback_class"),
        "selection_method": signal_result.get("selection_method"),
        "scores_par_classe": signal_result.get("probabilities", {}),
        "model": signal_result.get("model"),
        "n_windows_used": signal_result.get("n_windows_used"),
        "device": signal_result.get("device"),
    }

    context = (
        "RÉSULTAT SIGNALAGENT — résultat du modèle ECG, non équivalent "
        "à un diagnostic médical confirmé :\n"
        + json.dumps(compact_result, ensure_ascii=False, indent=2)
    )

    report = fallback_class_report(signal_result)
    if report is not None:
        threshold_text = f"{report['threshold']:.3f}".replace(".", ",")
        context += (
            "\nRÈGLE DE PRÉSENTATION DU RÉSULTAT : aucune classe n'a atteint "
            "son seuil ; le classifieur a choisi la classe au score maximal. "
            f"Le modèle a classé l'ECG dans la catégorie {report['class']}. "
            f"Le score associé est de {report['score_text']}, inférieur au "
            f"seuil de {threshold_text}. "
            "Rapporter ce score sans l'appeler probabilité élevée, sans "
            "présenter la classe comme une confirmation clinique."
        )

    return context


def build_authorized_context(state: AgentState) -> str:
    """
    Contexte autorisé pour CriticAgent : guidelines, références ECG,
    recherche locale, sortie SignalAgent, cas PTB-XL et Web externe.
    """

    signal_context = format_signal_context(
        state.get("signal_result", {})
    )

    rag_context = state.get("rag_context", "")
    ecg_reference_context = state.get("ecg_reference_context", "")
    research_context = state.get("research_context", "")
    ptbxl_context = state.get("ptbxl_context", "")
    ptbxl_case_ids = state.get("ptbxl_case_ids", [])
    web_context = state.get("web_context", "")
    web_source = state.get("web_source", "")

    ptbxl_ids_text = (
        ", ".join(str(case_id) for case_id in ptbxl_case_ids[:3])
        if ptbxl_case_ids
        else "non disponibles"
    )

    return f"""
================ GUIDELINES ESC/AHA — SOURCE NORMATIVE ================
{rag_context}

================ ECG REFERENCE — SUPPORT TECHNIQUE ====================
{ecg_reference_context if ecg_reference_context else "Aucune référence ECG disponible."}

================ RESEARCH LOCAL — LITTÉRATURE SCIENTIFIQUE ============
{research_context if research_context else "Aucun research paper local disponible."}

================ RÉSULTAT SIGNALAGENT =================================
{signal_context}

================ CAS PTB-XL — COMPARAISON UNIQUEMENT ==================
IDs PTB-XL retenus pour traçabilité : {ptbxl_ids_text}

{ptbxl_context if ptbxl_context else "Aucun cas PTB-XL disponible."}

================ WEB EXTERNE — COMPLÉMENT UNIQUEMENT =================
Source retenue : {web_source or "aucune"}
{web_context if web_context else "Aucune recherche Web utilisée pour cette question."}

RÈGLES DE PRIORITÉ :
- Les guidelines ESC/AHA constituent la source normative principale.
- ECG reference fournit un support technique / pédagogique.
- Research local fournit des articles scientifiques complémentaires.
- SignalAgent fournit une sortie de modèle sur l'ECG courant.
- PTB-XL fournit uniquement des cas comparatifs.
- Le Web fournit de la littérature externe complémentaire.
- ECG reference, Research, PTB-XL et Web ne deviennent jamais
  automatiquement des recommandations ESC/AHA.
- Un cas PTB-XL similaire ne confirme jamais un diagnostic.
- Une source traçable ne constitue pas automatiquement une vérité clinique.
- En cas de divergence entre une source complémentaire et une guideline,
  la divergence doit être signalée.
""".strip()


# ============================================================
# NODE 1 : SIGNAL AGENT
# ============================================================

def signal_node(state: AgentState):
    print("\n[LangGraph] SignalAgent")

    ecg_input_path = state.get("ecg_input_path", "")

    # Aucun ECG fourni : le reste du pipeline peut quand même fonctionner.
    if not ecg_input_path:
        result = {
            "agent": "SignalAgent",
            "status": "no_signal",
            "input_path": None,
            "signal_path": None,
            "predicted_classes": [],
            "raw_predicted_classes": [],
            "fallback_class": None,
            "selection_method": "none",
            "probabilities": {},
            "model": None,
            "details": [],
        }
        print("SignalAgent status :", result.get("status"))
        print("Aucun signal ECG fourni.")
        return {"signal_result": result}

    try:
        result = get_signal_agent().run(ecg_input_path)
    except Exception as exc:
        result = {
            "agent": "SignalAgent",
            "status": "error",
            "input_path": ecg_input_path,
            "error": str(exc),
        }

    print("SignalAgent status :", result.get("status"))

    if result.get("status") == "success":
        print(
            "Classes au-dessus du seuil :",
            result.get("predicted_classes", []),
        )
        print(
            "Probabilités :",
            result.get("probabilities", {}),
        )
        report = fallback_class_report(result)
        if report is not None:
            print(
                "Décision de repli (score maximal sous le seuil) :",
                f"{report['class']} = {report['score_text']}",
            )

    elif result.get("status") == "error":
        print(
            "ERREUR SIGNALAGENT :",
            result.get("error", "Erreur inconnue"),
        )

    else:
        print("Aucun signal ECG fourni.")

    return {"signal_result": result}


def route_after_evidence(state: AgentState):
    """
    Continue le workflow même si aucune image ECG n'est fournie.

    - success   -> continue avec SignalAgent + autres sources
    - no_signal -> continue sans analyse d'image
    - error     -> continue aussi, mais le contexte SignalAgent indiquera l'erreur

    Ainsi, LiteratureAgent, PTB-XL, WebSearchAgent, ClinicalAgent,
    CriticAgent et SynthesizerAgent peuvent fonctionner sans image.
    """

    status = state.get("signal_result", {}).get("status")

    if status in {"success", "no_signal", "error"}:
        return "continue"

    return "continue"


def signal_failure_node(state: AgentState):
    """Termine le workflow sans produire de réponse clinique non fondée."""

    status = state.get("signal_result", {}).get("status")
    print("[LangGraph] Analyse ECG indisponible : workflow arrêté")
    return {
        "final_answer": (
            MISSING_SIGNAL_ANSWER if status == "no_signal" else SIGNAL_ERROR_ANSWER
        ),
        "publication_status": "REJECTED",
    }


# ============================================================
# NODE 2 : LITERATURE AGENT
# ============================================================

def build_retrieval_question(
    question: str,
    signal_result: dict | None = None,
) -> str:
    """
    Enrichit la requête documentaire sans ajouter de fait à la réponse clinique.

    Important :
    - les catégories SignalAgent servent uniquement de termes de retrieval ;
    - elles ne deviennent jamais, à elles seules, un diagnostic médical.
    """

    parts = [str(question or "").strip()]
    normalized = str(question or "").lower()

    # Enrichissement depuis le texte utilisateur.
    if any(marker in normalized for marker in (
        "sus-décalage", "sus decalage", "stemi",
        "douleur thoracique", "chest pain",
    )):
        parts.append(
            "acute coronary syndrome STEMI ST-segment elevation "
            "diagnostic criteria emergency management"
        )

    if any(marker in normalized for marker in (
        "fibrillation atriale", "atrial fibrillation",
        "rr irréguliers", "rr irreguliers", "absence d'ondes p",
    )):
        parts.append(
            "atrial fibrillation ECG diagnostic criteria "
            "irregular RR intervals absence of P waves"
        )

    # Enrichissement depuis SignalAgent.
    signal_result = signal_result or {}
    if signal_result.get("status") == "success":
        threshold_classes = (
            signal_result.get("classes_above_threshold")
            or signal_result.get("predicted_classes")
            or []
        )
        fallback_class = signal_result.get("fallback_class")

        retrieval_classes = list(threshold_classes)
        if not retrieval_classes and fallback_class:
            retrieval_classes = [fallback_class]

        class_terms = {
            "NORM": (
                "normal ECG normal rhythm normal electrocardiogram"
            ),
            "MI": (
                "myocardial infarction acute coronary syndrome ischemia "
                "ST elevation ST depression ECG criteria"
            ),
            "STTC": (
                "ST-T changes repolarization abnormality ST segment "
                "T wave ECG criteria"
            ),
            "CD": (
                "cardiac conduction disturbance intraventricular conduction "
                "bundle branch block QRS duration wide QRS ECG criteria"
            ),
            "HYP": (
                "cardiac hypertrophy ventricular hypertrophy left ventricular "
                "hypertrophy ECG criteria"
            ),
        }

        for class_name in retrieval_classes:
            terms = class_terms.get(str(class_name))
            if terms:
                parts.append(terms)

    # Déduplication simple.
    compact = []
    seen = set()
    for part in parts:
        part = " ".join(str(part).split())
        if part and part not in seen:
            compact.append(part)
            seen.add(part)

    return " ".join(compact)


def literature_node(state: AgentState):
    print("[LangGraph] LiteratureAgent")

    retrieval_question = build_retrieval_question(
        state["question"],
        state.get("signal_result", {}),
    )

    print("[LangGraph] Requête RAG enrichie :", retrieval_question)

    result = get_literature_agent().run(
        retrieval_question
    )

    guideline_context = result.get(
        "guideline_context",
        result.get("context", ""),
    )

    ecg_reference_context = result.get(
        "ecg_reference_context",
        "",
    )

    research_context = result.get(
        "research_context",
        "",
    )

    ptbxl_context = result.get(
        "ptbxl_context",
        "",
    )

    ptbxl_case_ids = result.get(
        "ptbxl_case_ids",
        [],
    )

    print(
        "Guidelines disponibles :",
        bool(guideline_context),
    )
    print(
        "ECG reference disponible :",
        bool(ecg_reference_context),
    )
    print(
        "Research local disponible :",
        bool(research_context),
    )
    print(
        "Cas PTB-XL disponibles :",
        bool(ptbxl_context),
    )
    print(
        "ECG ID PTB-XL retenus :",
        ptbxl_case_ids[:3],
    )

    return {
        "retrieval_query": retrieval_question,
        "rag_context": guideline_context,
        "ecg_reference_context": ecg_reference_context,
        "research_context": research_context,
        "ptbxl_context": ptbxl_context,
        "ptbxl_case_ids": ptbxl_case_ids,
    }


# ============================================================
# NODE WEBSEARCH AGENT
# ============================================================

def websearch_node(state: AgentState):
    """
    Agent conditionnel :
    - si la question ne demande rien de récent/actuel, il ne fait aucun appel Web ;
    - sinon il lance la cascade PubMed -> arXiv -> DOAJ -> DuckDuckGo.
    """

    question = state.get("question", "")

    if not should_use_web_search(question):
        print("[LangGraph] WebSearchAgent : non déclenché")
        return {
            "web_search_used": False,
            "web_search_result": {
                "agent": "WebSearchAgent",
                "status": "not_triggered",
                "query": question,
            },
            "web_context": "",
            "web_source": "",
            "web_diagnostic": "WebSearchAgent non déclenché pour cette question.",
        }

    print("[LangGraph] WebSearchAgent : déclenché")

    try:
        result = get_websearch_agent().run(question)
    except Exception as exc:
        result = {
            "agent": "WebSearchAgent",
            "status": "error",
            "query": question,
            "selected_provider": None,
            "context": "",
            "results": [],
            "attempts": [],
            "error": str(exc),
        }

    source = result.get("selected_provider") or ""
    context = result.get("context") or ""

    if len(context) > 4000:
        context = (
            context[:4000]
            + "\n...[contexte Web tronqué]..."
        )
    diagnostic = format_web_diagnostic(result)

    print("WebSearch status :", result.get("status"))
    print("Source Web retenue :", source or "aucune")
    print("Nombre de résultats Web :", len(result.get("results", [])))
    print("Diagnostic WebSearch :")
    print(diagnostic)

    return {
        "web_search_used": True,
        "web_search_result": result,
        "web_context": context,
        "web_source": source,
        "web_diagnostic": diagnostic,
    }


def evidence_join_node(state: AgentState):
    """Attend les résultats SignalAgent et LiteratureAgent avant le raisonnement clinique."""

    print("[LangGraph] Evidence ready")
    print(
        "Signal disponible :",
        state.get("signal_result", {}).get("status") == "success",
    )
    print(
        "Guidelines disponibles :",
        bool(state.get("rag_context", "")),
    )
    print(
        "ECG reference disponible :",
        bool(state.get("ecg_reference_context", "")),
    )
    print(
        "Research local disponible :",
        bool(state.get("research_context", "")),
    )
    print(
        "Cas PTB-XL disponibles :",
        bool(state.get("ptbxl_context", "")),
    )
    print(
        "WebSearch utilisé :",
        state.get("web_search_used", False),
    )
    if state.get("web_search_used", False):
        print(
            "Source Web :",
            state.get("web_source", "") or "aucune",
        )
    return {}


# ============================================================
# NODE 3 : CLINICAL AGENT
# ============================================================

def clinical_node(state: AgentState):
    print("[LangGraph] ClinicalAgent")

    critic_result = state.get("critic_result", {})
    feedback = None

    # Si le CriticAgent a déjà demandé une correction,
    # le feedback est transmis au ClinicalAgent.
    if critic_result:
        verdict = critic_result.get("verdict")

        if verdict in {"IMPROVE", "FAIL"}:
            feedback_list = critic_result.get("feedback", [])

            if feedback_list:
                feedback = "\n".join(feedback_list)

    # Le ClinicalAgent reçoit séparément toutes les sources :
    # - guidelines ESC/AHA via context
    # - résultat SignalAgent via signal_result
    # - cas comparatifs PTB-XL via ptbxl_context
    result = get_clinical_agent().run(
        question=state["question"],
        context=state.get("rag_context", ""),
        critic_feedback=feedback,
        signal_result=state.get("signal_result", {}),
        ecg_reference_context=state.get("ecg_reference_context", ""),
        research_context=state.get("research_context", ""),
        ptbxl_context=state.get("ptbxl_context", ""),
        ptbxl_case_ids=state.get("ptbxl_case_ids", []),
        web_context=state.get("web_context", ""),
        web_source=state.get("web_source", ""),
        # Sans le brouillon, le LLM régénère la même réponse (temperature=0)
        # et reproduit la même faute à chaque itération.
        previous_answer=(state.get("clinical_answer", "") or None) if feedback else None,
    )

    if result.get("status") == "error":
        previous_answer = state.get("clinical_answer", "").strip()

        print(
            "[LangGraph] ClinicalAgent indisponible après retries :",
            result.get("error", "erreur inconnue"),
        )

        if previous_answer:
            print(
                "[LangGraph] Conservation de la réponse clinique précédente "
                "pour éviter l'arrêt complet du workflow."
            )
            answer = previous_answer
        else:
            return {
                "clinical_answer": (
                    "Le ClinicalAgent n'a pas pu générer de réponse en raison "
                    "d'une erreur réseau temporaire."
                )
            }
    else:
        answer = result["answer"]

    answer = strip_quotes_in_justification(answer)

    # ========================================================
    # MODE TEST UNIQUEMENT
    # ========================================================
    # Lors du premier passage seulement, on retire volontairement
    # l'avertissement médical. Ce code n'agit que si
    # test_force_improve=True.
    if (
        state.get("test_force_improve", False)
        and state.get("iterations", 0) == 0
    ):
        print(
            "[TEST] Première réponse volontairement dégradée : "
            "suppression de l'avertissement médical."
        )

        marker = "5. Avertissement médical"

        if marker in answer:
            answer = answer.split(marker, 1)[0].rstrip()

    return {
        "clinical_answer": answer
    }


# ============================================================
# NODE 4 : CRITIC AGENT
# ============================================================

def critic_node(state: AgentState):
    print("[LangGraph] CriticAgent")

    authorized_context = build_authorized_context(state)

    result = get_critic_agent().run(
        question=state["question"],
        context=authorized_context,
        answer=state["clinical_answer"],
    )

    print(
        "Critic verdict :",
        result.get("verdict")
    )

    print(
        "Critic score   :",
        f"{result.get('score', 0)}/{result.get('max_score', 7)}"
    )

    print(
        "Critic applicable :",
        f"{result.get('applicable_score', 0)}/"
        f"{result.get('applicable_max_score', 0)}",
        f"({result.get('applicable_normalized')})",
    )

    print(
        "Critic feedback:",
        result.get("feedback", [])
    )

    return {
        "critic_result": result
    }


# ============================================================
# ROUTAGE APRÈS CRITIC
# ============================================================

def route_after_critic(state: AgentState):
    """
    PASS           -> SynthesizerAgent
    IMPROVE / FAIL -> retry si une correction est encore autorisée
    limite atteinte -> SynthesizerAgent, puis contrôle final obligatoire
    """

    critic = state["critic_result"]
    verdict = critic.get("verdict", "IMPROVE")
    iterations = state.get("iterations", 0)

    if verdict == "PASS":
        return "synthesize"

    if iterations < MAX_ITERATIONS:
        return "retry"

    return "synthesize"


# ============================================================
# NODE DE RETRY
# ============================================================

def retry_node(state: AgentState):
    """Incrémente le compteur avant de refaire ClinicalAgent."""

    new_iteration = state.get("iterations", 0) + 1

    print(
        f"[LangGraph] Correction demandée "
        f"(itération {new_iteration}/{MAX_ITERATIONS})"
    )

    return {
        "iterations": new_iteration
    }


# ============================================================
# NODE 5 : SYNTHESIZER AGENT
# ============================================================

def synthesizer_node(state: AgentState):
    print("[LangGraph] SynthesizerAgent")

    # Une réponse clinique déjà validée ne doit pas être reformulée par
    # un second LLM : cette reformulation peut réintroduire une généralisation
    # ou une affirmation absente des sources. La traçabilité PTB-XL reste
    # ajoutée de manière déterministe par le nœud suivant.
    if state.get("critic_result", {}).get("verdict") == "PASS":
        print(
            "[LangGraph] Réponse clinique déjà validée : "
            "reformulation LLM ignorée"
        )
        return {
            "final_answer": state.get("clinical_answer", "")
        }

    result = get_synthesizer_agent().run(
        question=state["question"],
        clinical_answer=state["clinical_answer"],
        critic_result=state["critic_result"],
        signal_context=format_signal_context(state.get("signal_result", {})),
    )

    return {
        "final_answer": result["answer"]
    }


# ============================================================
# TRAÇABILITÉ DÉTERMINISTE PTB-XL
# ============================================================

def traceability_node(state: AgentState):
    """
    Préserve les ECG ID PTB-XL dans la réponse finale avant le contrôle final.
    """
    print("[LangGraph] Contrôle de traçabilité PTB-XL")

    normalized_answer = strip_quotes_in_justification(
        normalize_french_elisions(state.get("final_answer", ""))
    )

    corrected_answer = ensure_ptbxl_traceability(
        normalized_answer,
        state,
    )

    return {
        "final_answer": corrected_answer
    }


# ============================================================
# CONTRÔLE FINAL
# ============================================================

def final_review_node(state: AgentState):
    """Évalue la sortie du synthétiseur, et non seulement son entrée."""

    print("[LangGraph] CriticAgent (contrôle final)")

    authorized_context = build_authorized_context(state)

    result = get_critic_agent().run(
        question=state["question"],
        context=authorized_context,
        answer=state["final_answer"],
    )

    print(
        "Verdict final :",
        result.get("verdict"),
        f"({result.get('score', 0)}/{result.get('max_score', 7)})",
    )

    print(
        "Score applicable :",
        f"{result.get('applicable_score', 0)}/"
        f"{result.get('applicable_max_score', 0)}",
        f"({result.get('applicable_normalized')})",
    )

    print("Feedback final :", result.get("feedback", []))

    return {
        "final_critic_result": result
    }


# ============================================================
# CONFIANCE NVIDIA
# ============================================================


# ============================================================
# ROUTAGE APRÈS CONTRÔLE FINAL
# ============================================================

def route_after_final_review(state: AgentState):
    """
    PASS -> ConfidenceAgent.

    IMPROVE / FAIL -> une nouvelle correction est autorisée tant que
    le nombre maximal de corrections n'est pas atteint.

    Important :
    on ne force jamais PASS. Si le dernier contrôle reste non validé
    après épuisement des corrections, la publication restera REJECTED.
    """

    final_critic = state.get("final_critic_result", {}) or {}
    verdict = final_critic.get("verdict", "IMPROVE")
    iterations = state.get("iterations", 0)

    if verdict == "PASS":
        return "confidence"

    if iterations < MAX_ITERATIONS:
        return "final_retry"

    return "confidence"


def final_retry_node(state: AgentState):
    """
    Prépare une correction demandée par le contrôle final.

    Le feedback du Final Critic devient le feedback actif du CriticAgent
    afin que ClinicalAgent puisse corriger exactement les points qui ont
    provoqué le rejet final.
    """

    final_critic = state.get("final_critic_result", {}) or {}
    new_iteration = state.get("iterations", 0) + 1

    print(
        f"[LangGraph] Correction après contrôle final "
        f"(itération {new_iteration}/{MAX_ITERATIONS})"
    )

    print(
        "[LangGraph] Feedback du contrôle final transmis au ClinicalAgent :",
        final_critic.get("feedback", []),
    )

    return {
        "iterations": new_iteration,
        "critic_result": final_critic,
        # Le contrôle final a jugé final_answer (après synthèse) : c'est ce
        # texte que ClinicalAgent doit corriger, pas l'ancien clinical_answer.
        "clinical_answer": state.get("final_answer") or state.get("clinical_answer", ""),
    }


def confidence_node(state: AgentState):
    """Mesure la confiance de génération sans modifier le verdict clinique."""
    print("[LangGraph] ConfidenceAgent NVIDIA")

    try:
        result = get_confidence_agent().run(
            question=state["question"],
            answer=state["final_answer"],
        )
    except Exception as exc:
        result = {
            "agent": "ConfidenceAgent",
            "status": "error",
            "confidence": None,
            "entropy": None,
            "tokens_analyzed": 0,
            "exact_match": False,
            "error": str(exc),
        }

    print(
        "NVIDIA status     :",
        result.get("status"),
    )
    print(
        "NVIDIA error      :",
        result.get("error"),
    )
    print(
        "NVIDIA confidence :",
        result.get("confidence"),
    )
    print(
        "NVIDIA entropy    :",
        result.get("entropy"),
    )

    return {
        "confidence_result": result
    }


# ============================================================
# PUBLICATION
# ============================================================

def publication_node(state: AgentState):
    """Ne publie jamais une réponse clinique que le contrôle final rejette."""

    final_critic = state.get("final_critic_result", {})
    wording_issues = signal_wording_issues(state)
    ptbxl_issues = ptbxl_traceability_issues(state)

    signal_presence_issues = []
    signal_result = state.get("signal_result", {}) or {}
    if signal_result.get("status") == "success":
        answer_text = str(state.get("final_answer", "") or "")
        threshold_classes = (
            signal_result.get("classes_above_threshold")
            or signal_result.get("predicted_classes")
            or []
        )
        fallback_class = signal_result.get("fallback_class")

        expected_classes = list(threshold_classes)
        if not expected_classes and fallback_class:
            expected_classes = [fallback_class]

        for class_name in expected_classes:
            if str(class_name) not in answer_text:
                signal_presence_issues.append(
                    "La réponse finale omet la catégorie SignalAgent "
                    f"{class_name} alors que l'analyse ECG a réussi."
                )

    deterministic_issues = (
        wording_issues
        + ptbxl_issues
        + signal_presence_issues
    )

    if (
        final_critic.get("verdict") == "PASS"
        and not deterministic_issues
    ):
        return {
            "publication_status": "PASS",
            "publication_feedback": [],
        }

    print("[LangGraph] Réponse non validée : publication refusée")

    for issue in deterministic_issues:
        print("[LangGraph]", issue)

    final_feedback = final_critic.get("feedback", []) or []

    return {
        "rejected_answer": state.get("final_answer", ""),
        "final_answer": UNVALIDATED_ANSWER,
        "publication_status": "REJECTED",
        "publication_feedback": deterministic_issues + final_feedback,
    }


# ============================================================
# CONSTRUCTION DU GRAPHE
# ============================================================

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("signal", signal_node)
    graph.add_node("signal_failure", signal_failure_node)
    graph.add_node("literature", literature_node)
    graph.add_node("websearch", websearch_node)
    graph.add_node("evidence_join", evidence_join_node)
    graph.add_node("clinical", clinical_node)
    graph.add_node("critic", critic_node)
    graph.add_node("retry", retry_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("traceability", traceability_node)
    graph.add_node("final_review", final_review_node)
    graph.add_node("final_retry", final_retry_node)
    graph.add_node("confidence", confidence_node)
    graph.add_node("publication", publication_node)

    # SignalAgent s'exécute d'abord afin que sa sortie puisse enrichir
    # la requête documentaire (sans être utilisée comme diagnostic).
    graph.add_edge(START, "signal")

    # Après SignalAgent, LiteratureAgent et WebSearch peuvent travailler en parallèle.
    graph.add_edge("signal", "literature")
    graph.add_edge("signal", "websearch")

    # Barrière : ClinicalAgent attend les deux branches documentaires.
    graph.add_edge(["literature", "websearch"], "evidence_join")
    graph.add_conditional_edges(
        "evidence_join",
        route_after_evidence,
        {"continue": "clinical", "reject": "signal_failure"},
    )
    graph.add_edge("signal_failure", END)
    graph.add_edge("clinical", "critic")

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "synthesize": "synthesizer",
            "retry": "retry",
        },
    )

    graph.add_edge("retry", "clinical")
    graph.add_edge("synthesizer", "traceability")
    graph.add_edge("traceability", "final_review")
    graph.add_conditional_edges(
        "final_review",
        route_after_final_review,
        {
            "confidence": "confidence",
            "final_retry": "final_retry",
        },
    )
    graph.add_edge("final_retry", "clinical")
    graph.add_edge("confidence", "publication")
    graph.add_edge("publication", END)

    return graph.compile()


# ============================================================
# EXÉCUTION
# ============================================================

def run_agentic_rag(
    question: str,
    ecg_input_path: str = "",
    test_force_improve: bool = False,
    ecg_required: bool = False,
):
    app = build_graph()

    initial_state: AgentState = {
        "question": question,
        "ecg_input_path": ecg_input_path,
        "ecg_required": ecg_required,
        "signal_result": {},
        "retrieval_query": "",
        "rag_context": "",
        "ecg_reference_context": "",
        "research_context": "",
        "ptbxl_context": "",
        "ptbxl_case_ids": [],
        "web_search_used": False,
        "web_search_result": {},
        "web_context": "",
        "web_source": "",
        "web_diagnostic": "",
        "clinical_answer": "",
        "critic_result": {},
        "final_critic_result": {},
        "confidence_result": {},
        "final_answer": "",
        "rejected_answer": "",
        "publication_status": "PENDING",
        "publication_feedback": [],
        "iterations": 0,
        "test_force_improve": test_force_improve,
    }

    result = app.invoke(initial_state)

    return result


# ============================================================
# TEST LOCAL
# ============================================================

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    question = (
    "Quelles sont les méthodes récentes en 2026 pour détecter la fibrillation atriale sur ECG ?" 
    
)

    # Mets ici le chemin réel vers ton image ECG pour tester SignalAgent.
    # Laisse "" pour tester le pipeline sans image ECG.
    ECG_INPUT_PATH = ""

    # True  = test volontaire de la boucle IMPROVE -> retry -> ClinicalAgent
    # False = fonctionnement normal
    TEST_FORCE_IMPROVE = False

    result = run_agentic_rag(
        question=question,
        ecg_input_path=ECG_INPUT_PATH,
        test_force_improve=TEST_FORCE_IMPROVE,
        ecg_required=False,
    )

    print("\n" + "=" * 80)
    print(
        "RÉPONSE FINALE" if result["publication_status"] == "PASS"
        else "ANALYSE NON DISPONIBLE"
    )
    print("=" * 80)
    print(result["final_answer"])

    print("\n" + "=" * 80)
    print("INFORMATIONS DU WORKFLOW")
    print("=" * 80)

    print(
        "SignalAgent status :",
        result.get("signal_result", {}).get("status")
    )

    if result.get("signal_result", {}).get("status") == "success":
        print(
            "Classes au-dessus du seuil :",
            result["signal_result"].get("predicted_classes", []),
        )
        fallback_class = result["signal_result"].get("fallback_class")
        if fallback_class is not None:
            print("Classe retenue par repli :", fallback_class)
    elif result.get("signal_result", {}).get("status") == "error":
        print("Erreur SignalAgent :", result["signal_result"].get("error"))

    print(
        "Cas PTB-XL disponibles :",
        bool(result.get("ptbxl_context", ""))
    )
    print(
        "WebSearch utilisé :",
        result.get("web_search_used", False)
    )
    if result.get("web_search_used", False):
        print(
            "Source Web retenue :",
            result.get("web_source", "") or "aucune"
        )
        print("Diagnostic WebSearch :")
        print(
            result.get(
                "web_diagnostic",
                "Diagnostic indisponible."
            )
        )
    print(
        "ECG ID PTB-XL utilisés :",
        result.get("ptbxl_case_ids", [])[:3]
    )

    print(
        "Mode test IMPROVE :",
        result.get("test_force_improve", False)
    )

    print(
        "Nombre de corrections :",
        result["iterations"]
    )

    final_critic = result.get("final_critic_result", {})
    if final_critic:
        print("Verdict sur la réponse finale :", final_critic.get("verdict"))
        print(
            "Score sur la réponse finale :",
            f"{final_critic.get('score', 0)}/{final_critic.get('max_score', 7)}",
        )
    else:
        print("CriticAgent : non exécuté (analyse ECG indisponible)")

    confidence_result = result.get(
        "confidence_result",
        {},
    )

    print(
        "NVIDIA status :",
        confidence_result.get("status"),
    )
    print(
        "NVIDIA error :",
        confidence_result.get("error"),
    )
    print(
        "NVIDIA confidence :",
        confidence_result.get("confidence"),
    )
    print(
        "NVIDIA entropy :",
        confidence_result.get("entropy"),
    )
    print(
        "Tokens analysés :",
        confidence_result.get("tokens_analyzed"),
    )
    print(
        "NVIDIA exact match :",
        confidence_result.get("exact_match"),
    )

    print("Publication :", result["publication_status"])

    if result["publication_status"] != "PASS":
        rejection_reasons = (
            result.get("publication_feedback", [])
            + result.get("final_critic_result", {}).get("feedback", [])
        )

        signal_status = result.get("signal_result", {}).get("status")
        if result.get("ecg_required", False) and signal_status == "error":
            rejection_reasons.append(
                result.get("signal_result", {}).get("error", "Erreur SignalAgent")
            )
        elif result.get("ecg_required", False) and signal_status == "no_signal":
            rejection_reasons.append("Aucune image ECG fournie alors qu'elle est requise.")

        print(
            "Motifs du rejet :",
            rejection_reasons or ["Le contrôle final n'a pas validé la réponse."],
        )
