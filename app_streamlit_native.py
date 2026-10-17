# ============================================================
# app.py
# CardioVision IA - Streamlit Interface
# ============================================================

import io
import os
import re
import time
import tempfile
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import streamlit as st
from PIL import Image
import subprocess
import json
import sys




# ============================================================
# CONFIGURATION STREAMLIT
# ============================================================

st.set_page_config(
    page_title="CardioVision IA",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# INTERFACE STREAMLIT NATIVE
# ============================================================

# ============================================================
# SESSION STATE
# ============================================================

if "result" not in st.session_state:
    st.session_state.result = None

if "elapsed" not in st.session_state:
    st.session_state.elapsed = None

if "logs" not in st.session_state:
    st.session_state.logs = ""

if "example_text" not in st.session_state:
    st.session_state.example_text = ""


# ============================================================
# EXEMPLE DE CAS
# ============================================================

EXAMPLE_CASE = """
Femme de 68 ans, ayant des antécédents d’hypertension artérielle et de diabète de type 2, consultant aux urgences
pour une douleur thoracique constrictive irradiant vers le bras gauche depuis 45 minutes, associée à des sueurs et
des nausées. Signes vitaux : FC 98 bpm, TA 145/92 mmHg, SpO₂ 96 %. ECG : rythme sinusal, QRS 110 ms, sus-décalage 
du segment ST de 3 mm en V2-V3-V4, ondes T négatives en III et aVF, QTc 440 ms. Générer un rapport clinique complet 
comprenant le diagnostic ECG probable, le niveau d’urgence, les critères ESC applicables, la justification documentaire,
les recommandations de prise en charge uniquement si elles sont soutenues par le RAG, ainsi que les limites du contexte
et l’avertissement médical.
""".strip()


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, digits=3):
    """
    Affichage sûr d'un float.
    """
    if value is None:
        return "—"

    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def render_metric(
    label: str,
    value,
    hint: str = "",
):
    """Affiche une métrique avec les composants Streamlit natifs."""
    with st.container(border=True):
        st.metric(
            label=label,
            value="—" if value is None else str(value),
        )
        if hint:
            st.caption(str(hint))


def save_uploaded_ecg(uploaded_file):
    """
    Sauvegarde temporairement l'image ECG
    pour la transmettre à SignalAgent.
    """
    if uploaded_file is None:
        return ""

    suffix = (
        Path(uploaded_file.name)
        .suffix
        .lower()
    )

    if suffix not in {
        ".png",
        ".jpg",
        ".jpeg",
    }:
        suffix = ".png"

    temp_dir = (
        Path(tempfile.gettempdir())
        / "cardiovision_streamlit"
    )

    temp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        temp_dir
        / f"ecg_{int(time.time() * 1000)}{suffix}"
    )

    path.write_bytes(
        uploaded_file.getbuffer()
    )

    return str(path)


def run_pipeline(
    question: str,
    ecg_path: str = "",
):
    start = time.perf_counter()

    temp_dir = (
        Path(tempfile.gettempdir())
        / "cardiovision_streamlit"
    )

    temp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    request_path = (
        temp_dir
        / f"request_{int(time.time() * 1000)}.json"
    )

    payload = {
        "question": question,
        "ecg_input_path": ecg_path or "",
    }

    with open(
        request_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2,
        )

    project_root = Path(__file__).resolve().parent

    worker_path = (
        project_root
        / "pipeline_worker.py"
    )

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    process = subprocess.run(
        [
            sys.executable,
            str(worker_path),
            str(request_path),
        ],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    stdout = process.stdout or ""
    stderr = process.stderr or ""

    marker = "__CARDIOVISION_RESULT__"

    if marker in stdout:
        json_text = stdout.split(
            marker,
            1,
        )[1].strip()

        # Le résultat JSON est placé à la fin.
        result = json.loads(
            json_text
        )

        logs = stdout.split(
            marker,
            1,
        )[0]

        if stderr:
            logs += (
                "\n\nSTDERR:\n"
                + stderr
            )

        return (
            result,
            elapsed,
            logs,
        )

    error_marker = (
        "__CARDIOVISION_ERROR__"
    )

    if error_marker in stdout:
        error_text = stdout.split(
            error_marker,
            1,
        )[1]

        raise RuntimeError(
            error_text
        )

    raise RuntimeError(
        "Le processus CardioVision s'est arrêté "
        "sans retourner de résultat.\n\n"
        f"Exit code : {process.returncode}\n\n"
        f"STDOUT:\n{stdout}\n\n"
        f"STDERR:\n{stderr}"
    )

def split_context_chunks(context_text):
    """
    Découpe grossièrement les contextes RAG
    en passages pour améliorer l'affichage.
    """

    if not context_text:
        return []

    text = str(context_text)

    pattern = (
        r"(?="
        r"\[PASSAGE|"
        r"PASSAGE |"
        r"CAS PTB-XL|"
        r"CAS PTB"
        r")"
    )

    chunks = re.split(
        pattern,
        text,
    )

    return [
        chunk.strip()
        for chunk in chunks
        if chunk.strip()
    ]


def render_context(
    title: str,
    context_text: str,
):
    """
    Affiche les passages RAG.
    """

    if not context_text:

        st.info(
            f"Aucun contexte {title} disponible."
        )

        return

    chunks = split_context_chunks(
        context_text
    )

    if not chunks:

        st.code(
            str(context_text),
            language="text",
        )

        return

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):

        first_line = (
            chunk
            .splitlines()[0]
            [:100]
        )

        with st.expander(
            f"{index}. {first_line}"
        ):

            st.text(chunk)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🫀 CardioVision IA"
    )

    st.caption(
        "Système multi-agent d'aide à la décision "
        "clinique pour ECG."
    )

    st.divider()

    st.markdown(
        "### Architecture"
    )

    st.markdown(
        """
        **📈 SignalAgent**

        **📚 LiteratureAgent**

        **🌐 WebSearchAgent**

        **🩺 ClinicalAgent**

        **🛡️ CriticAgent**

        **🧠 SynthesizerAgent**

        **🎯 ConfidenceAgent**
        """
    )

    st.divider()

    st.markdown(
        "### Workflow"
    )

    st.caption(
        "Signal + Literature + WebSearch "
        "→ Clinical → Critic → Synthesizer "
        "→ Final Critic → Confidence → Publication"
    )

    st.divider()

    st.warning(
        "Prototype académique. "
        "Les résultats ne remplacent pas "
        "le jugement d'un médecin."
    )


# ============================================================
# HEADER
# ============================================================

st.title("🫀 CardioVision IA")
st.caption("Système multi-agent d'aide à la décision clinique pour l'analyse ECG.")
st.divider()


# ============================================================
# INPUT AREA
# ============================================================

left_column, right_column = st.columns(
    [1.65, 1]
)


# ------------------------------------------------------------
# QUESTION
# ------------------------------------------------------------

with left_column:

    st.subheader("🩺 Cas clinique / Question")
    st.caption("Décrivez un cas clinique ou posez directement une question médicale.")

    question = st.text_area(
        "Question",
        value=st.session_state.example_text,
        height=250,
        label_visibility="collapsed",
        placeholder=(
            "Exemple : patient présentant une douleur "
            "thoracique avec sus-décalage du segment ST..."
        ),
    )

    button_example, button_clear = st.columns(
        2
    )

    with button_example:

        if st.button(
            "📋 Charger un exemple",
            use_container_width=True,
        ):

            st.session_state.example_text = (
                EXAMPLE_CASE
            )

            st.rerun()

    with button_clear:

        if st.button(
            "🗑️ Réinitialiser",
            use_container_width=True,
        ):

            st.session_state.result = None
            st.session_state.elapsed = None
            st.session_state.logs = ""
            st.session_state.example_text = ""

            st.rerun()


# ------------------------------------------------------------
# ECG IMAGE
# ------------------------------------------------------------

with right_column:

    st.subheader("📈 Image ECG")
    st.caption("Optionnelle — SignalAgent sera activé si une image est fournie.")

    uploaded_ecg = st.file_uploader(
        "ECG",
        type=[
            "png",
            "jpg",
            "jpeg",
        ],
        label_visibility="collapsed",
    )

    if uploaded_ecg is not None:

        image = Image.open(
            uploaded_ecg
        )

        st.image(
            image,
            caption=uploaded_ecg.name,
            use_container_width=True,
        )

    else:

        st.info(
            "Aucune image ECG fournie. "
            "Le RAG médical peut quand même "
            "répondre aux questions."
        )


# ============================================================
# RUN BUTTON
# ============================================================

st.write("")

run_clicked = st.button(
    "▶️ Lancer l'analyse multi-agent",
    type="primary",
    use_container_width=True,
)


# ============================================================
# EXECUTION
# ============================================================

if run_clicked:

    if not question.strip():

        st.warning(
            "Veuillez saisir une question "
            "ou un cas clinique."
        )

        st.stop()

    # --------------------------------------------------------
    # ECG PATH
    # --------------------------------------------------------

    ecg_path = save_uploaded_ecg(
        uploaded_ecg
    )

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    progress = st.progress(0)

    status_message = st.empty()

    try:

        status_message.info(
            "Préparation du cas clinique..."
        )

        progress.progress(10)

        if ecg_path:

            status_message.info(
                "Image ECG chargée. "
                "Exécution du système multi-agent..."
            )

        else:

            status_message.info(
                "Exécution du système multi-agent "
                "sans image ECG..."
            )

        progress.progress(30)

        # ----------------------------------------------------
        # REAL PIPELINE
        # ----------------------------------------------------

        result, elapsed, logs = (
            run_pipeline(
                question=question,
                ecg_path=ecg_path,
            )
        )

        progress.progress(90)

        # ----------------------------------------------------
        # SAVE SESSION
        # ----------------------------------------------------

        st.session_state.result = (
            result
        )

        st.session_state.elapsed = (
            elapsed
        )

        st.session_state.logs = (
            logs
        )

        progress.progress(100)

        publication_status = (
            result.get(
                "publication_status",
                "PENDING",
            )
        )

        if publication_status == "PASS":

            status_message.success(
                f"✅ Analyse terminée en "
                f"{elapsed:.1f} secondes — "
                f"réponse validée."
            )

        else:

            status_message.warning(
                f"⚠️ Analyse terminée en "
                f"{elapsed:.1f} secondes — "
                f"publication : "
                f"{publication_status}."
            )

        time.sleep(0.2)

        progress.empty()

    except Exception as exc:

        progress.empty()

        status_message.error(
            f"Erreur pendant l'analyse : "
            f"{exc}"
        )

        st.exception(exc)


# ============================================================
# RESULTS
# ============================================================

result = st.session_state.result


if result:

    st.write("")

    publication_status = (
        result.get(
            "publication_status",
            "PENDING",
        )
    )

    elapsed = (
        st.session_state.elapsed
    )


    # ========================================================
    # STATUS BAR
    # ========================================================

    status_col, time_col = st.columns(
        [3, 1]
    )

    with status_col:

        if publication_status == "PASS":
            st.success("✓ Réponse validée")
        elif publication_status == "REJECTED":
            st.error("✕ Réponse rejetée")
        else:
            st.info(str(publication_status))

    with time_col:

        if elapsed is not None:

            st.caption(
                f"⏱ Temps total : "
                f"{elapsed:.1f} s"
            )


    # ========================================================
    # TABS
    # ========================================================

    (
        tab_final,
        tab_signal,
        tab_sources,
        tab_critic,
        tab_web,
        tab_performance,
        tab_logs,
    ) = st.tabs(
        [
            "📄 Réponse finale",
            "📈 SignalAgent",
            "📚 RAG & PTB-XL",
            "🛡️ Critic",
            "🌐 WebSearch",
            "📊 Performance",
            "🧾 Logs",
        ]
    )


    # ========================================================
    # TAB 1 — FINAL ANSWER
    # ========================================================

    with tab_final:

        st.markdown(
            "### Rapport clinique final"
        )

        final_answer = (
            result.get(
                "final_answer",
                "",
            )
            or
            "Aucune réponse disponible."
        )

        st.markdown(
            final_answer
        )


        # ----------------------------------------------------
        # REJECTED ANSWER
        # ----------------------------------------------------

        if publication_status != "PASS":

            publication_feedback = (
                result.get(
                    "publication_feedback",
                    [],
                )
                or []
            )

            if publication_feedback:

                st.error(
                    "Motifs de non-publication :"
                )

                for item in publication_feedback:

                    st.write(
                        f"- {item}"
                    )


            rejected_answer = (
                result.get(
                    "rejected_answer",
                    "",
                )
            )

            if rejected_answer:

                with st.expander(
                    "Voir la réponse rejetée"
                ):

                    st.markdown(
                        rejected_answer
                    )


        st.divider()


        # ----------------------------------------------------
        # MAIN METRICS
        # ----------------------------------------------------

        confidence_result = (
            result.get(
                "confidence_result",
                {},
            )
            or {}
        )

        final_critic = (
            result.get(
                "final_critic_result",
                {},
            )
            or {}
        )

        metric_1, \
        metric_2, \
        metric_3, \
        metric_4 = st.columns(4)


        with metric_1:

            render_metric(
                "Confidence",
                safe_float(
                    confidence_result.get(
                        "confidence"
                    )
                ),
                confidence_result.get(
                    "status",
                    "",
                ),
            )


        with metric_2:

            render_metric(
                "Entropy",
                safe_float(
                    confidence_result.get(
                        "entropy"
                    )
                ),
                (
                    "Plus faible = "
                    "génération plus stable"
                ),
            )


        with metric_3:

            render_metric(
                "Corrections",
                result.get(
                    "iterations",
                    0,
                ),
                (
                    "Boucles de correction "
                    "du workflow"
                ),
            )


        with metric_4:

            render_metric(
                "Final Critic",
                final_critic.get(
                    "verdict",
                    "—",
                ),
                (
                    f"{final_critic.get('score', '—')}/"
                    f"{final_critic.get('max_score', 7)}"
                ),
            )


    # ========================================================
    # TAB 2 — SIGNALAGENT
    # ========================================================

    with tab_signal:

        st.markdown(
            "### 📈 SignalAgent"
        )

        signal_result = (
            result.get(
                "signal_result",
                {},
            )
            or {}
        )

        signal_status = (
            signal_result.get(
                "status",
                "unknown",
            )
        )


        signal_col_1, \
        signal_col_2, \
        signal_col_3 = st.columns(3)


        with signal_col_1:

            render_metric(
                "Status",
                signal_status,
            )


        with signal_col_2:

            predicted_classes = (
                signal_result.get(
                    "predicted_classes",
                    [],
                )
                or []
            )

            fallback_class = (
                signal_result.get(
                    "fallback_class"
                )
            )

            if predicted_classes:

                class_display = ", ".join(
                    predicted_classes
                )

            elif fallback_class:

                class_display = (
                    fallback_class
                )

            else:

                class_display = "—"

            render_metric(
                "Classe(s)",
                class_display,
            )


        with signal_col_3:

            render_metric(
                "Méthode",
                signal_result.get(
                    "selection_method",
                    "—",
                ),
            )


        # ----------------------------------------------------
        # SIGNAL SUCCESS
        # ----------------------------------------------------

        if signal_status == "success":

            probabilities = (
                signal_result.get(
                    "probabilities",
                    {},
                )
                or {}
            )

            if probabilities:

                st.markdown(
                    "#### Scores par classe"
                )

                table_data = []

                for (
                    class_name,
                    score
                ) in probabilities.items():

                    table_data.append(
                        {
                            "Classe": class_name,
                            "Score": score,
                        }
                    )

                st.dataframe(
                    table_data,
                    use_container_width=True,
                    hide_index=True,
                )


            details = (
                signal_result.get(
                    "details",
                    [],
                )
                or []
            )

            if details:

                with st.expander(
                    "Détails des seuils"
                ):

                    st.json(
                        details
                    )


            st.caption(
                f"Modèle : "
                f"{signal_result.get('model', '—')} "
                f"· Fenêtres utilisées : "
                f"{signal_result.get('n_windows_used', '—')} "
                f"· Device : "
                f"{signal_result.get('device', '—')}"
            )


        # ----------------------------------------------------
        # NO SIGNAL
        # ----------------------------------------------------

        elif signal_status == "no_signal":

            st.info(
                "Aucune image ECG fournie. "
                "SignalAgent n'a pas effectué "
                "de classification."
            )


        # ----------------------------------------------------
        # ERROR
        # ----------------------------------------------------

        elif signal_status == "error":

            st.error(
                signal_result.get(
                    "error",
                    "Erreur SignalAgent inconnue.",
                )
            )


    # ========================================================
    # TAB 3 — RAG & PTB-XL
    # ========================================================

    with tab_sources:

        # ----------------------------------------------------
        # GUIDELINES
        # ----------------------------------------------------

        st.markdown(
            "### 📘 Guidelines ESC / AHA"
        )

        render_context(
            "guideline",
            result.get(
                "rag_context",
                "",
            ),
        )


        st.divider()


        # ----------------------------------------------------
        # ECG REFERENCES
        # ----------------------------------------------------

        st.markdown(
            "### 📗 Références ECG"
        )

        render_context(
            "ECG reference",
            result.get(
                "ecg_reference_context",
                "",
            ),
        )


        st.divider()


        # ----------------------------------------------------
        # RESEARCH
        # ----------------------------------------------------

        st.markdown(
            "### 📙 Research local"
        )

        render_context(
            "research",
            result.get(
                "research_context",
                "",
            ),
        )


        st.divider()


        # ----------------------------------------------------
        # PTB-XL
        # ----------------------------------------------------

        st.markdown(
            "### 🧬 PTB-XL"
        )

        ptbxl_ids = (
            result.get(
                "ptbxl_case_ids",
                [],
            )
            or []
        )


        if ptbxl_ids:

            ptb_columns = st.columns(
                min(
                    3,
                    len(ptbxl_ids),
                )
            )

            for index, ecg_id in enumerate(
                ptbxl_ids[:3]
            ):

                with ptb_columns[index]:

                    render_metric(
                        "ECG ID",
                        ecg_id,
                        "Cas comparatif",
                    )

        else:

            st.info(
                "Aucun ECG ID PTB-XL "
                "disponible."
            )


        ptbxl_context = (
            result.get(
                "ptbxl_context",
                "",
            )
        )

        if ptbxl_context:

            with st.expander(
                "Voir les cas PTB-XL récupérés"
            ):

                st.text(
                    ptbxl_context
                )


    # ========================================================
    # TAB 4 — CRITIC
    # ========================================================

    with tab_critic:

        critic_result = (
            result.get(
                "critic_result",
                {},
            )
            or {}
        )

        final_critic = (
            result.get(
                "final_critic_result",
                {},
            )
            or {}
        )


        # ----------------------------------------------------
        # INITIAL CRITIC
        # ----------------------------------------------------

        st.markdown(
            "### 🛡️ Critic initial"
        )

        critic_col_1, \
        critic_col_2, \
        critic_col_3 = st.columns(3)


        with critic_col_1:

            render_metric(
                "Verdict",
                critic_result.get(
                    "verdict",
                    "—",
                ),
            )


        with critic_col_2:

            render_metric(
                "Score",
                (
                    f"{critic_result.get('score', '—')}/"
                    f"{critic_result.get('max_score', 7)}"
                ),
            )


        with critic_col_3:

            applicable_score = (
                critic_result.get(
                    "applicable_score",
                    "—",
                )
            )

            applicable_max = (
                critic_result.get(
                    "applicable_max_score",
                    "—",
                )
            )

            render_metric(
                "Applicable",
                (
                    f"{applicable_score}/"
                    f"{applicable_max}"
                ),
                safe_float(
                    critic_result.get(
                        "applicable_normalized"
                    )
                ),
            )


        feedback = (
            critic_result.get(
                "feedback",
                [],
            )
            or []
        )

        if feedback:

            st.markdown(
                "#### Feedback"
            )

            for item in feedback:

                st.warning(
                    item
                )

        else:

            st.success(
                "Aucun feedback bloquant "
                "au contrôle initial."
            )


        st.divider()


        # ----------------------------------------------------
        # FINAL CRITIC
        # ----------------------------------------------------

        st.markdown(
            "### ✅ Contrôle final"
        )

        final_col_1, \
        final_col_2, \
        final_col_3 = st.columns(3)


        with final_col_1:

            render_metric(
                "Verdict",
                final_critic.get(
                    "verdict",
                    "—",
                ),
            )


        with final_col_2:

            render_metric(
                "Score",
                (
                    f"{final_critic.get('score', '—')}/"
                    f"{final_critic.get('max_score', 7)}"
                ),
            )


        with final_col_3:

            final_applicable_score = (
                final_critic.get(
                    "applicable_score",
                    "—",
                )
            )

            final_applicable_max = (
                final_critic.get(
                    "applicable_max_score",
                    "—",
                )
            )

            render_metric(
                "Applicable",
                (
                    f"{final_applicable_score}/"
                    f"{final_applicable_max}"
                ),
                safe_float(
                    final_critic.get(
                        "applicable_normalized"
                    )
                ),
            )


        final_feedback = (
            final_critic.get(
                "feedback",
                [],
            )
            or []
        )

        if final_feedback:

            st.markdown(
                "#### Feedback final"
            )

            for item in final_feedback:

                st.error(
                    item
                )

        else:

            st.success(
                "Aucune erreur bloquante "
                "détectée au contrôle final."
            )


    # ========================================================
    # TAB 5 — WEBSEARCH
    # ========================================================

    with tab_web:

        web_used = (
            result.get(
                "web_search_used",
                False,
            )
        )

        if not web_used:

            st.info(
                "WebSearchAgent n'a pas été "
                "déclenché pour cette question."
            )

        else:

            st.success(
                "🌐 WebSearchAgent a été déclenché."
            )

            st.write(
                "**Provider retenu :**",
                (
                    result.get(
                        "web_source",
                        "",
                    )
                    or
                    "aucun"
                ),
            )


            web_diagnostic = (
                result.get(
                    "web_diagnostic",
                    "",
                )
            )

            if web_diagnostic:

                st.markdown(
                    "#### Diagnostic WebSearch"
                )

                st.code(
                    web_diagnostic,
                    language="text",
                )


            web_context = (
                result.get(
                    "web_context",
                    "",
                )
            )

            if web_context:

                with st.expander(
                    "Contexte Web utilisé"
                ):

                    st.text(
                        web_context
                    )


            web_result = (
                result.get(
                    "web_search_result",
                    {},
                )
                or {}
            )

            if web_result:

                with st.expander(
                    "Résultat structuré WebSearch"
                ):

                    st.json(
                        web_result
                    )


    # ========================================================
    # TAB 6 — PERFORMANCE
    # ========================================================

    with tab_performance:

        confidence_result = (
            result.get(
                "confidence_result",
                {},
            )
            or {}
        )

        final_critic = (
            result.get(
                "final_critic_result",
                {},
            )
            or {}
        )


        perf_col_1, \
        perf_col_2, \
        perf_col_3, \
        perf_col_4 = st.columns(4)


        with perf_col_1:

            render_metric(
                "Latence",
                (
                    f"{elapsed:.1f} s"
                    if elapsed is not None
                    else "—"
                ),
                "Temps total du pipeline",
            )


        with perf_col_2:

            render_metric(
                "Confidence",
                safe_float(
                    confidence_result.get(
                        "confidence"
                    )
                ),
            )


        with perf_col_3:

            render_metric(
                "Entropy",
                safe_float(
                    confidence_result.get(
                        "entropy"
                    )
                ),
            )


        with perf_col_4:

            render_metric(
                "Corrections",
                result.get(
                    "iterations",
                    0,
                ),
            )


        st.write("")


        perf_col_5, \
        perf_col_6, \
        perf_col_7, \
        perf_col_8 = st.columns(4)


        with perf_col_5:

            render_metric(
                "Publication",
                publication_status,
            )


        with perf_col_6:

            render_metric(
                "Final Critic",
                (
                    f"{final_critic.get('score', '—')}/"
                    f"{final_critic.get('max_score', 7)}"
                ),
            )


        with perf_col_7:

            render_metric(
                "Confidence status",
                confidence_result.get(
                    "status",
                    "—",
                ),
            )


        with perf_col_8:

            render_metric(
                "Exact match",
                confidence_result.get(
                    "exact_match",
                    "—",
                ),
            )


        st.divider()

        st.caption(
            "La confiance NVIDIA mesure la stabilité "
            "de génération et ne constitue pas une "
            "probabilité de validité clinique."
        )


    # ========================================================
    # TAB 7 — LOGS
    # ========================================================

    with tab_logs:

        st.markdown(
            "### 🧾 Logs du workflow"
        )

        raw_logs = (
            st.session_state.logs
        )

        if raw_logs:

            st.code(
                raw_logs,
                language="text",
            )

        else:

            st.info(
                "Aucun log disponible."
            )


        st.divider()


        # ----------------------------------------------------
        # RAW STATE
        # ----------------------------------------------------

        with st.expander(
            "Afficher l'état LangGraph brut"
        ):

            st.json(
                result
            )


# ============================================================
# FOOTER
# ============================================================

st.write("")
st.write("")
st.caption("CardioVision IA · Prototype académique · Aide à la décision clinique uniquement")
