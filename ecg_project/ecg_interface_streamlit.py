# ============================================================
# INTERFACE STREAMLIT PRO — ECG IMAGE / WFDB -> PROBABILITÉS + CLASSES
#
# À placer dans la racine du projet :
#   D:\ecg_project\ecg_interface_streamlit.py
#
# Entrées supportées :
#   1) Image ECG
#   2) Signal WFDB .hea + .dat
#
# Elle utilise ecg_agent.py :
#   image -> extractor_tool -> digitization_tool -> classifier_tool
#   .hea/.dat -> classifier_tool directement
# ============================================================

import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURATION GÉNÉRALE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "inputs"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


# ============================================================
# STREAMLIT CONFIG + CSS
# ============================================================

st.set_page_config(
    page_title="ECG Diagnostic Assistant",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #f4f8fb;
        --card: #ffffff;
        --primary: #0f5e7e;
        --primary-dark: #073b50;
        --accent: #20b2aa;
        --soft-blue: #eaf6fb;
        --soft-green: #eafaf3;
        --soft-red: #fff1f2;
        --text: #102a43;
        --muted: #627d98;
        --border: #d9e8f0;
        --shadow: 0 18px 45px rgba(15, 94, 126, 0.12);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    .stApp {
        background: linear-gradient(135deg, #eef7fb 0%, #f7fbff 45%, #fff7f9 100%) !important;
        color: var(--text) !important;
    }

    header, footer, #MainMenu, [data-testid="stSidebar"], [data-testid="stDecoration"] {
        display: none !important;
        visibility: hidden !important;
    }

    .block-container {
        max-width: 1180px !important;
        padding-top: 2.2rem !important;
        padding-bottom: 3rem !important;
    }

    .hero {
        background: linear-gradient(135deg, #0f5e7e 0%, #168aad 55%, #20b2aa 100%);
        border-radius: 26px;
        padding: 34px 38px;
        color: white;
        box-shadow: var(--shadow);
        margin-bottom: 26px;
    }

    .hero h1 {
        font-size: 42px;
        line-height: 1.1;
        margin: 0 0 12px 0;
        font-weight: 800;
        color: white !important;
    }

    .hero p {
        font-size: 17px;
        margin: 0;
        max-width: 820px;
        color: rgba(255,255,255,0.92) !important;
    }

    .chips {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin-top: 22px;
    }

    .chip {
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.32);
        color: white;
        padding: 9px 14px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 14px;
    }

    .section-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 26px 28px;
        box-shadow: 0 12px 35px rgba(16,42,67,0.06);
        margin: 22px 0;
    }

    .section-title {
        font-size: 25px;
        font-weight: 800;
        color: var(--primary-dark);
        margin-bottom: 6px;
    }

    .section-subtitle {
        color: var(--muted);
        font-size: 15px;
        margin-bottom: 18px;
    }

    .result-card {
        background: white;
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 22px;
        box-shadow: 0 10px 28px rgba(16,42,67,0.07);
        height: 100%;
    }

    .badge {
        display: inline-block;
        background: linear-gradient(135deg, #0f5e7e, #20b2aa);
        color: white;
        padding: 10px 16px;
        border-radius: 14px;
        font-weight: 800;
        margin: 5px 6px 5px 0;
        box-shadow: 0 8px 18px rgba(32,178,170,0.22);
    }

    .warning-box {
        background: #fff8e6;
        color: #7c5700;
        border: 1px solid #ffe2a3;
        border-radius: 16px;
        padding: 14px 16px;
        margin: 12px 0;
        font-weight: 600;
    }

    .success-box {
        background: var(--soft-green);
        color: #0b6b45;
        border: 1px solid #bcebd6;
        border-radius: 16px;
        padding: 14px 16px;
        margin: 12px 0;
        font-weight: 700;
    }

    /* Force les widgets Streamlit en clair */
    div[data-testid="stFileUploader"] section {
        background: #ffffff !important;
        border: 2px dashed #9fd3e8 !important;
        border-radius: 18px !important;
        color: var(--text) !important;
    }

    div[data-testid="stFileUploader"] section * {
        color: var(--text) !important;
    }

    div[data-testid="stFileUploader"] button {
        background: #0f5e7e !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #0f5e7e, #20b2aa) !important;
        color: white !important;
        border: none !important;
        border-radius: 15px !important;
        padding: 0.78rem 1.35rem !important;
        font-weight: 800 !important;
        font-size: 16px !important;
        box-shadow: 0 12px 28px rgba(15,94,126,0.22) !important;
    }

    .stButton > button:disabled {
        background: #c9d6df !important;
        color: #6b7c8a !important;
        box-shadow: none !important;
    }

    div[role="radiogroup"] {
        gap: 14px !important;
    }

    div[role="radiogroup"] label {
        background: #ffffff !important;
        border: 1px solid #cde4f0 !important;
        border-radius: 16px !important;
        padding: 12px 18px !important;
        box-shadow: 0 6px 18px rgba(16,42,67,0.05);
    }

    div[role="radiogroup"] label * {
        color: var(--primary-dark) !important;
        font-weight: 700 !important;
    }

    .stDataFrame, [data-testid="stDataFrame"] {
        background: white !important;
        border-radius: 16px !important;
    }

    code, pre {
        background: #eef7fb !important;
        color: #0b3b52 !important;
        border-radius: 12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def save_uploaded_file(uploaded_file, save_dir: Path) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(uploaded_file.name).name
    out_path = save_dir / safe_name
    with open(out_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return out_path.resolve()


def run_ecg_agent(input_path: Path) -> Dict[str, Any]:
    try:
        from ecg_agent import ecg_agent
    except Exception as e:
        raise ImportError(
            "Impossible d'importer ecg_agent.py. Place ce fichier dans la racine du projet.\n"
            f"Erreur originale : {e}"
        )
    return ecg_agent(str(input_path))


def normalize_agent_result(result: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(result, dict) and "classification_result" in result:
        return result["classification_result"]
    return result


def probabilities_to_dataframe(probabilities: Dict[str, float], predicted_classes: list[str]) -> pd.DataFrame:
    rows = []
    for cls in CLASSES:
        prob = float(probabilities.get(cls, 0.0))
        rows.append({
            "Classe": cls,
            "Probabilité": round(prob, 4),
            "Probabilité (%)": round(prob * 100, 2),
            "Résultat": "Prédite" if cls in predicted_classes else "Non prédite",
        })
    return pd.DataFrame(rows)


def show_json_download(result: Dict[str, Any], filename: str):
    data = json.dumps(result, indent=4, ensure_ascii=False)
    st.download_button(
        label="Télécharger le résultat JSON",
        data=data.encode("utf-8"),
        file_name=filename,
        mime="application/json",
    )


# ============================================================
# INTERFACE
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🫀 ECG Diagnostic Assistant</h1>
        <p>Importe une image ECG ou un signal WFDB, puis obtiens les probabilités par classe et la prédiction finale du modèle.</p>
        <div class="chips">
            <span class="chip">Image ECG</span>
            <span class="chip">WFDB .hea/.dat</span>
            <span class="chip">NORM · MI · STTC · CD · HYP</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Petit contrôle discret des fichiers modèle/scaler, sans sidebar
missing = []
model_candidates = [
    MODEL_DIR / "best_resnet1d_wang_finetuned_reconstructed.pt",
    MODEL_DIR / "best_resnet1d_wang_records100_posweight3_adam.pt",
    MODEL_DIR / "best_resnet1d_wang_records100_posweight3.pt",
]
if not any(p.exists() for p in model_candidates):
    missing.append("modèle .pt")
if not (MODEL_DIR / "global_standard_scaler.json").exists():
    missing.append("global_standard_scaler.json")

if missing:
    st.markdown(
        f"<div class='warning-box'>Attention : fichier(s) manquant(s) dans <b>models/</b> : {', '.join(missing)}</div>",
        unsafe_allow_html=True,
    )

st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>1. Choisir le type d’entrée</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-subtitle'>Choisis une image ECG ou un couple WFDB composé de deux fichiers : .hea et .dat.</div>",
    unsafe_allow_html=True,
)

input_type = st.radio(
    "Type d’entrée",
    ["Image ECG", "Signal WFDB .hea + .dat"],
    horizontal=True,
    label_visibility="collapsed",
)

upload_dir = INPUT_DIR / "ui_uploads"
selected_input_path: Optional[Path] = None

if input_type == "Image ECG":
    uploaded = st.file_uploader(
        "Importer une image ECG",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        label_visibility="visible",
    )

    if uploaded is not None:
        selected_input_path = save_uploaded_file(uploaded, upload_dir)
        left, right = st.columns([1.3, 1])
        with left:
            st.image(str(selected_input_path), caption="Image ECG importée", use_container_width=True)
        with right:
            st.markdown("<div class='success-box'>Image chargée avec succès.</div>", unsafe_allow_html=True)
            st.write("Fichier utilisé :")
            st.code(str(selected_input_path))

else:
    c1, c2 = st.columns(2)
    with c1:
        uploaded_hea = st.file_uploader("Importer le fichier .hea", type=["hea"])
    with c2:
        uploaded_dat = st.file_uploader("Importer le fichier .dat associé", type=["dat"])

    dat_path = None
    if uploaded_hea is not None:
        selected_input_path = save_uploaded_file(uploaded_hea, upload_dir)
    if uploaded_dat is not None:
        dat_path = save_uploaded_file(uploaded_dat, upload_dir)

    if uploaded_hea is not None and uploaded_dat is not None:
        st.markdown("<div class='success-box'>Fichiers WFDB chargés avec succès.</div>", unsafe_allow_html=True)
        st.write("Fichier .hea :")
        st.code(str(selected_input_path))
        st.write("Fichier .dat :")
        st.code(str(dat_path))

        if selected_input_path.with_suffix("").name != dat_path.with_suffix("").name:
            st.markdown(
                "<div class='warning-box'>Attention : le .hea et le .dat n'ont pas le même nom de base. Exemple correct : 01036_lr.hea et 01036_lr.dat.</div>",
                unsafe_allow_html=True,
            )

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>2. Lancer la prédiction</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-subtitle'>Le résultat affichera les probabilités, les classes prédites et la sortie JSON complète.</div>",
    unsafe_allow_html=True,
)

run_button = st.button("Lancer la classification", type="primary", disabled=(selected_input_path is None))
st.markdown("</div>", unsafe_allow_html=True)

if run_button:
    if selected_input_path is None:
        st.error("Aucun fichier d'entrée sélectionné.")
        st.stop()

    if selected_input_path.suffix.lower() == ".hea" and not selected_input_path.with_suffix(".dat").exists():
        st.error("Le fichier .dat associé est introuvable dans le même dossier que le .hea.")
        st.stop()

    with st.spinner("Traitement en cours..."):
        try:
            agent_result = run_ecg_agent(selected_input_path)
            classification = normalize_agent_result(agent_result)
        except Exception as e:
            st.error("Erreur pendant l'exécution de l'agent.")
            st.exception(e)
            st.stop()

    probabilities = classification.get("probabilities", {})
    predicted_classes = classification.get("predicted_classes", [])
    details = classification.get("details", [])

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>3. Résultat de classification</div>", unsafe_allow_html=True)
    st.markdown("<div class='success-box'>Classification terminée avec succès.</div>", unsafe_allow_html=True)

    r1, r2 = st.columns([1, 2])

    with r1:
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.markdown("### Classes prédites")
        if predicted_classes:
            badges = "".join([f"<span class='badge'>{cls}</span>" for cls in predicted_classes])
            st.markdown(badges, unsafe_allow_html=True)
        else:
            st.warning("Aucune classe prédite.")

        if "n_windows_used" in classification:
            st.markdown("---")
            st.write("Fenêtres utilisées :")
            st.code(str(classification["n_windows_used"]))
        st.markdown("</div>", unsafe_allow_html=True)

    with r2:
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.markdown("### Probabilités par classe")
        prob_df = probabilities_to_dataframe(probabilities, predicted_classes)
        st.dataframe(prob_df, use_container_width=True, hide_index=True)
        st.bar_chart(prob_df.set_index("Classe")[["Probabilité"]])
        st.markdown("</div>", unsafe_allow_html=True)

    if details:
        st.markdown("### Détails avec seuils")
        detail_df = pd.DataFrame(details)
        st.dataframe(detail_df, use_container_width=True, hide_index=True)

    st.markdown("### Sortie JSON complète")
    st.json(agent_result)
    show_json_download(agent_result, filename=f"{selected_input_path.stem}_agent_output.json")
    st.markdown("</div>", unsafe_allow_html=True)
