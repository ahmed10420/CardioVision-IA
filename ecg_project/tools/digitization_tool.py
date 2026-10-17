# ============================================================
# DIGITIZATION TOOL (CORRIGÉ)
#
# Role:
#   Input  : extractor output JSON OR ECG image OR signal
#   Output : digital ECG signal CSV
#
# It uses ECG-Digitiser:
#   python -m src.run.digitize -d input_dir -m MODEL_DIR -o output_dir
#
# CORRECTION IMPORTANTE PAR RAPPORT À LA VERSION PRÉCÉDENTE :
#   Le repo ECG-Digitiser précise dans son README que le nnU-Net OFFICIEL
#   (installé via `pip install nnunetv2` depuis PyPI) a un bug connu avec
#   les images RGB (IndexError sur normalization_schemes), et fournit son
#   propre FORK corrigé dans le sous-dossier nnUNet/ du repo (à installer
#   via `cd nnUNet && pip install .`).
#
#   Si ce fork n'est pas installé (par exemple après réinstallation d'un
#   environnement, changement de machine...), la digitisation peut soit
#   planter avec IndexError, soit tourner mais donner une segmentation de
#   moins bonne qualité selon la version de nnunetv2 réellement active.
#
#   Cet outil vérifie maintenant, une fois par processus, que le nnunetv2
#   chargé provient bien du fork du repo (pas de PyPI) et affiche un
#   avertissement clair sinon -> évite de découvrir le problème après coup
#   via des signaux reconstruits de mauvaise qualité sans explication.
#
# Final output:
#   CSV with shape (T, 12)
#   columns = I, II, III, aVR, aVL, aVF, V1...V6
# ============================================================

import json
import os
import sys
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

TOOL_DIR = Path(__file__).resolve().parent
BASE_DIR = TOOL_DIR.parent

# ECG-Digitiser core is now embedded inside this project.
# No external absolute Windows path is required.
REPO_DIR = BASE_DIR
MODEL_DIR = BASE_DIR / "models" / "M3"

LEADS = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]

TARGET_LEN = 1000

# Vérifié une seule fois par processus (pas à chaque appel) pour éviter
# le coût d'un import répété -> voir _fork_checked ci-dessous.
_fork_checked = False


# ============================================================
# VÉRIFICATION DU FORK nnU-Net (NOUVEAU)
# ============================================================

def check_nnunet_fork():
    """Vérifie que le nnunetv2 actuellement importable provient du fork
    du repo (REPO_DIR/nnUNet), pas de l'installation officielle PyPI qui
    plante sur les images RGB. N'installe rien automatiquement (ça reste
    une étape d'environnement à faire une fois, cf. README du repo :
    `cd nnUNet && pip install .`) -> avertit seulement, pour qu'un
    problème de digitisation silencieux ne soit pas confondu avec un bug
    de prétraitement d'image."""
    global _fork_checked
    if _fork_checked:
        return

    try:
        import nnunetv2
        nnunet_path = Path(nnunetv2.__file__).resolve()
        expected_fork_dir = (BASE_DIR / "nnUNet").resolve()

        if str(expected_fork_dir) in str(nnunet_path):
            print(f"✅ nnunetv2 chargé depuis le fork du repo : {nnunet_path}")
        else:
            print(f"⚠️  ATTENTION : nnunetv2 chargé depuis {nnunet_path}, "
                  f"qui ne semble PAS être le fork corrigé du repo "
                  f"({expected_fork_dir}). Si la digitisation échoue avec "
                  f"une IndexError sur des images RGB, ou si la qualité de "
                  f"segmentation semble anormalement mauvaise, installez le "
                  f"fork corrigé : cd \"{REPO_DIR / 'nnUNet'}\" && pip install .")
    except ImportError:
        print("⚠️ nnunetv2 introuvable dans l'environnement Python actuel — "
              "la digitisation va échouer. Installez-le (idéalement le fork "
              f"local : cd \"{BASE_DIR / 'nnUNet'}\" && pip install .)")

    _fork_checked = True


# ============================================================
# BASIC SIGNAL UTILS
# ============================================================

def resample_signal(signal: np.ndarray, target_len: int = TARGET_LEN) -> np.ndarray:
    """
    Input shape  : (12, T)
    Output shape : (12, target_len)
    """

    signal = np.asarray(signal, dtype=np.float32)

    if signal.ndim != 2:
        raise ValueError(f"Signal must be 2D, got {signal.shape}")

    n_leads, current_len = signal.shape

    if n_leads != 12:
        raise ValueError(f"Expected 12 leads, received {n_leads}")

    if current_len == target_len:
        return signal.astype(np.float32)

    old_t = np.linspace(0, 1, current_len)
    new_t = np.linspace(0, 1, target_len)

    output = np.zeros((12, target_len), dtype=np.float32)

    for lead_idx in range(12):
        y = signal[lead_idx]
        valid = ~np.isnan(y)

        if valid.sum() < 2:
            output[lead_idx] = 0.0
        else:
            output[lead_idx] = np.interp(new_t, old_t[valid], y[valid])

    return output.astype(np.float32)


def save_signal_to_csv(signal: np.ndarray, csv_path: Path) -> Path:
    """
    Save signal shape (12, T) to CSV shape (T, 12).
    """

    signal = np.asarray(signal, dtype=np.float32)

    if signal.shape[0] != 12:
        raise ValueError(f"Expected shape (12,T), got {signal.shape}")

    df = pd.DataFrame(signal.T, columns=LEADS)
    df.to_csv(csv_path, index=False)

    return csv_path


def load_signal_from_csv(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)

    if all(lead in df.columns for lead in LEADS):
        signal = df[LEADS].values.T.astype(np.float32)
    else:
        values = df.values.astype(np.float32)

        if values.shape[1] == 12:
            signal = values.T
        elif values.shape[0] == 12:
            signal = values
        else:
            raise ValueError(
                f"Invalid CSV shape {values.shape}. Expected (T,12) or (12,T)."
            )

    signal = np.nan_to_num(signal)

    return signal.astype(np.float32)


def load_signal_from_wfdb(header_path: Path, expected_duration_sec: float = 10.0,
                           duration_tolerance: float = 0.05) -> np.ndarray:
    """Charge le signal reconstruit par digitisation. Retourne (12, T).

    NOTE : le repo écrit le signal reconstruit à fs=500 (voir config.py :
    FREQUENCY=500, LONG_SIGNAL_LENGTH_SEC=10) -> 5000 échantillons pour
    10s, alors que le signal original PTB-XL utilisé par le classifieur
    est records100 (100Hz, 1000 échantillons pour 10s). Les DEUX
    représentent la même durée réelle (10s), donc le rééchantillonnage
    par fraction de durée (resample_signal, plus loin dans la chaîne)
    reste correct quelle que soit la fréquence native ici."""

    try:
        import wfdb
    except ImportError:
        raise ImportError(
            "wfdb is not installed. Install it using:\n"
            "pip install wfdb"
        )

    record_base = header_path.with_suffix("")
    record = wfdb.rdrecord(str(record_base))

    raw = record.p_signal.astype(np.float32)
    raw = np.nan_to_num(raw)

    actual_duration = raw.shape[0] / float(record.fs)
    if abs(actual_duration - expected_duration_sec) > expected_duration_sec * duration_tolerance:
        print(f"⚠️ Signal reconstruit de durée inhabituelle : {actual_duration:.2f}s "
              f"(attendu ~{expected_duration_sec}s) -> digitisation probablement "
              f"partiellement corrompue pour cet enregistrement.")

    sig_names = list(record.sig_name)

    signal = np.zeros((raw.shape[0], 12), dtype=np.float32)

    for i, lead in enumerate(LEADS):
        if lead in sig_names:
            signal[:, i] = raw[:, sig_names.index(lead)]
        else:
            signal[:, i] = 0.0

    return signal.T.astype(np.float32)


# ============================================================
# INPUT PARSING
# ============================================================

def parse_extractor_output(extracted_input_path: Path) -> Path:
    """
    Accepts:
        - extractor output JSON
        - image path
        - csv signal path
        - hea path

    Returns:
        path to prepared image/signal.
    """

    extracted_input_path = Path(extracted_input_path).resolve()

    if not extracted_input_path.exists():
        raise FileNotFoundError(f"Input not found: {extracted_input_path}")

    suffix = extracted_input_path.suffix.lower()

    if suffix == ".json":
        data = json.loads(extracted_input_path.read_text(encoding="utf-8"))

        if "prepared_image" in data:
            return Path(data["prepared_image"]).resolve()

        if "input_image" in data:
            return Path(data["input_image"]).resolve()

        raise ValueError(
            f"JSON does not contain prepared_image or input_image: {extracted_input_path}"
        )

    return extracted_input_path


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


def is_csv_file(path: Path) -> bool:
    return path.suffix.lower() == ".csv"


def is_wfdb_file(path: Path) -> bool:
    return path.suffix.lower() == ".hea"


# ============================================================
# ECG-DIGITISER EXECUTION
# ============================================================

def find_digitizer_output_signal(case_output_dir: Path) -> Path:
    """
    ECG-Digitiser usually produces WFDB .hea/.dat.
    We search first .hea, then .csv.
    """

    hea_files = [p for p in case_output_dir.rglob("*.hea") if "_input_image" not in str(p)]
    if len(hea_files) > 0:
        return hea_files[0]

    csv_files = list(case_output_dir.rglob("*.csv"))
    if len(csv_files) > 0:
        return csv_files[0]

    raise FileNotFoundError(
        f"No digitized signal found inside: {case_output_dir}"
    )


def run_ecg_digitiser(image_path: Path, case_output_dir: Path) -> Path:
    """
    Run ECG-Digitiser on one image.

    NOTE : image_path est censée être l'image DÉJÀ préparée par
    extractor_tool.py (RGB, résolution/contraste adaptés) -> on la copie
    telle quelle, sans re-prétraitement ici (évite un double traitement).
    """

    if not REPO_DIR.exists():
        raise FileNotFoundError(f"ECG-Digitiser repo not found: {REPO_DIR}")

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"M3 model dir not found: {MODEL_DIR}")

    check_nnunet_fork()

    shutil.rmtree(case_output_dir, ignore_errors=True)
    case_output_dir.mkdir(parents=True, exist_ok=True)

    input_dir = case_output_dir / "_input_image"
    input_dir.mkdir(parents=True, exist_ok=True)

    copied_image = input_dir / image_path.name
    shutil.copy2(image_path, copied_image)

    cmd = [
        sys.executable,
        "-m",
        "src.run.digitize",
        "-d", str(input_dir),
        "-m", str(MODEL_DIR),
        "-o", str(case_output_dir),
        "--show_image",
    ]

    print("\nCommande ECG-Digitiser:")
    print(" ".join(cmd))

    # Le pipeline principal peut être lancé avec un Python global qui ne connaît
    # pas le fork local nnUNet. On injecte donc explicitement le dossier local
    # dans PYTHONPATH pour que le subprocess retrouve nnunetv2.
    env = os.environ.copy()

    local_nnunet = (BASE_DIR / "nnUNet").resolve()
    existing_pythonpath = env.get("PYTHONPATH", "")

    env["PYTHONPATH"] = (
        str(local_nnunet)
        if not existing_pythonpath
        else str(local_nnunet) + os.pathsep + existing_pythonpath
    )

    result = subprocess.run(
        cmd,
        cwd=str(REPO_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    log_path = case_output_dir / "digitize_log.txt"
    log_path.write_text(
        "COMMAND:\n" + " ".join(cmd) +
        "\n\nPYTHON EXECUTABLE:\n" + sys.executable +
        "\n\nPYTHONPATH:\n" + env.get("PYTHONPATH", "") +
        "\n\nRETURN CODE:\n" + str(result.returncode) +
        "\n\nSTDOUT:\n" + result.stdout +
        "\n\nSTDERR:\n" + result.stderr,
        encoding="utf-8",
        errors="replace"
    )

    if result.returncode != 0:
        combined_log = result.stdout + "\n" + result.stderr

        if "Signal is empty" in combined_log:
            reason = "Signal is empty"
        elif "IndexError" in combined_log and "normalization_schemes" in combined_log:
            reason = ("IndexError sur normalization_schemes -> le nnunetv2 utilisé n'est "
                       "probablement PAS le fork corrigé du repo, voir l'avertissement "
                       "affiché au-dessus (check_nnunet_fork).")
        elif "IndexError" in combined_log:
            reason = "IndexError"
        elif "ModuleNotFoundError" in combined_log:
            reason = "ModuleNotFoundError"
        elif "FileNotFoundError" in combined_log:
            reason = "FileNotFoundError"
        else:
            reason = "Unknown digitization error"

        raise RuntimeError(
            f"ECG-Digitiser failed: {reason}. "
            f"Check log file: {log_path}"
        )

    produced_signal = find_digitizer_output_signal(case_output_dir)

    return produced_signal


# ============================================================
# MAIN DIGITIZATION TOOL
# ============================================================

def digitization_tool(extracted_input_path: str, output_dir: str = None) -> Path:
    """
    Function called by ecg_agent.py.

    Input:
        - extractor output JSON
        - ECG image
        - existing CSV signal
        - existing WFDB .hea

    Output:
        CSV signal path ready for classifier_tool.
    """

    extracted_input_path = Path(extracted_input_path).resolve()

    if output_dir is None:
        output_dir = BASE_DIR / "outputs"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    real_input_path = parse_extractor_output(extracted_input_path)

    print("\n" + "-" * 60)
    print("DIGITIZATION TOOL")
    print("-" * 60)
    print("Input received :", extracted_input_path)
    print("Real input     :", real_input_path)
    print("Output dir     :", output_dir)

    case_name = real_input_path.stem
    final_csv = output_dir / f"{case_name}_digitized_signal.csv"

    # --------------------------------------------------------
    # Case 1: input is image -> run ECG-Digitiser
    # --------------------------------------------------------
    if is_image_file(real_input_path):
        case_output_dir = output_dir / f"{case_name}_digitization"

        produced_signal = run_ecg_digitiser(
            image_path=real_input_path,
            case_output_dir=case_output_dir
        )

        print("Produced signal:", produced_signal)

        if produced_signal.suffix.lower() == ".hea":
            signal = load_signal_from_wfdb(produced_signal)

        elif produced_signal.suffix.lower() == ".csv":
            signal = load_signal_from_csv(produced_signal)

        else:
            raise ValueError(f"Unsupported produced signal: {produced_signal}")

        signal = resample_signal(signal, TARGET_LEN)
        save_signal_to_csv(signal, final_csv)

    # --------------------------------------------------------
    # Case 2: input is already CSV
    # --------------------------------------------------------
    elif is_csv_file(real_input_path):
        signal = load_signal_from_csv(real_input_path)
        signal = resample_signal(signal, TARGET_LEN)
        save_signal_to_csv(signal, final_csv)

    # --------------------------------------------------------
    # Case 3: input is WFDB
    # --------------------------------------------------------
    elif is_wfdb_file(real_input_path):
        signal = load_signal_from_wfdb(real_input_path)
        signal = resample_signal(signal, TARGET_LEN)
        save_signal_to_csv(signal, final_csv)

    else:
        raise ValueError(
            f"Unsupported input for digitization tool: {real_input_path.suffix}"
        )

    metadata = {
        "tool": "digitization_tool",
        "role": "ECG image/signal to digital ECG CSV",
        "input_received": str(extracted_input_path),
        "real_input_path": str(real_input_path),
        "output_signal_csv": str(final_csv),
        "output_shape": {
            "rows": TARGET_LEN,
            "columns": 12
        },
        "leads": LEADS,
        "next_step": "classifier_tool"
    }

    metadata_path = output_dir / f"{case_name}_digitization_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    print("Final digitized CSV:", final_csv)
    print("Digitization metadata:", metadata_path)

    return final_csv


# ============================================================
# TEST DIRECT
# ============================================================

if __name__ == "__main__":
    input_path = BASE_DIR / "inputs" / "ecg_image.png"

    result_csv = digitization_tool(
        extracted_input_path=input_path,
        output_dir=BASE_DIR / "outputs"
    )

    print("\nDigitized signal CSV:")
    print(result_csv)
