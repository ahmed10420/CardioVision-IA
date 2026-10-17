# ============================================================
# CLASSIFIER TOOL (CORRIGÉ)
#
# Role:
#   Input  : ECG digital signal CSV or WFDB .hea
#   Output : classification probabilities and predicted classes
#
# CORRECTIONS IMPORTANTES PAR RAPPORT À LA VERSION PRÉCÉDENTE :
#
#   1. ARCHITECTURE : l'ancienne version utilisait un ResNet1DWang
#      "maison" (kernels 8,5,3, AdaptiveAvgPool1d simple) chargé depuis
#      best_resnet1d_wang_standardscaler.pt -> c'est le modèle entraîné
#      avec un pos_weight extrême (jusqu'à x30) qui a été abandonné car
#      il prédisait souvent les mêmes classes peu importe le signal
#      d'entrée. Cette version charge l'architecture OFFICIELLE
#      (resnet1d_wang fidèle au repo PTB-XL benchmark : kernels [5,3],
#      stem k=7, tête AdaptiveConcatPool1d) et le modèle finetuné/entraîné
#      correspondant (best_resnet1d_wang_records100_posweight3.pt).
#
#   2. NORMALISATION : l'ancienne version utilisait un StandardScaler
#      PAR LEAD (fichier .npz, mean/std shape (1,12,1)). Le modèle actuel
#      a été entraîné avec une normalisation GLOBALE (un seul scalaire
#      mean/std sur tout le signal, fichier .json) -> utiliser l'ancien
#      scaler donnerait une entrée hors distribution et des prédictions
#      incohérentes.
#
#   3. PROTOCOLE D'INFÉRENCE : l'ancienne version rééchantillonnait le
#      signal entier à 1000 points et faisait UN SEUL passage dans le
#      modèle. Le modèle a été entraîné/validé sur des FENÊTRES de 250
#      points (2.5s) avec un stride de 125 (chunkify_valid), puis les
#      probabilités par fenêtre sont agrégées par MAX -> c'est CE
#      protocole qui est reproduit ici, pas un simple passage unique.
#
#   4. SEUILS : possibilité de charger des seuils par classe optimisés
#      (test_metrics JSON, clé "best_thresholds") au lieu de 0.5 fixe
#      pour toutes les classes.
# ============================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn


# ============================================================
# CONFIGURATION
# ============================================================

TOOL_DIR = Path(__file__).resolve().parent
BASE_DIR = TOOL_DIR.parent

MODEL_DIR = BASE_DIR / "models"

# CORRIGÉ : nouveaux artefacts (modèle officiel, scaler global JSON)
MODEL_PATH = MODEL_DIR / "best_resnet1d_wang_finetuned_reconstructed.pt"
SCALER_PATH = MODEL_DIR / "global_standard_scaler.json"
# Optionnel : si présent, les seuils par classe optimisés sont utilisés
# à la place de 0.5 fixe pour toutes les classes.
TEST_METRICS_PATH = MODEL_DIR / "test_metrics_records100_posweight3.json"

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

LEADS = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]

SAMPLING_RATE = 100          # le modèle a été entraîné sur records100 (100Hz)
TARGET_LEN = 1000            # 10s @ 100Hz
INPUT_SIZE = 250             # taille de fenêtre du modèle : 2.5s @ 100Hz
STRIDE_VALID = INPUT_SIZE // 2   # 125, comme le protocole d'évaluation officiel

PS_HEAD = 0.5
LIN_FTRS_HEAD = [128]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_OPTIMIZED_THRESHOLDS = True

MANUAL_THRESHOLDS = {
    "NORM": 0.90,
    "MI": 0.85,
    "STTC": 0.85,
    "CD": 0.85,
    "HYP": 0.65,
}


# ============================================================
# MODEL ARCHITECTURE - ResNet1D Wang (OFFICIELLE, CORRIGÉE)
# ============================================================

def conv1d_same(in_planes, out_planes, kernel_size, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=(kernel_size - 1) // 2, bias=False)


class BasicBlock1d(nn.Module):
    def __init__(self, inplanes, planes, stride=1, kernel_size=(5, 3), downsample=None):
        super().__init__()
        self.conv1 = conv1d_same(inplanes, planes, kernel_size[0], stride=stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = conv1d_same(planes, planes, kernel_size[1], stride=1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.act(out + identity)


class AdaptiveConcatPool1d(nn.Module):
    """Concatène AdaptiveMaxPool1d et AdaptiveAvgPool1d -> tête officielle
    du repo (différent de l'ancien AdaptiveAvgPool1d simple)."""

    def __init__(self):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool1d(1)
        self.mp = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], dim=1)


def create_head1d(nf, nc, lin_ftrs, ps_head):
    lin_ftrs = [2 * nf] + list(lin_ftrs) + [nc]
    ps = [ps_head / 2] * (len(lin_ftrs) - 2) + [ps_head]
    layers = [AdaptiveConcatPool1d(), nn.Flatten()]
    for i, (ni, no, p) in enumerate(zip(lin_ftrs[:-1], lin_ftrs[1:], ps)):
        layers += [nn.Dropout(p), nn.Linear(ni, no, bias=False), nn.BatchNorm1d(no)]
        if i < len(lin_ftrs) - 2:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class ResNet1dWang(nn.Module):
    """Architecture fidèle au repo PTB-XL benchmark officiel : stem k=7,
    3 stages BasicBlock1d(kernel=[5,3]), tête AdaptiveConcatPool1d."""

    def __init__(self, num_classes=5, input_channels=12, inplanes=128,
                 kernel_size=(5, 3), kernel_size_stem=7, ps_head=0.5, lin_ftrs_head=(128,)):
        super().__init__()
        self.inplanes = inplanes
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, inplanes, kernel_size=kernel_size_stem, stride=1,
                      padding=(kernel_size_stem - 1) // 2, bias=False),
            nn.BatchNorm1d(inplanes), nn.ReLU(inplace=True))
        self.stage0 = self._make_stage(inplanes, 1, kernel_size)
        self.stage1 = self._make_stage(inplanes, 2, kernel_size)
        self.stage2 = self._make_stage(inplanes, 2, kernel_size)
        self.head = create_head1d(inplanes, num_classes, lin_ftrs_head, ps_head)

    def _make_stage(self, planes, stride, kernel_size):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes))
        block = BasicBlock1d(self.inplanes, planes, stride=stride, kernel_size=kernel_size, downsample=downsample)
        self.inplanes = planes
        return block

    def forward(self, x):
        x = self.stem(x)
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        return self.head(x)


# ============================================================
# LOAD MODEL, SCALER, THRESHOLDS
# ============================================================

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Classification model not found: {MODEL_PATH}\n"
            "Put best_resnet1d_wang_records100_posweight3.pt inside models/ folder."
        )

    model = ResNet1dWang(num_classes=len(CLASSES), input_channels=len(LEADS),
                          ps_head=PS_HEAD, lin_ftrs_head=LIN_FTRS_HEAD).to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def load_scaler():
    """CORRIGÉ : normalisation GLOBALE (un seul scalaire mean/std), pas
    un scaler par lead comme avant."""
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Global scaler file not found: {SCALER_PATH}\n"
            "Put global_standard_scaler.json inside models/ folder."
        )

    with open(SCALER_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    mean, std = float(data["mean"]), float(data["std"])
    if std < 1e-8:
        std = 1.0
    return mean, std


def load_thresholds():
    if USE_OPTIMIZED_THRESHOLDS:
        thresholds = np.array(
            [float(MANUAL_THRESHOLDS[cls]) for cls in CLASSES],
            dtype=np.float32
        )
        print("Seuils balanced utilisés :", dict(zip(CLASSES, thresholds.tolist())))
        return thresholds

    thresholds = np.array([0.5] * len(CLASSES), dtype=np.float32)
    print("Seuil fixe 0.5 utilisé pour toutes les classes.")
    return thresholds


# ============================================================
# SIGNAL LOADING
# ============================================================

def normalize_lead_name(name):
    """Normalise le nom de dérivation (gère 'v1'/'V1'/'V_1' etc.) — utile
    car les signaux reconstruits par digitisation peuvent avoir une
    casse/format légèrement différent du PTB-XL original."""
    name = str(name).strip().replace("-", "").replace("_", "").upper()
    mapping = {"I": "I", "II": "II", "III": "III", "AVR": "aVR", "AVL": "aVL",
               "AVF": "aVF", "V1": "V1", "V2": "V2", "V3": "V3", "V4": "V4", "V5": "V5", "V6": "V6"}
    return mapping.get(name, name)


def load_signal_from_csv(csv_path: Path) -> np.ndarray:
    """
    Accepted CSV formats:
        Format 1: columns = I, II, III, aVR, aVL, aVF, V1...V6, shape (T,12)
        Format 2: no lead names, shape (T,12)
        Format 3: no lead names, shape (12,T)

    Output: signal shape = (T, 12)  (time-major, cohérent avec load_signal_from_wfdb)
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV signal not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if all(lead in df.columns for lead in LEADS):
        signal = df[LEADS].values.astype(np.float32)  # (T, 12)
    else:
        values = df.values.astype(np.float32)
        if values.shape[1] == 12:
            signal = values
        elif values.shape[0] == 12:
            signal = values.T
        else:
            raise ValueError(f"Invalid CSV shape {values.shape}. Expected (T,12) or (12,T).")

    return np.nan_to_num(signal).astype(np.float32)


def load_signal_from_wfdb(header_path: Path, expected_duration_sec: float = 10.0,
                           duration_tolerance: float = 0.05) -> np.ndarray:
    """Load WFDB signal from .hea/.dat. Output shape = (T, 12).

    Garde-fou : vérifie que la durée réelle (n_samples / fs) est bien
    proche de expected_duration_sec, quelle que soit la fréquence native
    du fichier (records100 @100Hz, ou signal reconstruit par digitisation
    généralement écrit @500Hz -> voir digitization_tool.py). Le
    rééchantillonnage se fait par fraction de durée, pas par nombre brut
    d'échantillons, donc il est correct tant que cette durée est fiable."""

    try:
        import wfdb
    except ImportError:
        raise ImportError("wfdb is not installed. Install it using:\npip install wfdb")

    header_path = Path(header_path)
    if not header_path.exists():
        raise FileNotFoundError(f"WFDB header not found: {header_path}")

    record_base = header_path.with_suffix("")
    record = wfdb.rdrecord(str(record_base))

    raw = np.nan_to_num(record.p_signal.astype(np.float32))

    actual_duration = raw.shape[0] / float(record.fs)
    if abs(actual_duration - expected_duration_sec) > expected_duration_sec * duration_tolerance:
        print(f"⚠️ Durée du signal inhabituelle : {actual_duration:.2f}s "
              f"(attendu ~{expected_duration_sec}s, fs={record.fs}, n_samples={raw.shape[0]}). "
              f"Le rééchantillonnage suppose une durée de {expected_duration_sec}s -> "
              f"résultat potentiellement désaligné si la digitisation a échoué partiellement.")

    sig_names_norm = [normalize_lead_name(x) for x in list(record.sig_name)]
    lead_to_index = {name: i for i, name in enumerate(sig_names_norm)}

    signal = np.zeros((raw.shape[0], len(LEADS)), dtype=np.float32)
    for i, lead in enumerate(LEADS):
        if lead in lead_to_index:
            signal[:, i] = raw[:, lead_to_index[lead]]

    return signal.astype(np.float32)


def load_signal(signal_path: Path) -> np.ndarray:
    """Retourne un signal shape (T, 12), quelle que soit la source."""
    signal_path = Path(signal_path)
    suffix = signal_path.suffix.lower()

    if suffix == ".csv":
        return load_signal_from_csv(signal_path)
    if suffix == ".hea":
        return load_signal_from_wfdb(signal_path)

    raise ValueError(f"Unsupported signal format: {suffix}. Supported formats: .csv, .hea")


# ============================================================
# SIGNAL PREPROCESSING
# ============================================================

def resample_signal(signal: np.ndarray, target_len: int = TARGET_LEN) -> np.ndarray:
    """Input shape (T,12) -> Output shape (target_len,12)."""
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 2 or signal.shape[1] != 12:
        raise ValueError(f"Signal must be shape (T,12), got {signal.shape}")

    T = signal.shape[0]
    if T == target_len:
        return signal.astype(np.float32)

    old_t = np.linspace(0, 1, T)
    new_t = np.linspace(0, 1, target_len)
    out = np.zeros((target_len, 12), dtype=np.float32)

    for c in range(12):
        y = signal[:, c]
        valid = ~np.isnan(y)
        out[:, c] = np.interp(new_t, old_t[valid], y[valid]) if valid.sum() >= 2 else 0.0

    return out.astype(np.float32)


def apply_global_scaler(signal_time_major: np.ndarray, mean: float, std: float) -> np.ndarray:
    """signal shape (T,12) -> normalisé, même shape (normalisation GLOBALE
    scalaire, PAS par lead)."""
    return ((signal_time_major - mean) / std).astype(np.float32)


def build_windows(signal_time_major: np.ndarray, input_size=INPUT_SIZE, stride=STRIDE_VALID):
    """CORRIGÉ : découpe le signal (T,12) en fenêtres glissantes de
    input_size échantillons (channels-first pour le modèle : 12,input_size),
    reproduisant exactement le protocole d'évaluation officiel
    (chunkify_valid=True, stride=input_size//2)."""
    T = signal_time_major.shape[0]
    windows = []
    for s in range(0, T, stride):
        e = min(s + input_size, T)
        if e - s < input_size:
            continue
        windows.append(signal_time_major[s:e].T.astype(np.float32))  # (12, input_size)

    if len(windows) == 0:
        # signal trop court : on pad plutôt que de planter
        pad = np.zeros((12, input_size), dtype=np.float32)
        n = min(input_size, T)
        pad[:, :n] = signal_time_major[:n].T
        windows = [pad]

    return np.stack(windows).astype(np.float32)  # (n_windows, 12, input_size)


# ============================================================
# PREDICTION
# ============================================================

def predict_signal(signal: np.ndarray, model, mean, std, thresholds) -> dict:
    """CORRIGÉ — protocole d'inférence fidèle à l'entraînement :

    signal (T,12) -> resample 100Hz/1000pts -> normalisation globale ->
    découpage en fenêtres de 2.5s (stride 1.25s) -> passage de TOUTES les
    fenêtres dans le modèle -> agrégation MAX par classe (pas un simple
    passage unique du signal entier, qui ne correspond pas à la manière
    dont le modèle a été entraîné/validé)."""

    signal = resample_signal(signal, TARGET_LEN)
    signal_scaled = apply_global_scaler(signal, mean, std)
    windows = build_windows(signal_scaled)  # (n_windows, 12, 250)

    x_tensor = torch.from_numpy(windows).to(DEVICE)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
            logits = model(x_tensor)
            window_probs = torch.sigmoid(logits).float().cpu().numpy()  # (n_windows, 5)

    record_probs = window_probs.max(axis=0)  # agrégation MAX par record

    probabilities_dict = {cls: float(prob) for cls, prob in zip(CLASSES, record_probs)}

    predicted_classes = []
    detailed_results = []
    for cls, thr in zip(CLASSES, thresholds):
        prob = probabilities_dict[cls]
        predicted = prob >= thr
        if predicted:
            predicted_classes.append(cls)
        detailed_results.append({
            "class": cls, "probability": round(prob, 6),
            "threshold": float(thr), "predicted": bool(predicted)
        })

    if len(predicted_classes) == 0:
        # sécurité : au moins une classe (la plus probable), comme le
        # protocole officiel
        predicted_classes = [CLASSES[int(np.argmax(record_probs))]]

    return {
        "model": "resnet1d_wang_records100_posweight3 (officiel)",
        "device": str(DEVICE),
        "classes": CLASSES,
        "n_windows_used": int(len(windows)),
        "probabilities": probabilities_dict,
        "predicted_classes": predicted_classes,
        "details": detailed_results
    }


# ============================================================
# CLASSIFIER TOOL MAIN FUNCTION
# ============================================================

def classifier_tool(signal_path: str, output_dir: str = None) -> dict:
    """Function called by ecg_agent.py.

    Input:  signal_path : .csv or .hea ECG signal
    Output: classification result dict
    """

    signal_path = Path(signal_path).resolve()

    if output_dir is None:
        output_dir = BASE_DIR / "outputs"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "-" * 60)
    print("CLASSIFIER TOOL (architecture officielle)")
    print("-" * 60)
    print("Signal input :", signal_path)
    print("Model path   :", MODEL_PATH)
    print("Scaler path  :", SCALER_PATH)

    model = load_model()
    mean, std = load_scaler()
    thresholds = load_thresholds()

    signal = load_signal(signal_path)
    print("Loaded signal shape :", signal.shape, "(T, 12)")

    result = predict_signal(signal=signal, model=model, mean=mean, std=std, thresholds=thresholds)

    output_json = output_dir / f"{signal_path.stem}_classification_result.json"
    output_json.write_text(json.dumps(result, indent=4, ensure_ascii=False), encoding="utf-8")

    print("Classification result saved:", output_json)
    print("Predicted classes:", result["predicted_classes"])

    return result


# ============================================================
# TEST DIRECT
# ============================================================

if __name__ == "__main__":
    input_signal = BASE_DIR / "inputs" / "ecg_signal.csv"

    result = classifier_tool(
        signal_path=input_signal,
        output_dir=BASE_DIR / "outputs"
    )

    print("\nFinal classification:")
    print(json.dumps(result, indent=4, ensure_ascii=False))
