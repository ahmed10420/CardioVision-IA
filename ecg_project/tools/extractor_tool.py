# ============================================================
# EXTRACTOR TOOL - PRÉTRAITEMENT ADAPTATIF (CORRIGÉ)
#
# Role:
#   Input  : ECG image
#   Output : prepared ECG image + extractor JSON
#
# CORRECTION IMPORTANTE PAR RAPPORT À LA VERSION PRÉCÉDENTE :
#   L'ancienne version appliquait un prétraitement FIXE et agressif à
#   TOUTE image, peu importe sa qualité de départ :
#       CONTRAST_FACTOR = 1.45 (fixe)
#       SHARPNESS_FACTOR = 1.25 (fixe)
#       + un filtre ImageFilter.SHARPEN supplémentaire inconditionnel
#   -> Ce réglage sursature les images déjà propres et n'aide pas
#      forcément les images vraiment dégradées (c'est le problème que
#      l'on cherche justement à éviter).
#
#   Cette version reprend le prétraitement ADAPTATIF validé (corrélation
#   0.86 confirmée en test local) : les paramètres de contraste, netteté,
#   autocontrast et filtre médian sont choisis en fonction de la qualité
#   RÉELLE de l'image d'entrée (contraste mesuré, ratio de pixels sombres),
#   avec un garde-fou anti-saturation en sortie.
#
# Agent logic:
#   image -> extractor_tool -> digitization_tool -> classifier_tool
# ============================================================

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter


# ============================================================
# CONFIGURATION
# ============================================================

TOOL_DIR = Path(__file__).resolve().parent
BASE_DIR = TOOL_DIR.parent

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

# Résolution cible validée (corrélation 0.86 confirmée avec cette valeur)
TARGET_WIDTH = 2156
MAX_HEIGHT = 2200
FORCE_TARGET_WIDTH = True

# Garde-fou anti-saturation : si après prétraitement l'image dépasse ces
# seuils, on adoucit automatiquement (voir reduce_saturation_if_needed).
MAX_SAFE_CONTRAST = 235
MAX_SAFE_DARK_RATIO = 0.28


# ============================================================
# IMAGE QUALITY ANALYSIS
# ============================================================

def analyze_image_quality(img: Image.Image) -> dict:
    """Analyse la qualité visuelle de l'image (contraste, luminosité,
    netteté des contours) -> sert à choisir les paramètres adaptatifs
    ET à valider qu'on ne sur-corrige pas une image déjà bonne."""

    rgb = img.convert("RGB")
    gray = np.array(ImageOps.grayscale(rgb)).astype(np.float32)

    w, h = rgb.size

    p5 = float(np.percentile(gray, 5))
    p95 = float(np.percentile(gray, 95))

    dark_ratio = float(np.mean(gray < 80))
    very_dark_ratio = float(np.mean(gray < 40))

    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    edge_score = float(np.mean(gx) + np.mean(gy))

    quality = {
        "width": int(w),
        "height": int(h),
        "aspect_ratio": float(w / h),
        "mean_intensity": float(np.mean(gray)),
        "std_intensity": float(np.std(gray)),
        "min_intensity": float(np.min(gray)),
        "max_intensity": float(np.max(gray)),
        "contrast_p95_p5": float(p95 - p5),
        "dark_ratio": dark_ratio,
        "very_dark_ratio": very_dark_ratio,
        "edge_score": edge_score,
        "warnings": []
    }

    if w < 1200:
        quality["warnings"].append("resolution_width_low")
    if quality["contrast_p95_p5"] < 100:
        quality["warnings"].append("contrast_low")
    if dark_ratio < 0.02:
        quality["warnings"].append("ecg_trace_pixels_low")
    if quality["contrast_p95_p5"] >= 250:
        quality["warnings"].append("contrast_saturated")
    if dark_ratio > 0.30:
        quality["warnings"].append("too_many_dark_pixels_possible_noise")

    return quality


# ============================================================
# PARAMÈTRES ADAPTATIFS — CŒUR DE LA CORRECTION
# ============================================================

def choose_adaptive_parameters(q: dict) -> dict:
    """CORRIGÉ : au lieu d'un contraste/netteté FIXE appliqué à toute
    image, les paramètres sont choisis selon la qualité mesurée de
    l'image d'entrée -> évite de sursaturer une image déjà propre et
    d'amplifier le bruit d'une image déjà dégradée."""

    width, contrast, dark_ratio = q["width"], q["contrast_p95_p5"], q["dark_ratio"]

    if contrast < 80:
        autocontrast_cutoff = 1.0
    elif contrast < 120:
        autocontrast_cutoff = 0.5
    else:
        autocontrast_cutoff = 0.0

    if contrast < 70:
        contrast_factor = 1.35
    elif contrast < 100:
        contrast_factor = 1.20
    elif contrast < 130:
        contrast_factor = 1.08
    elif contrast < 170:
        contrast_factor = 1.00
    else:
        contrast_factor = 0.95  # image déjà très contrastée -> on adoucit

    if dark_ratio > 0.10:
        contrast_factor = min(contrast_factor, 1.00)
    if dark_ratio > 0.18:
        contrast_factor = min(contrast_factor, 0.95)
    if width < 900:
        contrast_factor = min(contrast_factor, 1.08)

    if contrast < 90 and dark_ratio < 0.06:
        sharpness_factor = 1.15
    elif contrast < 130:
        sharpness_factor = 1.08
    else:
        sharpness_factor = 1.00
    if dark_ratio > 0.10:
        sharpness_factor = min(sharpness_factor, 1.02)

    use_median_filter = dark_ratio > 0.12

    return {
        "autocontrast_cutoff": float(autocontrast_cutoff),
        "contrast_factor": float(contrast_factor),
        "sharpness_factor": float(sharpness_factor),
        "use_median_filter": bool(use_median_filter),
    }


def reduce_saturation_if_needed(img: Image.Image) -> tuple[Image.Image, dict]:
    """Garde-fou : si le prétraitement a rendu l'image trop contrastée
    ou trop sombre, on adoucit automatiquement plutôt que de laisser
    passer une image sur-corrigée vers le digitiseur."""

    q = analyze_image_quality(img)
    adjusted = False
    reasons = []

    if q["contrast_p95_p5"] > MAX_SAFE_CONTRAST:
        adjusted = True
        reasons.append("contrast_after_too_high")
    if q["dark_ratio"] > MAX_SAFE_DARK_RATIO:
        adjusted = True
        reasons.append("dark_ratio_after_too_high")

    if not adjusted:
        return img, {"saturation_adjusted": False, "adjustment_reason": [], "quality_after_adjustment": q}

    img2 = ImageEnhance.Contrast(img).enhance(0.85)
    img2 = ImageEnhance.Sharpness(img2).enhance(0.95)
    q2 = analyze_image_quality(img2)

    return img2, {
        "saturation_adjusted": True, "adjustment_reason": reasons,
        "quality_before_adjustment": q, "quality_after_adjustment": q2,
    }


# ============================================================
# PREPROCESSING FUNCTIONS
# ============================================================

def resize_keep_aspect(img: Image.Image, target_width: int = TARGET_WIDTH) -> Image.Image:
    w, h = img.size

    if not FORCE_TARGET_WIDTH and w >= 1200:
        return img
    if w == target_width:
        return img

    scale = target_width / float(w)
    new_w = target_width
    new_h = int(round(h * scale))

    if new_h > MAX_HEIGHT:
        scale = MAX_HEIGHT / float(h)
        new_w = int(round(w * scale))
        new_h = MAX_HEIGHT

    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def add_white_background_if_needed(img: Image.Image) -> Image.Image:
    """Si l'image contient un canal alpha, on la place sur fond blanc.
    L'image reste en RGB (3 canaux) — NE PAS convertir en niveaux de gris
    ici : le fork nnU-Net corrigé du repo attend du RGB (voir
    digitization_tool.py pour la vérification du fork)."""

    if img.mode in ("RGBA", "LA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        return background

    return img.convert("RGB")


def preprocess_ecg_image(input_path: Path, output_path: Path) -> tuple[Path, dict]:
    """Prétraitement ADAPTATIF avant ECG-Digitiser (version corrigée) :

      1. Ouvrir image + corriger orientation EXIF
      2. Analyser la qualité RÉELLE de l'image
      3. Choisir les paramètres adaptatifs en fonction de cette qualité
      4. Fond blanc si canal alpha + conversion RGB
      5. Redimensionnement vers la largeur cible
      6. Filtre médian SEULEMENT si l'image est bruitée/sombre
      7. Autocontrast SEULEMENT si nécessaire
      8. Contraste et netteté ADAPTATIFS (pas fixes)
      9. Garde-fou anti-saturation
     10. Sauvegarde PNG
    """

    original_img = Image.open(input_path)
    original_img = ImageOps.exif_transpose(original_img)

    original_quality = analyze_image_quality(original_img)
    params = choose_adaptive_parameters(original_quality)

    img = add_white_background_if_needed(original_img)
    img = resize_keep_aspect(img, TARGET_WIDTH)

    if params["use_median_filter"]:
        img = img.filter(ImageFilter.MedianFilter(size=3))

    if params["autocontrast_cutoff"] > 0:
        img = ImageOps.autocontrast(img, cutoff=params["autocontrast_cutoff"])

    img = ImageEnhance.Contrast(img).enhance(params["contrast_factor"])
    img = ImageEnhance.Sharpness(img).enhance(params["sharpness_factor"])
    # NOTE : pas de ImageFilter.SHARPEN supplémentaire ici (contrairement à
    # l'ancienne version) -> ce filtre fixe en plus de l'enhance ci-dessus
    # amplifiait le bruit sur les images déjà correctement nettes.

    img, saturation_info = reduce_saturation_if_needed(img)
    processed_quality = analyze_image_quality(img)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)

    preprocessing_info = {
        "target_width": TARGET_WIDTH,
        "max_height": MAX_HEIGHT,
        "force_target_width": FORCE_TARGET_WIDTH,
        "adaptive_parameters_used": params,
        "saturation_control": saturation_info,
        "original_quality": original_quality,
        "processed_quality": processed_quality,
        "improvement": {
            "width_before": original_quality["width"],
            "width_after": processed_quality["width"],
            "height_before": original_quality["height"],
            "height_after": processed_quality["height"],
            "contrast_before": original_quality["contrast_p95_p5"],
            "contrast_after": processed_quality["contrast_p95_p5"],
            "dark_ratio_before": original_quality["dark_ratio"],
            "dark_ratio_after": processed_quality["dark_ratio"],
            "edge_score_before": original_quality["edge_score"],
            "edge_score_after": processed_quality["edge_score"],
        }
    }

    return output_path, preprocessing_info


# ============================================================
# MAIN EXTRACTOR TOOL
# ============================================================

def extractor_tool(image_path: str, output_dir: str = None) -> Path:
    """Function called by ecg_agent.py.

    Input:
        ECG image path

    Output:
        extractor JSON containing prepared image path
    """

    image_path = Path(image_path).resolve()

    if output_dir is None:
        output_dir = BASE_DIR / "outputs"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        raise FileNotFoundError(f"Image introuvable : {image_path}")

    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Format image non supporté : {image_path.suffix}")

    case_name = image_path.stem

    extractor_dir = output_dir / f"{case_name}_extractor"
    extractor_dir.mkdir(parents=True, exist_ok=True)

    prepared_image = extractor_dir / f"{case_name}_prepared.png"
    metadata_path = extractor_dir / f"{case_name}_extractor_output.json"

    prepared_image, preprocessing_info = preprocess_ecg_image(
        input_path=image_path,
        output_path=prepared_image
    )

    metadata = {
        "tool": "extractor_tool",
        "role": "adaptive ECG image preparation for digitization",
        "input_image": str(image_path),
        "prepared_image": str(prepared_image),
        "preprocessing": preprocessing_info,
        "next_step": "digitization_tool"
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n" + "-" * 60)
    print("EXTRACTOR TOOL - PRÉTRAITEMENT ADAPTATIF")
    print("-" * 60)
    print("Input image       :", image_path)
    print("Prepared image    :", prepared_image)
    print("Metadata          :", metadata_path)

    oq, pq = preprocessing_info["original_quality"], preprocessing_info["processed_quality"]
    params = preprocessing_info["adaptive_parameters_used"]

    print("\nImage quality before:")
    print("  size      :", oq["width"], "x", oq["height"])
    print("  contrast  :", round(oq["contrast_p95_p5"], 2))
    print("  dark_ratio:", round(oq["dark_ratio"], 4))

    print("\nParamètres adaptatifs choisis :")
    print("  autocontrast_cutoff:", params["autocontrast_cutoff"])
    print("  contrast_factor    :", params["contrast_factor"])
    print("  sharpness_factor   :", params["sharpness_factor"])
    print("  median_filter      :", params["use_median_filter"])

    print("\nImage quality after:")
    print("  size      :", pq["width"], "x", pq["height"])
    print("  contrast  :", round(pq["contrast_p95_p5"], 2))
    print("  dark_ratio:", round(pq["dark_ratio"], 4))

    if preprocessing_info["saturation_control"]["saturation_adjusted"]:
        print("\n⚠️ Anti-saturation activé :", preprocessing_info["saturation_control"]["adjustment_reason"])

    if len(pq["warnings"]) > 0:
        print("\nWarnings après prétraitement:")
        for w in pq["warnings"]:
            print("  -", w)

    return metadata_path


# ============================================================
# TEST DIRECT
# ============================================================

if __name__ == "__main__":
    input_image = BASE_DIR / "inputs" / "ecg_image.png"

    result = extractor_tool(
        image_path=input_image,
        output_dir=BASE_DIR / "outputs"
    )

    print("\nExtractor output JSON:")
    print(result)
