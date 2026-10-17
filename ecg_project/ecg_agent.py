# ============================================================
# ECG AGENT - MAIN ORCHESTRATOR
#
# Logic required:
#
# if input is image:
#     call extractor tool
#     call digitization tool
# end if
#
# call classifier tool
#
# Output:
#   raw classification result:
#   - probabilities
#   - predicted classes
#   - signal path used by classifier
# ============================================================

import json
from pathlib import Path


# ============================================================
# IMPORT TOOLS
# ============================================================

from tools.extractor_tool import extractor_tool
from tools.digitization_tool import digitization_tool
from tools.classifier_tool import classifier_tool


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "inputs"
OUTPUT_DIR = BASE_DIR / "outputs"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
SIGNAL_EXTENSIONS = [".csv", ".hea"]


# ============================================================
# INPUT CHECKER
# ============================================================

def is_image(input_path: Path) -> bool:
    return input_path.suffix.lower() in IMAGE_EXTENSIONS


def is_signal(input_path: Path) -> bool:
    return input_path.suffix.lower() in SIGNAL_EXTENSIONS


# ============================================================
# ECG AGENT
# ============================================================

def ecg_agent(input_path: str) -> dict:
    """
    Main ECG agent.

    Input:
        - ECG image: png/jpg/jpeg/bmp/tiff
        - ECG signal: csv or hea

    Logic:
        if input is image:
            call extractor tool
            call digitization tool
        end if

        call classifier tool

    Output:
        classification result only.
        No medical report.
    """

    input_path = Path(input_path).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print("\n" + "=" * 70)
    print("ECG AGENT STARTED")
    print("=" * 70)
    print("Input:", input_path)

    # --------------------------------------------------------
    # Case 1: input is image
    # --------------------------------------------------------
    if is_image(input_path):
        print("\nInput type: ECG IMAGE")

        print("\n[1] Calling extractor tool...")
        extracted_output_path = extractor_tool(
            image_path=input_path,
            output_dir=OUTPUT_DIR
        )

        print("Extractor output:", extracted_output_path)

        print("\n[2] Calling digitization tool...")
        signal_path = digitization_tool(
            extracted_input_path=extracted_output_path,
            output_dir=OUTPUT_DIR
        )

        print("Digitized signal:", signal_path)

    # --------------------------------------------------------
    # Case 2: input is already a signal
    # --------------------------------------------------------
    elif is_signal(input_path):
        print("\nInput type: ECG DIGITAL SIGNAL")
        print("No extractor/digitization needed.")

        signal_path = input_path

    else:
        raise ValueError(
            f"Unsupported input format: {input_path.suffix}. "
            "Supported formats: image, .csv, .hea"
        )

    # --------------------------------------------------------
    # Final step: classifier tool
    # --------------------------------------------------------
    print("\n[3] Calling classifier tool...")

    classification_result = classifier_tool(
        signal_path=signal_path,
        output_dir=OUTPUT_DIR
    )

    print("\nClassification completed.")

    # --------------------------------------------------------
    # Final agent output
    # --------------------------------------------------------
    agent_output = {
        "input_path": str(input_path),
        "signal_path_used_by_classifier": str(signal_path),
        "classification_result": classification_result
    }

    output_json = OUTPUT_DIR / f"{input_path.stem}_agent_output.json"

    output_json.write_text(
        json.dumps(agent_output, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("ECG AGENT FINISHED")
    print("=" * 70)
    print("Output JSON:", output_json)

    return agent_output


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    # Example 1: ECG image
    input_file = INPUT_DIR / "ecg_image.png"

    # Example 2: ECG signal CSV
    # input_file = INPUT_DIR / "ecg_signal.csv"

    result = ecg_agent(input_file)

    print("\nFinal result:")
    print(json.dumps(result, indent=4, ensure_ascii=False))