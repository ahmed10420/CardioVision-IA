import json
import sys
import traceback

from orchestrator.langgraph_orchestrator import run_agentic_rag
import sys

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

def make_json_safe(obj):
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, dict):
        return {
            str(k): make_json_safe(v)
            for k, v in obj.items()
        }

    if isinstance(obj, (list, tuple, set)):
        return [
            make_json_safe(v)
            for v in obj
        ]

    # LangChain Document
    if hasattr(obj, "page_content"):
        return {
            "page_content": str(
                getattr(obj, "page_content", "")
            ),
            "metadata": make_json_safe(
                getattr(obj, "metadata", {})
            ),
        }

    return str(obj)


def main():
    if len(sys.argv) < 2:
        raise ValueError(
            "Fichier de requête JSON manquant."
        )

    input_path = sys.argv[1]

    with open(
        input_path,
        "r",
        encoding="utf-8",
    ) as f:
        payload = json.load(f)

    result = run_agentic_rag(
        question=payload["question"],
        ecg_input_path=payload.get(
            "ecg_input_path",
            "",
        ),
        test_force_improve=False,
        ecg_required=False,
    )

    safe_result = make_json_safe(
        result
    )

    print(
        "__CARDIOVISION_RESULT__"
        + json.dumps(
            safe_result,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    try:
        main()

    except Exception:
        print(
            "__CARDIOVISION_ERROR__"
            + traceback.format_exc()
        )
        sys.exit(1)