"""Verrou d'intégrité des 30 réponses de référence de l'évaluation."""

from __future__ import annotations

import hashlib
import json


FROZEN_REFERENCE_COUNT = 30
FROZEN_REFERENCES_SHA256 = (
    "4151c5642899a4513664e37e3234098079935d3e4305bc755d740f9068c2b6c3"
)


def reference_answers_sha256(questions: list[dict]) -> str:
    """Calcule une empreinte stable des couples (id, réponse de référence)."""

    payload = json.dumps(
        [
            (item.get("id"), item.get("reference_answer", ""))
            for item in questions
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_frozen_references(questions: list[dict]) -> None:
    """Refuse une évaluation si le ground truth figé a été modifié."""

    digest = reference_answers_sha256(questions)
    if (
        len(questions) != FROZEN_REFERENCE_COUNT
        or digest != FROZEN_REFERENCES_SHA256
    ):
        raise RuntimeError(
            "Les réponses de référence figées ont été modifiées. "
            "Toute évolution du ground truth doit être versionnée et auditée "
            "explicitement avant de mettre à jour son empreinte."
        )
